#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import glob
import importlib
import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Sequence

import av
import pyarrow as pa
import torch
import torchvision
from datasets.features.features import register_feature
from collections import OrderedDict
from PIL import Image

_TORCHCODEC_AVAILABLE: bool | None = None
_DECODER_CACHE_MAX_SIZE = 32
_DECODER_CACHE_PID: int | None = None
_DECODER_CACHE: OrderedDict[tuple[str, str], Any] = OrderedDict()


def _get_torchcodec_decoder(video_path: Path | str, device: str):
    """Return a fork-safe, per-worker bounded LRU TorchCodec VideoDecoder."""
    global _TORCHCODEC_AVAILABLE, _DECODER_CACHE_PID
    if _TORCHCODEC_AVAILABLE is None:
        _TORCHCODEC_AVAILABLE = importlib.util.find_spec("torchcodec") is not None
    if not _TORCHCODEC_AVAILABLE:
        raise ImportError("torchcodec is required but not available.")

    pid = os.getpid()
    if _DECODER_CACHE_PID != pid:
        _DECODER_CACHE.clear()
        _DECODER_CACHE_PID = pid

    key = (os.fspath(video_path), device)
    decoder = _DECODER_CACHE.get(key)
    if decoder is not None:
        _DECODER_CACHE.move_to_end(key)
        return decoder

    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(video_path, device=device, seek_mode="approximate")
    _DECODER_CACHE[key] = decoder
    if len(_DECODER_CACHE) > _DECODER_CACHE_MAX_SIZE:
        _DECODER_CACHE.popitem(last=False)
    return decoder


def _timestamps_to_frame_indices(
    timestamps: Sequence[float],
    average_fps: float,
    num_frames: int | None,
    begin_stream_seconds: float = 0.0,
) -> tuple[list[int], list[float]]:
    """Convert timestamps to clamped frame indices while preserving original query timestamps for tolerance checks."""
    query_timestamps = [float(timestamp) for timestamp in timestamps]
    frame_indices = [round((timestamp - begin_stream_seconds) * average_fps) for timestamp in query_timestamps]
    if num_frames is not None and int(num_frames) > 0:
        max_frame_index = int(num_frames) - 1
        frame_indices = [max(0, min(index, max_frame_index)) for index in frame_indices]
    else:
        frame_indices = [max(0, index) for index in frame_indices]
    return frame_indices, query_timestamps


def get_safe_default_codec():
    if importlib.util.find_spec("torchcodec"):
        return "torchcodec"
    else:
        logging.warning("'torchcodec' is not available in your platform, falling back to 'pyav' as a default decoder")
        return "pyav"


def decode_video_frames(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
) -> torch.Tensor:
    """
    Decodes video frames using the specified backend.

    Args:
        video_path (Path): Path to the video file.
        timestamps (list[float]): List of timestamps to extract frames.
        tolerance_s (float): Allowed deviation in seconds for frame retrieval.
        backend (str, optional): Backend to use for decoding. Defaults to "torchcodec" when available in the platform; otherwise, defaults to "pyav"..

    Returns:
        torch.Tensor: Decoded frames.

    Currently supports torchcodec on cpu and pyav.
    """
    if backend is None:
        backend = get_safe_default_codec()
    if backend == "torchcodec":
        return decode_video_frames_torchcodec(video_path, timestamps, tolerance_s)
    elif backend == "pyav":
        return decode_video_frames_pyav(video_path, timestamps, tolerance_s)
    elif backend == "video_reader":
        return decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend)
    else:
        raise ValueError(f"Unsupported video backend: {backend}")


def decode_video_frames_pyav(video_path, timestamps, tolerance_s):
    """Decode with PyAV directly, converting only frames near queried timestamps to RGB."""
    if not timestamps:
        raise ValueError("Video decoding requires at least one timestamp.")
    loaded_frames, loaded_timestamps = [], []
    query_floats = [float(ts) for ts in timestamps]
    min_q = min(query_floats)
    max_q = max(query_floats)
    # Only run costly rgb24 conversion on frames close to any queried timestamp.
    window = max(float(tolerance_s) * 4.0, 0.25)
    with av.open(os.fspath(video_path)) as container:
        stream = container.streams.video[0]
        # Seeking to a preceding keyframe preserves accurate frame selection.
        container.seek(int(min_q / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * stream.time_base)
            if any(abs(timestamp - q) <= window for q in query_floats):
                loaded_frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1))
                loaded_timestamps.append(timestamp)
            if timestamp >= max_q + tolerance_s:
                break
    if not loaded_frames:
        raise ValueError(f"No video frames decoded from {video_path} near timestamps {query_floats}.")
    query = torch.tensor(query_floats, dtype=torch.float64)
    loaded = torch.tensor(loaded_timestamps, dtype=torch.float64)
    distances, indices = (query[:, None] - loaded[None, :]).abs().min(dim=1)
    if not (distances < tolerance_s).all():
        raise ValueError(
            f"Video timestamps exceed tolerance_s={tolerance_s} for {video_path}: "
            f"requested={query.tolist()}, nearest={loaded[indices].tolist()}."
        )
    return torch.stack([loaded_frames[index] for index in indices]).contiguous()


def decode_video_frames_torchvision(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated to the requested timestamps of a video

    The backend can be either "pyav" (default) or "video_reader".
    "video_reader" requires installing torchvision from source, see:
    https://github.com/pytorch/vision/blob/main/torchvision/csrc/io/decoder/gpu/README.rst
    (note that you need to compile against ffmpeg<4.3)

    While both use cpu, "video_reader" is supposedly faster than "pyav" but requires additional setup.
    For more info on video decoding, see `benchmark/video/README.md`

    See torchvision doc for more info on these two backends:
    https://pytorch.org/vision/0.18/index.html?highlight=backend#torchvision.set_video_backend

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """
    video_path = str(video_path)

    # set backend
    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True  # pyav doesn't support accurate seek

    # set a video stream reader
    # TODO(rcadene): also load audio stream at the same time
    reader = torchvision.io.VideoReader(video_path, "video")
    try:
        first_ts = min(timestamps)
        last_ts = max(timestamps)
        reader.seek(first_ts, keyframes_only=keyframes_only)

        loaded_frames = []
        loaded_ts = []
        for frame in reader:
            current_ts = frame["pts"]
            if log_loaded_timestamps:
                logging.info(f"frame loaded at timestamp={current_ts:.4f}")
            loaded_frames.append(frame["data"])
            loaded_ts.append(current_ts)
            if current_ts >= last_ts:
                break
    finally:
        if backend == "pyav" and getattr(reader, "container", None) is not None:
            reader.container.close()
        reader = None

    query_ts = torch.tensor(timestamps, dtype=torch.float64)
    loaded_ts = torch.tensor(loaded_ts, dtype=torch.float64)

    # compute distances between each query timestamp and timestamps of all loaded frames
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
        f"\nbackend: {backend}"
    )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # Keep image tensors in uint8 [0,255]. Qwen-VL image processors rescale
    # internally; returning float [0,1] here would be rescaled a second time.
    closest_frames = closest_frames.type(torch.uint8)

    assert len(timestamps) == len(closest_frames)
    return closest_frames


def decode_video_frames_torchcodec(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    device: str = "cpu",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated with the requested timestamps of a video using torchcodec.

    Note: Setting device="cuda" outside the main process, e.g. in data loader workers, will lead to CUDA initialization errors.

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """

    decoder = _get_torchcodec_decoder(video_path, device=device)
    metadata = decoder.metadata
    average_fps = metadata.average_fps
    begin_stream_seconds = float(getattr(metadata, "begin_stream_seconds", 0.0) or 0.0)
    num_frames = getattr(metadata, "num_frames", None)
    frame_indices, query_timestamps = _timestamps_to_frame_indices(
        timestamps,
        average_fps,
        num_frames,
        begin_stream_seconds=begin_stream_seconds,
    )

    # Deduplicate clamped frame indices (common at episode starts with history
    # offsets) so repeated frames are decoded once instead of multiple times.
    unique_indices = list(dict.fromkeys(frame_indices))
    frames_batch = decoder.get_frames_at(indices=unique_indices)

    loaded_frames = []
    loaded_ts = []
    for frame, pts in zip(frames_batch.data, frames_batch.pts_seconds, strict=False):
        loaded_frames.append(frame)
        loaded_ts.append(pts.item())
        if log_loaded_timestamps:
            logging.info(f"Frame loaded at timestamp={pts:.4f}")

    query_ts = torch.tensor(query_timestamps, dtype=torch.float64)
    loaded_ts = torch.tensor(loaded_ts, dtype=torch.float64)

    # compute distances between each query timestamp and loaded timestamps
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
    )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # Keep image tensors in uint8 [0,255]. Qwen-VL image processors rescale
    # internally; returning float [0,1] here would be rescaled a second time.
    closest_frames = closest_frames.type(torch.uint8)

    assert len(timestamps) == len(closest_frames)
    return closest_frames


def encode_video_frames(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int | None = 2,
    crf: int | None = 30,
    fast_decode: int = 0,
    log_level: int | None = av.logging.ERROR,
    overwrite: bool = False,
) -> None:
    """More info on ffmpeg arguments tuning on `benchmark/video/README.md`"""
    # Check encoder availability
    if vcodec not in ["h264", "hevc", "libsvtav1"]:
        raise ValueError(f"Unsupported video codec: {vcodec}. Supported codecs are: h264, hevc, libsvtav1.")

    video_path = Path(video_path)
    imgs_dir = Path(imgs_dir)

    video_path.parent.mkdir(parents=True, exist_ok=overwrite)

    # Encoders/pixel formats incompatibility check
    if (vcodec == "libsvtav1" or vcodec == "hevc") and pix_fmt == "yuv444p":
        logging.warning(f"Incompatible pixel format 'yuv444p' for codec {vcodec}, auto-selecting format 'yuv420p'")
        pix_fmt = "yuv420p"

    # Get input frames
    template = "frame_" + ("[0-9]" * 6) + ".png"
    input_list = sorted(glob.glob(str(imgs_dir / template)), key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # Define video output frame size (assuming all input frames are the same size)
    if len(input_list) == 0:
        raise FileNotFoundError(f"No images found in {imgs_dir}.")
    with Image.open(input_list[0]) as dummy_image:
        width, height = dummy_image.size

    # Define video codec options
    video_options = {}

    if g is not None:
        video_options["g"] = str(g)

    if crf is not None:
        video_options["crf"] = str(crf)

    if fast_decode:
        key = "svtav1-params" if vcodec == "libsvtav1" else "tune"
        value = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
        video_options[key] = value

    # Set logging level
    if log_level is not None:
        # "While less efficient, it is generally preferable to modify logging with Python’s logging"
        logging.getLogger("libav").setLevel(log_level)

    # Create and open output file (overwrite by default)
    with av.open(str(video_path), "w") as output:
        output_stream = output.add_stream(vcodec, fps, options=video_options)
        output_stream.pix_fmt = pix_fmt
        output_stream.width = width
        output_stream.height = height

        # Loop through input frames and encode them
        for input_data in input_list:
            with Image.open(input_data) as pil_image:
                input_image = pil_image.convert("RGB")
                input_frame = av.VideoFrame.from_image(input_image)
            packet = output_stream.encode(input_frame)
            if packet:
                output.mux(packet)

        # Flush the encoder
        packet = output_stream.encode()
        if packet:
            output.mux(packet)

    # Reset logging level
    if log_level is not None:
        av.logging.restore_default_callback()

    if not video_path.exists():
        raise OSError(f"Video encoding did not work. File not found: {video_path}.")


@dataclass
class VideoFrame:
    # TODO(rcadene, lhoestq): move to Hugging Face `datasets` repo
    """
    Provides a type for a dataset containing video frames.

    Example:

    ```python
    data_dict = [{"image": {"path": "videos/episode_0.mp4", "timestamp": 0.3}}]
    features = {"image": VideoFrame()}
    Dataset.from_dict(data_dict, features=Features(features))
    ```
    """

    pa_type: ClassVar[Any] = pa.struct({"path": pa.string(), "timestamp": pa.float32()})
    _type: str = field(default="VideoFrame", init=False, repr=False)

    def __call__(self):
        return self.pa_type


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        "'register_feature' is experimental and might be subject to breaking changes in the future.",
        category=UserWarning,
    )
    # to make VideoFrame available in HuggingFace `datasets`
    register_feature(VideoFrame, "VideoFrame")


def get_audio_info(video_path: Path | str) -> dict:
    # Set logging level
    logging.getLogger("libav").setLevel(av.logging.ERROR)

    # Getting audio stream information
    audio_info = {}
    with av.open(str(video_path), "r") as audio_file:
        try:
            audio_stream = audio_file.streams.audio[0]
        except IndexError:
            # Reset logging level
            av.logging.restore_default_callback()
            return {"has_audio": False}

        audio_info["audio.channels"] = audio_stream.channels
        audio_info["audio.codec"] = audio_stream.codec.canonical_name
        # In an ideal loseless case : bit depth x sample rate x channels = bit rate.
        # In an actual compressed case, the bit rate is set according to the compression level : the lower the bit rate, the more compression is applied.
        audio_info["audio.bit_rate"] = audio_stream.bit_rate
        audio_info["audio.sample_rate"] = audio_stream.sample_rate  # Number of samples per second
        # In an ideal loseless case : fixed number of bits per sample.
        # In an actual compressed case : variable number of bits per sample (often reduced to match a given depth rate).
        audio_info["audio.bit_depth"] = audio_stream.format.bits
        audio_info["audio.channel_layout"] = audio_stream.layout.name
        audio_info["has_audio"] = True

    # Reset logging level
    av.logging.restore_default_callback()

    return audio_info


def get_video_info(video_path: Path | str) -> dict:
    # Set logging level
    logging.getLogger("libav").setLevel(av.logging.ERROR)

    # Getting video stream information
    video_info = {}
    with av.open(str(video_path), "r") as video_file:
        try:
            video_stream = video_file.streams.video[0]
        except IndexError:
            # Reset logging level
            av.logging.restore_default_callback()
            return {}

        video_info["video.height"] = video_stream.height
        video_info["video.width"] = video_stream.width
        video_info["video.codec"] = video_stream.codec.canonical_name
        video_info["video.pix_fmt"] = video_stream.pix_fmt
        video_info["video.is_depth_map"] = False

        # Calculate fps from r_frame_rate
        video_info["video.fps"] = int(video_stream.base_rate)

        pixel_channels = get_video_pixel_channels(video_stream.pix_fmt)
        video_info["video.channels"] = pixel_channels

    # Reset logging level
    av.logging.restore_default_callback()

    # Adding audio stream information
    video_info.update(**get_audio_info(video_path))

    return video_info


def get_video_pixel_channels(pix_fmt: str) -> int:
    if "gray" in pix_fmt or "depth" in pix_fmt or "monochrome" in pix_fmt:
        return 1
    elif "rgba" in pix_fmt or "yuva" in pix_fmt:
        return 4
    elif "rgb" in pix_fmt or "yuv" in pix_fmt:
        return 3
    else:
        raise ValueError("Unknown format")


def get_image_pixel_channels(image: Image):
    if image.mode == "L":
        return 1  # Grayscale
    elif image.mode == "LA":
        return 2  # Grayscale + Alpha
    elif image.mode == "RGB":
        return 3  # RGB
    elif image.mode == "RGBA":
        return 4  # RGBA
    else:
        raise ValueError("Unknown format")
