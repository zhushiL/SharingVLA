"""Adds NumPy array support to msgpack.

msgpack is good for (de)serializing data over a network for multiple reasons:
- msgpack is secure (as opposed to pickle/dill/etc which allow for arbitrary code execution)
- msgpack is widely used and has good cross-language support
- msgpack does not require a schema (as opposed to protobuf/flatbuffers/etc) which is convenient in dynamically typed
    languages like Python and JavaScript
- msgpack is fast and efficient (as opposed to readable formats like JSON/YAML/etc); I found that msgpack was ~4x faster
    than pickle for serializing large arrays using the below strategy

The code below is adapted from https://github.com/lebedov/msgpack-numpy. The reason not to use that library directly is
that it falls back to pickle for object arrays.
"""

import functools
import math

import msgpack
import numpy as np

_ALLOWED_DTYPE_KINDS = ("b", "i", "u", "f")
_MAX_UNPACK_BYTES = 256 * 1024 * 1024  # 256 MiB safety cap per array


def pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind not in _ALLOWED_DTYPE_KINDS:
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        contiguous = obj if obj.flags.c_contiguous else np.ascontiguousarray(obj)
        return {
            b"__ndarray__": True,
            b"data": memoryview(contiguous).cast("B"),
            b"dtype": contiguous.dtype.str,
            b"shape": contiguous.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        dtype = np.dtype(obj[b"dtype"])
        if dtype.kind not in _ALLOWED_DTYPE_KINDS or dtype.hasobject:
            raise ValueError(f"Disallowed ndarray dtype: {dtype}")
        shape = tuple(int(dim) for dim in obj[b"shape"])
        if any(dim < 0 for dim in shape):
            raise ValueError(f"Invalid negative ndarray shape: {shape}")
        expected_bytes = math.prod(shape) * dtype.itemsize
        if expected_bytes > _MAX_UNPACK_BYTES:
            raise ValueError(f"Decoded ndarray exceeds safety limit ({expected_bytes} bytes)")
        raw = obj[b"data"]
        if not isinstance(raw, (bytes, bytearray, memoryview)) or len(raw) != expected_bytes:
            raise ValueError(f"Corrupted ndarray buffer length: expected {expected_bytes}, got {len(raw) if isinstance(raw, (bytes, bytearray, memoryview)) else type(raw).__name__}")
        # Return a writable copy so torch.from_numpy() never warns or aliases immutable bytes.
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()

    if b"__npgeneric__" in obj:
        dtype = np.dtype(obj[b"dtype"])
        if dtype.kind not in _ALLOWED_DTYPE_KINDS or dtype.hasobject:
            raise ValueError(f"Disallowed numpy scalar dtype: {dtype}")
        return dtype.type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
