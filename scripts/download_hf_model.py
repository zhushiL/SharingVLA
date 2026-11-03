"""Download an immutable Hugging Face model snapshot and record its provenance."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--local_dir", default="pretrained_models/Qwen3.5-2B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max_workers", type=int, default=4)
    args = parser.parse_args()

    info = HfApi().model_info(args.repo_id, revision=args.revision, files_metadata=True)
    destination = Path(args.local_dir).expanduser().resolve()
    print(f"Downloading {args.repo_id}@{info.sha} to {destination}", flush=True)
    snapshot_download(
        repo_id=args.repo_id,
        revision=info.sha,
        local_dir=destination,
        max_workers=args.max_workers,
    )
    files = []
    for entry in info.siblings:
        path = destination / entry.rfilename
        if not path.is_file() or path.stat().st_size != entry.size:
            raise RuntimeError(f"Missing or incomplete download: {path}")
        record = {"path": entry.rfilename, "bytes": entry.size}
        if entry.lfs is not None:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            record["sha256"] = digest.hexdigest()
            if record["sha256"] != entry.lfs.sha256:
                raise RuntimeError(f"SHA-256 mismatch: {path}")
        files.append(record)
    manifest = {"repo_id": args.repo_id, "revision": info.sha, "files": files}
    (destination / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {len(files)} files ({sum(item['bytes'] for item in files):,} bytes).", flush=True)


if __name__ == "__main__":
    main()
