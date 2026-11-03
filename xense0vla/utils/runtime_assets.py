"""Portable runtime assets for standalone VLA HF checkpoints."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml


TRAINING_CONFIG_NAME = "vla_cli.yaml"
RUNTIME_ASSETS_DIR = "runtime_assets"
RUNTIME_NORM_STATS_NAME = "norm_stats.json"
RUNTIME_ROBOT_CONFIGS_DIR = "robot_configs"

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_training_manifest(directory: str | Path) -> Path:
    """Locate the required deployment manifest inside the checkpoint."""
    root = Path(directory).expanduser().resolve()
    candidate = root / TRAINING_CONFIG_NAME
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing {TRAINING_CONFIG_NAME} in {root}")
    resolved = candidate.resolve()
    if resolved.parent != root:
        raise ValueError(f"Training manifest escapes checkpoint root: {candidate}")
    return resolved


def _resolve_training_source(path_value: str, output_dir: str | Path) -> Path:
    """Resolve a path recorded by the training CLI before it is packaged."""

    source = Path(path_value).expanduser()
    if source.is_absolute():
        candidates = (source,)
    else:
        candidates = (
            Path.cwd() / source,
            _PROJECT_ROOT / source,
            Path(output_dir).expanduser().resolve() / source,
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    tried = ", ".join(repr(str(candidate)) for candidate in candidates)
    raise FileNotFoundError(
        f"Training asset {path_value!r} does not exist. Tried: {tried}."
    )


def _load_mapping(path: Path, *, kind: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"{kind} {str(path)!r} must contain a YAML mapping.")
    return payload


def _find_norm_stats_source(
    data_config: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    configured_path = data_config.get("norm_stats_file")
    if not configured_path:
        raise ValueError(
            "A standalone VLA checkpoint requires an explicit "
            "data.norm_stats_file; robot YAML fallbacks are not supported."
        )
    source = _resolve_training_source(str(configured_path), output_dir)
    if not source.is_file():
        raise FileNotFoundError(
            f"data.norm_stats_file must be a file, got {str(source)!r}."
        )
    return source


def _validate_norm_stats(path: Path) -> None:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict) or not isinstance(payload.get("norm_stats"), dict):
        raise ValueError(
            f"Normalization file {str(path)!r} must contain a norm_stats object."
        )


def bundle_runtime_assets(
    output_dir: str | Path,
    checkpoint_dir: str | Path,
) -> Path:
    """Copy the deployment manifest and its runtime data into ``checkpoint_dir``.

    The copied manifest contains paths relative to its own directory. This
    makes the complete HF checkpoint movable without preserving the training
    repository layout.
    """

    output_dir = Path(output_dir).expanduser().resolve()
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    source_manifest = resolve_training_manifest(output_dir)
    if not source_manifest.is_file():
        raise FileNotFoundError(
            "Cannot create a standalone VLA HF checkpoint because "
            f"{str(source_manifest)!r} is missing."
        )

    manifest = _load_mapping(source_manifest, kind="Training manifest")
    data_config = manifest.get("data")
    if not isinstance(data_config, dict):
        raise ValueError(
            f"Training manifest {str(source_manifest)!r} has no data mapping."
        )

    robot_config_root_value = data_config.get("robot_config_root")
    if not robot_config_root_value:
        raise ValueError(
            "A standalone VLA checkpoint requires data.robot_config_root."
        )
    robot_config_root = _resolve_training_source(
        str(robot_config_root_value),
        output_dir,
    )
    if not robot_config_root.is_dir():
        raise NotADirectoryError(
            f"data.robot_config_root must be a directory, got {str(robot_config_root)!r}."
        )

    robot_config_paths = sorted(
        path
        for path in robot_config_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    )
    if not robot_config_paths:
        raise FileNotFoundError(
            f"No robot YAML files were found under {str(robot_config_root)!r}."
        )

    norm_stats_source = _find_norm_stats_source(
        data_config,
        output_dir,
    )
    _validate_norm_stats(norm_stats_source)

    runtime_root = checkpoint_dir / RUNTIME_ASSETS_DIR
    runtime_norm_path = runtime_root / RUNTIME_NORM_STATS_NAME
    runtime_robot_root = runtime_root / RUNTIME_ROBOT_CONFIGS_DIR
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_robot_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(norm_stats_source, runtime_norm_path)
    for source_path in robot_config_paths:
        relative_path = source_path.relative_to(robot_config_root)
        destination = runtime_robot_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)

    data_config["norm_stats_file"] = (
        Path(RUNTIME_ASSETS_DIR) / RUNTIME_NORM_STATS_NAME
    ).as_posix()
    data_config["robot_config_root"] = (
        Path(RUNTIME_ASSETS_DIR) / RUNTIME_ROBOT_CONFIGS_DIR
    ).as_posix()

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    destination_manifest = checkpoint_dir / TRAINING_CONFIG_NAME
    with destination_manifest.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            manifest,
            file,
            sort_keys=False,
            allow_unicode=True,
        )
    return destination_manifest


def resolve_manifest_asset_path(
    path_value: str | Path | None,
    manifest_path: str | Path,
) -> str:
    """Resolve a required checkpoint-local asset relative to its manifest."""

    if path_value is None or not str(path_value).strip():
        raise ValueError("The checkpoint manifest is missing a required asset path.")

    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Checkpoint manifest does not exist: {str(manifest)!r}.")

    value = Path(path_value).expanduser()
    if value.is_absolute():
        raise ValueError(
            "Checkpoint manifest asset paths must be relative to the manifest; "
            f"got absolute path {str(value)!r}."
        )

    checkpoint_root = manifest.parent
    candidate = (checkpoint_root / value).resolve()
    try:
        candidate.relative_to(checkpoint_root)
    except ValueError as exc:
        raise ValueError(
            f"Manifest asset path {str(value)!r} escapes checkpoint root "
            f"{str(checkpoint_root)!r}."
        ) from exc
    if not candidate.exists():
        raise FileNotFoundError(
            f"Manifest asset {str(value)!r} does not exist relative to "
            f"{str(manifest)!r}: expected {str(candidate)!r}."
        )
    return str(candidate)


def resolve_backbone_source(checkpoint_path: str | Path) -> str:
    """Return the required checkpoint-local Qwen config/processor bundle."""

    checkpoint_dir = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(
            f"VLA checkpoint directory does not exist: {str(checkpoint_dir)!r}."
        )

    canonical_backbone = (checkpoint_dir / "vlm_backbone").resolve()
    try:
        canonical_backbone.relative_to(checkpoint_dir)
    except ValueError as exc:
        raise ValueError(
            f"Checkpoint vlm_backbone escapes checkpoint root {str(checkpoint_dir)!r}."
        ) from exc
    if not canonical_backbone.is_dir():
        raise FileNotFoundError(
            "A standalone VLA checkpoint requires a local "
            f"vlm_backbone directory at {str(canonical_backbone)!r}."
        )
    config_path = canonical_backbone / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            "The checkpoint-local vlm_backbone is incomplete; missing "
            f"{str(config_path)!r}."
        )
    return str(canonical_backbone)
