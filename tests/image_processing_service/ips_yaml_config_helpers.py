"""Helpers for building IPS test configs from the real YAML file."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from image_processing.image_processing_service import (  # type: ignore[import-not-found]
    ImageProcessingServiceConfig,
    OverflowPolicy,
)
from scripts.run_rpm_on_frames import load_yaml_config  # type: ignore[import-not-found]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_IPS_CONFIG_PATH = (
    _PROJECT_ROOT / "config" / "image_processing_service" / "image_processing_service.yaml"
)


def _normalize_overflow_policy(config_data: dict[str, Any]) -> None:
    """Normalize string overflow policy values to OverflowPolicy enum."""
    overflow_policy = config_data.get("overflow_policy")
    if isinstance(overflow_policy, str):
        config_data["overflow_policy"] = OverflowPolicy(overflow_policy)


def build_ips_config_from_yaml(
    camera_ids: Sequence[str],
    config_path: Path | None = None,
    **overrides: Any,
) -> ImageProcessingServiceConfig:
    """Load IPS YAML config, apply minimal overrides, and build config object."""
    target_path = _DEFAULT_IPS_CONFIG_PATH if config_path is None else config_path
    config_data = dict(load_yaml_config(target_path))
    config_data["camera_ids"] = list(camera_ids)
    config_data.update(overrides)
    _normalize_overflow_policy(config_data)
    return ImageProcessingServiceConfig(**config_data)
