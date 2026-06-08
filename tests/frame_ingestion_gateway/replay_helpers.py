"""Helpers for replaying image assets as fake camera ingress frames."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

_TESTS_DIR = Path(__file__).resolve().parents[1]
_SRC_DIR = _TESTS_DIR.parent / "src"
for _path in (_SRC_DIR, _TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from image_processing.frame_ingestion_gateway import (  # type: ignore[import-not-found]
    IngressFrameMessage,
)

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def get_asset_path(asset_name: str) -> Path:
    """Return the absolute path for one gateway replay asset."""
    return _ASSETS_DIR / asset_name


def read_asset_bytes(asset_name: str) -> bytes:
    """Read one replay asset from disk as raw bytes."""
    return get_asset_path(asset_name).read_bytes()


def get_asset_dimensions(asset_name: str) -> tuple[int, int]:
    """Decode one replay asset and return its width and height."""
    asset_path = get_asset_path(asset_name)
    image = cv2.imread(str(asset_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode replay asset: {asset_path}")
    height, width = image.shape[:2]
    return width, height


def build_replay_message(
    asset_name: str,
    *,
    frame_id: str,
    camera_id: str,
    timestamp_ms: int,
    source_format: str = "JPEG",
) -> IngressFrameMessage:
    """Build an ingress message from one replay asset."""
    width, height = get_asset_dimensions(asset_name)
    return IngressFrameMessage(
        frame_id=frame_id,
        camera_id=camera_id,
        timestamp_ms=timestamp_ms,
        width=width,
        height=height,
        source_format=source_format,
        payload_bytes=read_asset_bytes(asset_name),
    )
