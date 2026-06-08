"""Ingress message and normalized buffer contracts for gateway runtime."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IngressFrameMessage:
    """Raw ingress frame envelope received from transport."""

    frame_id: str
    camera_id: str
    timestamp_ms: int
    width: int
    height: int
    source_format: str
    source_layout: str | None = None
    source_num_color_channels: int | None = None
    source_bits_per_channel: int | None = None
    payload_bytes: bytes = b""


@dataclass(frozen=True)
class NormalizedFrameBuffer:
    """Canonical normalized frame buffer produced by normalizer."""

    width: int
    height: int
    color_format: str
    layout: str
    dtype: str
    value_range: str
    num_color_channels: int
    bits_per_channel: int
    image_bytes: bytes
