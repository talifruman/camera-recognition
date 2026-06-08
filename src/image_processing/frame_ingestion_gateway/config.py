"""Configuration model for Frame Ingestion Gateway."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from image_processing.shared.contracts import FramePacketSink


@dataclass(frozen=True)
class FrameIngestionGatewayConfig:
    """Immutable runtime configuration for gateway startup."""

    bind_address: str
    service_name: str
    max_concurrent_streams: int
    configured_camera_ids: Sequence[str]
    supported_source_formats: Sequence[str]
    max_frame_width: int
    max_frame_height: int
    max_payload_bytes: int
    allowed_timestamp_skew_ms: int
    allowed_timestamp_lag_ms: int
    strict_timestamp_validation_enabled: bool
    duplicate_frame_tracking_enabled: bool
    duplicate_frame_tracking_window_size: int
    stop_timeout_ms: int
    log_rate_limit_window_ms: int
    log_rate_limit_max_per_key: int
    sink: FramePacketSink
    on_unknown_camera_rejected: Callable[[str], None] | None = None
