"""Gateway health snapshot structures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GatewayHealth:
    """Public health view for Frame Ingestion Gateway."""

    state: str
    transport_state: str
    accepting_frames: bool
    degraded: bool
    last_error_code: str
    last_error_message: str
    last_error_timestamp_ms: int
    active_ingress_workers: int
    configured_camera_count: int
    frames_in_total: int
    frames_accepted_total: int
    frames_rejected_total: int
    frames_published_total: int
    sink_enqueue_rejected_total: int
    decode_failed_total: int
    normalization_failed_total: int
    unsupported_source_format_total: int
    unknown_camera_rejected_total: int
    duplicate_frame_id_observed_total: int
    timestamp_anomaly_total: int
    gateway_stop_timeout_total: int
    gateway_receive_cancel_failed_total: int
    ingest_latency_ms_avg: float
    decode_latency_ms_avg: float
    normalize_latency_ms_avg: float
    worker_error_total: int
    suppressed_log_total: int
    last_sink_reject_reason: str
