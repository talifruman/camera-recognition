"""Health snapshot models for the Image Processing Service."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class WorkerStatus:
    """Worker runtime status for one camera lane."""

    worker_alive: bool
    last_frame_started_at_ms: int
    last_frame_completed_at_ms: int
    last_error_code: str


@dataclass(frozen=True, slots=True)
class LaneHealth:
    """Per-camera lane health snapshot."""

    queue_depth: int
    queue_bytes: int
    oldest_frame_age_ms: int
    queue_age_p95_ms: float
    worker_status: WorkerStatus
    warmup_state: str
    warmup_completed_at_ms: int


@dataclass(frozen=True, slots=True)
class ImageProcessingServiceHealth:
    """Aggregate service health snapshot."""

    state: str
    configured_camera_count: int
    active_processing_workers: int
    total_queued_frames: int
    total_queued_bytes: int
    queue_depth_per_camera: dict[str, int] = field(default_factory=dict)
    queue_bytes_per_camera: dict[str, int] = field(default_factory=dict)
    frames_dropped_oldest_total: int = 0
    frames_dropped_newest_total: int = 0
    unknown_camera_rejected_total: int = 0
    stale_frames_dropped_total: int = 0
    worker_error_total: int = 0
    worker_timeout_total: int = 0
    degraded_reason_code: str = ""
    degraded_reason_message: str = ""
    lane_health_by_camera_id: dict[str, LaneHealth] = field(default_factory=dict)
    queue_age_p95_per_camera: dict[str, float] = field(default_factory=dict)
    average_queue_wait_ms: float = 0.0
    max_queue_wait_ms: float = 0.0
    oldest_frame_age_ms_per_camera: dict[str, int] = field(default_factory=dict)
    worker_alive: dict[str, bool] = field(default_factory=dict)
    last_frame_started_at_ms: dict[str, int] = field(default_factory=dict)
    last_frame_completed_at_ms: dict[str, int] = field(default_factory=dict)
    last_error_code: dict[str, str] = field(default_factory=dict)
    cold_start_frame_total: int = 0
    camera_warmup_state: dict[str, str] = field(default_factory=dict)
    warmup_completed_at_ms: dict[str, int] = field(default_factory=dict)