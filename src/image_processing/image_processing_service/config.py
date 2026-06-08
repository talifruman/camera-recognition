"""Configuration model for the Image Processing Service."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .errors import ImageProcessingServiceConfigurationError


class OverflowPolicy(Enum):
    """Queue overflow policy for per-camera frame queues."""

    DROP_OLDEST = "DROP_OLDEST"
    DROP_NEWEST = "DROP_NEWEST"
    REJECT = "REJECT"


@dataclass(frozen=True)
class ImageProcessingServiceConfig:
    """Immutable service configuration validated before startup."""

    camera_ids: Sequence[str]
    max_queue_size_per_camera: int
    max_queue_bytes_per_camera: int
    reserved_queue_bytes_per_camera: int
    max_total_queued_bytes: int
    max_frame_age_ms: int
    overflow_policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST
    drain_on_shutdown: bool = True
    max_drain_timeout_ms: int = 30_000
    worker_shutdown_timeout_ms: int = 5_000
    queue_degraded_utilization_threshold: float = 0.8
    memory_degraded_utilization_threshold: float = 0.8
    frame_drop_degraded_threshold_per_minute: int = 60
    worker_lag_degraded_threshold_ms: int = 2_000
    stale_frame_drop_enabled: bool = True
    stale_frame_degraded_threshold_per_minute: int = 30
    max_motion_regions_per_frame: int = 8
    max_person_rois_per_frame: int = 16
    max_face_rois_per_frame: int = 32

    def validate(self) -> None:
        """Validate the service configuration."""
        self._validate_camera_ids()
        self._validate_queues()
        self._validate_timeouts()
        self._validate_thresholds()

    def _validate_camera_ids(self) -> None:
        camera_ids = list(self.camera_ids)
        if not camera_ids:
            raise ImageProcessingServiceConfigurationError("camera_ids must be non-empty")
        if len(set(camera_ids)) != len(camera_ids):
            raise ImageProcessingServiceConfigurationError("camera_ids must be unique")

    def _validate_queues(self) -> None:
        positive_limits = [
            self.max_queue_size_per_camera,
            self.max_queue_bytes_per_camera,
            self.max_total_queued_bytes,
            self.max_frame_age_ms,
        ]
        if any(limit <= 0 for limit in positive_limits):
            raise ImageProcessingServiceConfigurationError("queue limits must be positive")
        if self.reserved_queue_bytes_per_camera < 0:
            raise ImageProcessingServiceConfigurationError("reserved bytes must be non-negative")
        if self.reserved_queue_bytes_per_camera > self.max_total_queued_bytes:
            raise ImageProcessingServiceConfigurationError(
                "reserved bytes per camera must not exceed total queued bytes"
            )
        if self.reserved_queue_bytes_per_camera > self.max_queue_bytes_per_camera:
            raise ImageProcessingServiceConfigurationError(
                "reserved bytes per camera must not exceed per-camera bytes"
            )
        if self.overflow_policy not in OverflowPolicy:
            raise ImageProcessingServiceConfigurationError("overflow_policy is invalid")

    def _validate_timeouts(self) -> None:
        if self.max_drain_timeout_ms < 0 or self.worker_shutdown_timeout_ms < 0:
            raise ImageProcessingServiceConfigurationError("timeouts must be non-negative")

    def _validate_thresholds(self) -> None:
        thresholds = [
            self.queue_degraded_utilization_threshold,
            self.memory_degraded_utilization_threshold,
        ]
        if any(not (0.0 < threshold <= 1.0) for threshold in thresholds):
            raise ImageProcessingServiceConfigurationError("utilization thresholds must be in (0, 1]")
        if self.frame_drop_degraded_threshold_per_minute < 0:
            raise ImageProcessingServiceConfigurationError("drop threshold must be non-negative")
        if self.worker_lag_degraded_threshold_ms < 0:
            raise ImageProcessingServiceConfigurationError("worker lag threshold must be non-negative")
        if self.stale_frame_degraded_threshold_per_minute < 0:
            raise ImageProcessingServiceConfigurationError("stale threshold must be non-negative")
        positive_rois = [
            self.max_motion_regions_per_frame,
            self.max_person_rois_per_frame,
            self.max_face_rois_per_frame,
        ]
        if any(limit <= 0 for limit in positive_rois):
            raise ImageProcessingServiceConfigurationError("ROI caps must be positive")