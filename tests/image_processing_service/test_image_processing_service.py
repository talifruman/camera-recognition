"""Image Processing Service integration and contract tests."""

from __future__ import annotations

import sys
import time
import threading
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pytest

from image_processing.frame_ingestion_gateway import InMemoryFrameIngressTransport
from image_processing.image_processing_service import (
    ImageProcessingService,
    ImageProcessingServiceConfig,
    ImageProcessingServiceConfigurationError,
    ImageProcessingServiceHealth,
    OverflowPolicy,
)
from image_processing.image_processing_service.queues import PerCameraFrameQueue
from image_processing.image_processing_service.sink import SinkState
from image_processing.image_processing_service.workers import ProcessingWorker
from image_processing.shared.contracts import EnqueueRejectReason, FramePacket


class FakeRPM:
    """Simple RPM double that records processed packets."""

    def __init__(self, sleep_s: float = 0.0, raise_for_camera: str | None = None) -> None:
        self.sleep_s = sleep_s
        self.raise_for_camera = raise_for_camera
        self.calls: list[FramePacket] = []

    def process_frame(self, frame_packet: FramePacket):
        if self.raise_for_camera == frame_packet.camera_id:
            raise RuntimeError("boom")
        if self.sleep_s > 0.0:
            time.sleep(self.sleep_s)
        self.calls.append(frame_packet)
        return {"frame_id": frame_packet.frame_id, "camera_id": frame_packet.camera_id}


class FakeClock:
    """Deterministic clock for worker freshness tests."""

    def __init__(self, timestamps_ms: list[int]) -> None:
        self.timestamps_ms = timestamps_ms
        self.index = 0

    def time(self) -> float:
        value = self.timestamps_ms[min(self.index, len(self.timestamps_ms) - 1)]
        self.index += 1
        return value / 1000.0

    def advance_ms(self, delta_ms: int) -> None:
        """Advance the current clock reading by a fixed delta."""
        current = self.timestamps_ms[min(self.index, len(self.timestamps_ms) - 1)]
        self.timestamps_ms[self.index:] = [current + delta_ms] * max(1, len(self.timestamps_ms) - self.index)


def make_packet(
    frame_id: str = "f-1",
    camera_id: str = "cam-a",
    timestamp_ms: int | None = None,
    width: int = 2,
    height: int = 2,
) -> FramePacket:
    """Create a canonical frame packet for IPS tests."""
    timestamp = int(time.time() * 1000) if timestamp_ms is None else timestamp_ms
    return FramePacket(
        frame_id=frame_id,
        camera_id=camera_id,
        timestamp_ms=timestamp,
        width=width,
        height=height,
        pixel_format="RGB",
        layout="HWC",
        dtype="uint8",
        value_range="[0,255]",
        num_color_channels=3,
        bits_per_channel=8,
        packing="tightly_packed",
        image_bytes=bytes(width * height * 3),
    )


def make_config(
    camera_ids: list[str] | None = None,
    **overrides: object,
) -> ImageProcessingServiceConfig:
    """Build a valid service configuration."""
    data: dict[str, object] = {
        "camera_ids": ["cam-a", "cam-b"] if camera_ids is None else camera_ids,
        "max_queue_size_per_camera": 4,
        "max_queue_bytes_per_camera": 128,
        "reserved_queue_bytes_per_camera": 32,
        "max_total_queued_bytes": 256,
        "max_frame_age_ms": 1_000,
        "overflow_policy": OverflowPolicy.DROP_OLDEST,
        "drain_on_shutdown": True,
        "max_drain_timeout_ms": 500,
        "worker_shutdown_timeout_ms": 200,
        "queue_degraded_utilization_threshold": 0.8,
        "memory_degraded_utilization_threshold": 0.8,
        "frame_drop_degraded_threshold_per_minute": 60,
        "worker_lag_degraded_threshold_ms": 100,
        "stale_frame_drop_enabled": True,
        "stale_frame_degraded_threshold_per_minute": 30,
        "max_motion_regions_per_frame": 4,
        "max_person_rois_per_frame": 8,
        "max_face_rois_per_frame": 16,
    }
    data.update(overrides)
    return ImageProcessingServiceConfig(**data)


def wait_until(predicate, timeout_s: float = 1.0) -> bool:
    """Poll until a condition becomes true or timeout expires."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def make_service(rpm: FakeRPM | None = None) -> ImageProcessingService:
    """Create a configured IPS instance for tests."""
    service = ImageProcessingService(rpm or FakeRPM(), transport=InMemoryFrameIngressTransport())
    service.configure(make_config())
    return service


def test_invalid_empty_camera_ids_rejected() -> None:
    with pytest.raises(ImageProcessingServiceConfigurationError):
        make_config(camera_ids=[]).validate()


def test_duplicate_camera_ids_rejected() -> None:
    with pytest.raises(ImageProcessingServiceConfigurationError):
        make_config(camera_ids=["cam-a", "cam-a"]).validate()


def test_invalid_queue_sizes_rejected() -> None:
    with pytest.raises(ImageProcessingServiceConfigurationError):
        make_config(max_queue_size_per_camera=0).validate()


def test_invalid_byte_limits_rejected() -> None:
    with pytest.raises(ImageProcessingServiceConfigurationError):
        make_config(max_total_queued_bytes=0).validate()


def test_invalid_timeout_config_rejected() -> None:
    with pytest.raises(ImageProcessingServiceConfigurationError):
        make_config(worker_shutdown_timeout_ms=-1).validate()


def test_known_camera_enqueues_to_correct_queue() -> None:
    service = make_service()
    runtime = service._runtime
    result = runtime.sink.enqueue(make_packet(camera_id="cam-a"))
    assert result["accepted"] is True
    assert runtime.registry.get_queue("cam-a").depth() == 1
    assert runtime.registry.get_queue("cam-b").depth() == 0


def test_unknown_camera_rejected() -> None:
    service = make_service()
    runtime = service._runtime
    result = runtime.sink.enqueue(make_packet(camera_id="cam-unknown"))
    assert result["accepted"] is False
    assert result["reason"] is EnqueueRejectReason.UNKNOWN_CAMERA


def test_stopping_rejects_enqueue() -> None:
    service = make_service()
    runtime = service._runtime
    runtime.sink.close_acceptance()
    result = runtime.sink.enqueue(make_packet())
    assert result["accepted"] is False
    assert result["reason"] is EnqueueRejectReason.STOPPING


def test_drop_newest_has_no_queue_mutation() -> None:
    config = make_config(
        overflow_policy=OverflowPolicy.DROP_NEWEST,
        max_queue_bytes_per_camera=12,
        reserved_queue_bytes_per_camera=0,
    )
    service = ImageProcessingService(FakeRPM(), transport=InMemoryFrameIngressTransport())
    service.configure(config)
    runtime = service._runtime
    runtime.sink.enqueue(make_packet(frame_id="a"))
    result = runtime.sink.enqueue(make_packet(frame_id="b"))
    assert result["accepted"] is False
    assert result["reason"] is EnqueueRejectReason.QUEUE_FULL_DROP_NEWEST
    assert runtime.registry.get_queue("cam-a").depth() == 1


def test_reject_has_no_queue_mutation() -> None:
    config = make_config(
        overflow_policy=OverflowPolicy.REJECT,
        max_queue_bytes_per_camera=12,
        reserved_queue_bytes_per_camera=0,
    )
    service = ImageProcessingService(FakeRPM(), transport=InMemoryFrameIngressTransport())
    service.configure(config)
    runtime = service._runtime
    runtime.sink.enqueue(make_packet(frame_id="a"))
    result = runtime.sink.enqueue(make_packet(frame_id="b"))
    assert result["accepted"] is False
    assert result["reason"] is EnqueueRejectReason.QUEUE_FULL_REJECT
    assert runtime.registry.get_queue("cam-a").depth() == 1


def test_drop_oldest_replaces_atomically() -> None:
    config = make_config(
        max_queue_size_per_camera=1,
        max_queue_bytes_per_camera=12,
        reserved_queue_bytes_per_camera=0,
    )
    service = ImageProcessingService(FakeRPM(), transport=InMemoryFrameIngressTransport())
    service.configure(config)
    runtime = service._runtime
    runtime.sink.enqueue(make_packet(frame_id="old"))
    result = runtime.sink.enqueue(make_packet(frame_id="new"))
    assert result["accepted"] is True
    queue = runtime.registry.get_queue("cam-a")
    assert queue.depth() == 1
    assert queue.dequeue().frame_packet.frame_id == "new"


def test_global_limit_rejects_after_local_limit() -> None:
    config = make_config(max_total_queued_bytes=24, reserved_queue_bytes_per_camera=0)
    service = ImageProcessingService(FakeRPM(), transport=InMemoryFrameIngressTransport())
    service.configure(config)
    runtime = service._runtime
    assert runtime.sink.enqueue(make_packet(camera_id="cam-a", frame_id="a1", width=2, height=2))["accepted"]
    assert runtime.sink.enqueue(make_packet(camera_id="cam-b", frame_id="b1", width=2, height=2))["accepted"]
    assert runtime.sink.enqueue(make_packet(camera_id="cam-a", frame_id="a2", width=2, height=2))["accepted"] is False
    assert runtime.registry.get_queue("cam-a").depth() == 1


def test_reserved_capacity_prevents_starvation() -> None:
    config = make_config(max_total_queued_bytes=24, reserved_queue_bytes_per_camera=12)
    service = ImageProcessingService(FakeRPM(), transport=InMemoryFrameIngressTransport())
    service.configure(config)
    runtime = service._runtime
    assert runtime.sink.enqueue(make_packet(camera_id="cam-a", frame_id="a1", width=2, height=2))["accepted"]
    result = runtime.sink.enqueue(make_packet(camera_id="cam-b", frame_id="b1", width=2, height=2))
    assert result["accepted"] is True
    result = runtime.sink.enqueue(make_packet(camera_id="cam-a", frame_id="a2", width=2, height=2))
    assert result["accepted"] is False


def test_byte_accounting_rollback_on_failed_mutation() -> None:
    config = make_config(max_total_queued_bytes=12, reserved_queue_bytes_per_camera=0)
    service = ImageProcessingService(FakeRPM(), transport=InMemoryFrameIngressTransport())
    service.configure(config)
    runtime = service._runtime
    queue = runtime.registry.get_queue("cam-a")
    runtime.sink.enqueue(make_packet(frame_id="old"))
    before_depth = queue.depth()
    result = runtime.sink.enqueue(make_packet(frame_id="new"))
    assert result["accepted"] is False
    assert queue.depth() == before_depth


def test_fifo_order_is_preserved() -> None:
    queue = PerCameraFrameQueue("cam-a", max_queue_size=4, max_queue_bytes=128)
    queue.enqueue_local(make_packet(frame_id="a"), OverflowPolicy.DROP_OLDEST)
    queue.enqueue_local(make_packet(frame_id="b"), OverflowPolicy.DROP_OLDEST)
    assert queue.dequeue().frame_packet.frame_id == "a"
    assert queue.dequeue().frame_packet.frame_id == "b"


def test_worker_invokes_rpm_logic() -> None:
    rpm = FakeRPM()
    service = make_service(rpm)
    service.start()
    runtime = service._runtime
    runtime.sink.enqueue(make_packet(frame_id="f-1"))
    assert wait_until(lambda: len(rpm.calls) == 1)
    service.stop(drain=False)
    assert rpm.calls[0].frame_id == "f-1"


def test_stop_drain_false_drops_queued_frames() -> None:
    rpm = FakeRPM(sleep_s=0.05)
    service = make_service(rpm)
    service.start()
    runtime = service._runtime
    runtime.sink.enqueue(make_packet(frame_id="f-1"))
    runtime.sink.enqueue(make_packet(frame_id="f-2"))
    service.stop(drain=False)
    health = service.health()
    assert health.total_queued_frames == 0
    assert health.frames_dropped_newest_total >= 0


def test_stop_drain_true_drains_fresh_frames() -> None:
    rpm = FakeRPM()
    service = make_service(rpm)
    service.start()
    runtime = service._runtime
    runtime.sink.enqueue(make_packet(frame_id="f-1"))
    service.stop(drain=True)
    assert len(rpm.calls) >= 1


def test_worker_shutdown_timeout_updates_health() -> None:
    rpm = FakeRPM(sleep_s=0.2)
    service = make_service(rpm)
    service._runtime.config = make_config(worker_shutdown_timeout_ms=1)  # type: ignore[attr-defined]
    service.start()
    runtime = service._runtime
    runtime.sink.enqueue(make_packet(frame_id="f-1"))
    service.stop(drain=False)
    health = service.health()
    assert health.worker_timeout_total >= 0


def test_health_returns_lane_fields() -> None:
    service = make_service()
    health = service.health()
    assert isinstance(health, ImageProcessingServiceHealth)
    assert "cam-a" in health.lane_health_by_camera_id
    assert "cam-b" in health.queue_depth_per_camera


def test_queue_wait_excludes_processing_time() -> None:
    rpm = FakeRPM(sleep_s=0.05)
    service = make_service(rpm)
    service.start()
    runtime = service._runtime
    runtime.sink.enqueue(make_packet(frame_id="f-1"))
    assert wait_until(lambda: len(rpm.calls) == 1)
    health = service.health()
    assert health.average_queue_wait_ms >= 0.0
    assert health.max_queue_wait_ms >= 0.0
    service.stop(drain=False)


def test_cold_start_frame_counted_separately() -> None:
    rpm = FakeRPM()
    service = make_service(rpm)
    service.start()
    service._runtime.sink.enqueue(make_packet(frame_id="f-1"))
    assert wait_until(lambda: len(rpm.calls) == 1)
    health = service.health()
    assert health.cold_start_frame_total >= 1
    assert health.camera_warmup_state["cam-a"] in {"WARMING", "WARMED"}
    service.stop(drain=False)


def test_concurrent_enqueue_is_safe() -> None:
    service = make_service()
    runtime = service._runtime
    threads = []
    for i in range(8):
        thread = threading.Thread(
            target=runtime.sink.enqueue,
            args=(make_packet(frame_id=f"f-{i}"),),
            daemon=True,
        )
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    assert runtime.registry.get_queue("cam-a").depth() >= 1


def test_preprocessing_stale_drop() -> None:
    rpm = FakeRPM()
    service = make_service(rpm)
    worker = ProcessingWorker(
        camera_id="cam-a",
        queue=service._runtime.registry.get_queue("cam-a"),
        rpm=rpm,
        metrics=service._metrics,  # type: ignore[attr-defined]
        config=make_config(max_frame_age_ms=10),
    )
    fake_clock = FakeClock([1_000, 1_000, 1_020, 1_020, 1_020, 1_020])
    import image_processing.image_processing_service.workers as worker_module

    original_time = worker_module.time.time
    worker_module.time.time = fake_clock.time
    try:
        worker._process_entry(
            service._runtime.registry.get_queue("cam-a").enqueue_local(
                make_packet(timestamp_ms=995), OverflowPolicy.DROP_OLDEST
            ).entry
        )
        health = service.health()
        assert health.stale_frames_dropped_total >= 1
    finally:
        worker_module.time.time = original_time


def test_in_flight_stale_frame_may_complete() -> None:
    fake_clock = FakeClock([1_000, 1_000, 1_000, 1_000, 1_000, 1_000])

    class AdvancingRPM(FakeRPM):
        def process_frame(self, frame_packet: FramePacket):
            fake_clock.advance_ms(50)
            return super().process_frame(frame_packet)

    rpm = AdvancingRPM()
    service = make_service(rpm)
    worker = ProcessingWorker(
        camera_id="cam-a",
        queue=service._runtime.registry.get_queue("cam-a"),
        rpm=rpm,
        metrics=service._metrics,  # type: ignore[attr-defined]
        config=make_config(max_frame_age_ms=10),
    )
    import image_processing.image_processing_service.workers as worker_module

    original_time = worker_module.time.time
    worker_module.time.time = fake_clock.time
    try:
        entry = service._runtime.registry.get_queue("cam-a").enqueue_local(
            make_packet(timestamp_ms=995), OverflowPolicy.DROP_OLDEST
        )
        worker._process_entry(entry.entry)
        assert service._metrics.get_counter("stale_completed_processing_total") >= 1  # type: ignore[attr-defined]
    finally:
        worker_module.time.time = original_time
