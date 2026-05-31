"""Runtime smoke tests for IPS using replayed fake camera frames and stub RPM."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
_TESTS_DIR = Path(__file__).resolve().parents[1]
for _path in (_SRC_DIR, _TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from frame_ingestion_gateway.replay_helpers import (  # type: ignore[import-not-found]
    build_replay_message,
)
from image_processing.frame_ingestion_gateway import (  # type: ignore[import-not-found]
    InMemoryFrameIngressTransport,
)
from image_processing.image_processing_service import (  # type: ignore[import-not-found]
    ImageProcessingService,
)
from image_processing_service.ips_yaml_config_helpers import (  # type: ignore[import-not-found]
    build_ips_config_from_yaml,
)

from image_processing.shared.contracts import (  # type: ignore[import-not-found]
    EnqueueRejectReason,
    FramePacket,
)


def _wait_until(predicate, timeout_s: float = 2.0) -> bool:
    """Poll until a predicate returns true or timeout elapses."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class StubRPM:
    """Stub RPM used to validate IPS runtime flow deterministically."""

    def __init__(
        self,
        per_camera_delay_s: dict[str, float] | None = None,
        call_event: threading.Event | None = None,
    ) -> None:
        self._per_camera_delay_s = per_camera_delay_s or {}
        self._call_event = call_event
        self.calls: list[FramePacket] = []
        self._lock = threading.Lock()

    def process_frame(self, frame_packet: FramePacket) -> dict[str, Any]:
        delay_s = self._per_camera_delay_s.get(frame_packet.camera_id, 0.0)
        if delay_s > 0:
            time.sleep(delay_s)
        with self._lock:
            self.calls.append(frame_packet)
        if self._call_event is not None:
            self._call_event.set()
        return {
            "frame_id": frame_packet.frame_id,
            "camera_id": frame_packet.camera_id,
            "persons": [],
        }


@pytest.fixture
def replay_assets() -> list[str]:
    return [
        "frame_000008.jpg",
        "frame_000009.jpg",
        "frame_000010.jpg",
        "frame_000011.jpg",
        "frame_000015.jpg",
    ]


def _start_service(
    rpm: StubRPM,
    camera_ids: list[str],
    **config_overrides: Any,
) -> tuple[ImageProcessingService, InMemoryFrameIngressTransport]:
    """Construct, configure, and start the IPS runtime for smoke tests."""
    transport = InMemoryFrameIngressTransport()
    service = ImageProcessingService(rpm=rpm, transport=transport)
    service.configure(
        build_ips_config_from_yaml(
            camera_ids=camera_ids,
            **config_overrides,
        )
    )
    service.start()
    return service, transport


def _inject_frames(
    transport: InMemoryFrameIngressTransport,
    *,
    camera_id: str,
    frame_prefix: str,
    asset_names: list[str],
    timestamp_ms: int,
) -> None:
    """Inject replay assets as ingress frames for one camera lane."""
    for index, asset_name in enumerate(asset_names):
        message = build_replay_message(
            asset_name,
            frame_id=f"{frame_prefix}-{index}",
            camera_id=camera_id,
            timestamp_ms=timestamp_ms + index,
        )
        transport.inject_message(message)


def _make_packet(frame_id: str, camera_id: str, timestamp_ms: int) -> FramePacket:
    """Create a minimal canonical packet for sink-gate assertions."""
    return FramePacket(
        frame_id=frame_id,
        camera_id=camera_id,
        timestamp_ms=timestamp_ms,
        width=2,
        height=2,
        pixel_format="RGB",
        layout="HWC",
        dtype="uint8",
        value_range="[0,255]",
        num_color_channels=3,
        bits_per_channel=8,
        packing="tightly_packed",
        image_bytes=bytes(12),
    )


def test_runtime_basic_end_to_end_smoke(replay_assets: list[str]) -> None:
    rpm = StubRPM()
    service, transport = _start_service(rpm, ["cam-a"])
    try:
        now_ms = int(time.time() * 1000)
        _inject_frames(
            transport,
            camera_id="cam-a",
            frame_prefix="basic",
            asset_names=replay_assets[:4],
            timestamp_ms=now_ms,
        )
        assert _wait_until(lambda: len(rpm.calls) == 4, timeout_s=3.0)
        health = service.health()
        assert health.state == "RUNNING"
        assert health.active_processing_workers == 1
        assert health.total_queued_frames == 0
        assert health.worker_error_total == 0
        assert health.last_frame_completed_at_ms["cam-a"] > 0
        assert service._metrics.get_camera_counter("cam-a", "processed_per_camera") >= 4
    finally:
        service.stop(drain=False)


def test_runtime_multi_camera_isolation_with_slow_lane(
    replay_assets: list[str],
) -> None:
    rpm = StubRPM(per_camera_delay_s={"cam-slow": 0.08})
    service, transport = _start_service(rpm, ["cam-fast", "cam-slow", "cam-mid"])
    try:
        now_ms = int(time.time() * 1000)
        _inject_frames(
            transport,
            camera_id="cam-slow",
            frame_prefix="slow",
            asset_names=replay_assets[:3],
            timestamp_ms=now_ms,
        )
        _inject_frames(
            transport,
            camera_id="cam-fast",
            frame_prefix="fast",
            asset_names=replay_assets[:3],
            timestamp_ms=now_ms,
        )
        _inject_frames(
            transport,
            camera_id="cam-mid",
            frame_prefix="mid",
            asset_names=replay_assets[:2],
            timestamp_ms=now_ms,
        )
        assert _wait_until(
            lambda: service._metrics.get_camera_counter("cam-fast", "processed_per_camera") >= 2,
            timeout_s=2.0,
        )
        assert _wait_until(
            lambda: service._metrics.get_camera_counter("cam-slow", "processed_per_camera") >= 2,
            timeout_s=4.0,
        )
        assert _wait_until(
            lambda: service._metrics.get_camera_counter("cam-mid", "processed_per_camera") >= 1,
            timeout_s=4.0,
        )
        fast_completed = service.health().last_frame_completed_at_ms["cam-fast"]
        slow_completed = service.health().last_frame_completed_at_ms["cam-slow"]
        assert fast_completed > 0
        assert slow_completed > 0
        assert service._metrics.get_camera_counter("cam-fast", "processed_per_camera") >= 3
        assert service._metrics.get_camera_counter("cam-slow", "processed_per_camera") >= 2
        assert service._metrics.get_camera_counter("cam-mid", "processed_per_camera") >= 1
    finally:
        service.stop(drain=False)


def test_runtime_drop_oldest_overflow_keeps_latest_order(replay_assets: list[str]) -> None:
    rpm = StubRPM(per_camera_delay_s={"cam-a": 0.12})
    service, transport = _start_service(
        rpm,
        ["cam-a"],
        max_queue_size_per_camera=1,
    )
    try:
        now_ms = int(time.time() * 1000)
        _inject_frames(
            transport,
            camera_id="cam-a",
            frame_prefix="overflow",
            asset_names=replay_assets[:5],
            timestamp_ms=now_ms,
        )
        assert _wait_until(lambda: len(rpm.calls) >= 2, timeout_s=8.0)
        processed_ids = [packet.frame_id for packet in rpm.calls]
        processed_indices = [int(frame_id.split("-")[-1]) for frame_id in processed_ids]
        assert processed_indices == sorted(processed_indices)
        assert any(
            curr - prev > 1
            for prev, curr in zip(processed_indices, processed_indices[1:])
        )
        assert service._metrics.get_counter("frames_dropped_oldest_total") >= 1
    finally:
        service.stop(drain=False)


def test_runtime_stale_frames_dropped_before_rpm(
    replay_assets: list[str],
) -> None:
    rpm = StubRPM()
    service, transport = _start_service(rpm, ["cam-a"], max_frame_age_ms=20)
    try:
        stale_ms = int(time.time() * 1000) - 3_000
        _inject_frames(
            transport,
            camera_id="cam-a",
            frame_prefix="stale",
            asset_names=replay_assets[:3],
            timestamp_ms=stale_ms,
        )
        assert _wait_until(
            lambda: service.health().stale_frames_dropped_total >= 3,
            timeout_s=3.0,
        )
        assert len(rpm.calls) == 0
    finally:
        service.stop(drain=False)


def test_runtime_stopping_gate_rejects_new_enqueue() -> None:
    rpm = StubRPM()
    service, _transport = _start_service(rpm, ["cam-a"])
    runtime = service._runtime
    try:
        runtime.sink.close_acceptance()
        packet = _make_packet(
            frame_id="stopping-0",
            camera_id="cam-a",
            timestamp_ms=int(time.time() * 1000),
        )
        result = runtime.sink.enqueue(packet)
        assert result["accepted"] is False
        assert result["reason"] is EnqueueRejectReason.STOPPING
    finally:
        service.stop(drain=False)


def test_runtime_stop_drain_false_completes_without_deadlock(
    replay_assets: list[str],
) -> None:
    rpm = StubRPM(per_camera_delay_s={"cam-a": 0.08})
    service, transport = _start_service(rpm, ["cam-a"])
    now_ms = int(time.time() * 1000)
    _inject_frames(
        transport,
        camera_id="cam-a",
        frame_prefix="shutdown-drop",
        asset_names=replay_assets[:4],
        timestamp_ms=now_ms,
    )
    service.stop(drain=False)
    health = service.health()
    assert health.state in {"STOPPED", "DEGRADED"}
    assert health.active_processing_workers == 0


def test_runtime_stop_drain_true_completes_without_deadlock(
    replay_assets: list[str],
) -> None:
    rpm = StubRPM()
    service, transport = _start_service(rpm, ["cam-a"])
    now_ms = int(time.time() * 1000)
    _inject_frames(
        transport,
        camera_id="cam-a",
        frame_prefix="shutdown-drain",
        asset_names=replay_assets[:3],
        timestamp_ms=now_ms,
    )
    service.stop(drain=True)
    health = service.health()
    assert health.state in {"STOPPED", "DEGRADED"}
    assert health.active_processing_workers == 0
    assert len(rpm.calls) >= 1
