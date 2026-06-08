"""Frozen-spec tests for Frame Ingestion Gateway.

These tests enforce the ingress-only, sink-publication architecture.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_processing.frame_ingestion_gateway import (  # type: ignore[import-not-found]
    FrameIngestionGateway,
    FrameIngestionGatewayConfig,
    GatewayLifecycleError,
    InMemoryFrameIngressTransport,
    IngressFrameMessage,
)
from image_processing.shared.contracts import (  # type: ignore[import-not-found]
    EnqueueRejectReason,
    EnqueueResult,
    FramePacket,
    FramePacketSink,
)

SUPPORTED_FORMATS = [
    "RGB",
    "BGR",
    "GRAY8",
    "YUV420",
    "NV12",
    "YUY2",
    "JPEG",
    "MJPEG",
]


class CapturingSink(FramePacketSink):
    """Sink test double that captures published packets."""

    def __init__(self) -> None:
        self.received: list[FramePacket] = []
        self.reject_mode = False
        self.reject_reason = EnqueueRejectReason.QUEUE_FULL_REJECT

    def enqueue(self, frame_packet: FramePacket) -> EnqueueResult:
        """Capture packet and return accepted/rejected result."""
        if self.reject_mode:
            return {
                "accepted": False,
                "reason": self.reject_reason,
            }
        self.received.append(frame_packet)
        return {"accepted": True, "reason": None}


def make_config(sink: FramePacketSink, **overrides: object) -> FrameIngestionGatewayConfig:
    """Build a default gateway config with optional overrides."""
    data = {
        "bind_address": "127.0.0.1:50061",
        "service_name": "ingress-tests",
        "max_concurrent_streams": 32,
        "configured_camera_ids": ["cam-a", "cam-b"],
        "supported_source_formats": list(SUPPORTED_FORMATS),
        "max_frame_width": 1920,
        "max_frame_height": 1080,
        "max_payload_bytes": 10 * 1024 * 1024,
        "allowed_timestamp_skew_ms": 200,
        "allowed_timestamp_lag_ms": 5000,
        "strict_timestamp_validation_enabled": False,
        "duplicate_frame_tracking_enabled": False,
        "duplicate_frame_tracking_window_size": 32,
        "stop_timeout_ms": 300,
        "log_rate_limit_window_ms": 5000,
        "log_rate_limit_max_per_key": 1,
        "sink": sink,
    }
    data.update(overrides)
    return FrameIngestionGatewayConfig(**data)


def make_message(
    source_format: str,
    payload_bytes: bytes,
    *,
    frame_id: str = "f-1",
    camera_id: str = "cam-a",
    timestamp_ms: int = 1_000,
    width: int = 4,
    height: int = 4,
    source_layout: str | None = None,
    source_bits_per_channel: int | None = None,
) -> IngressFrameMessage:
    """Build an ingress frame message for tests."""
    return IngressFrameMessage(
        frame_id=frame_id,
        camera_id=camera_id,
        timestamp_ms=timestamp_ms,
        width=width,
        height=height,
        source_format=source_format,
        source_layout=source_layout,
        source_num_color_channels=None,
        source_bits_per_channel=source_bits_per_channel,
        payload_bytes=payload_bytes,
    )


def make_jpeg(width: int, height: int, bgr: tuple[int, int, int]) -> bytes:
    """Encode a small BGR image into JPEG bytes."""
    img = np.full((height, width, 3), bgr, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", img)
    assert ok
    return encoded.tobytes()


def start_gateway(
    sink: FramePacketSink,
    *,
    config_overrides: dict[str, object] | None = None,
    transport: InMemoryFrameIngressTransport | None = None,
) -> tuple[FrameIngestionGateway, InMemoryFrameIngressTransport]:
    """Create, configure, and start a gateway instance for tests."""
    concrete_transport = transport or InMemoryFrameIngressTransport()
    gateway = FrameIngestionGateway(transport=concrete_transport)
    config = make_config(sink=sink, **(config_overrides or {}))
    gateway.configure(config)
    gateway.start()
    return gateway, concrete_transport


def wait_until(predicate: Callable[[], bool], timeout_s: float = 1.0) -> bool:
    """Poll predicate until true or timeout."""
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.mark.parametrize(
    "format_name,payload_builder",
    [
        ("RGB", lambda: bytes(range(4 * 4 * 3))),
        ("BGR", lambda: bytes([0, 0, 255] * 16)),
        ("GRAY8", lambda: bytes([120] * 16)),
        ("JPEG", lambda: make_jpeg(4, 4, (0, 0, 255))),
        ("MJPEG", lambda: make_jpeg(4, 4, (0, 255, 0))),
    ],
)
def test_supported_formats_publish_canonical_frame_packet(
    format_name: str,
    payload_builder: Callable[[], bytes],
) -> None:
    sink = CapturingSink()
    gateway, transport = start_gateway(sink)
    try:
        msg = make_message(format_name, payload_builder())
        transport.inject_message(msg)
        assert wait_until(lambda: len(sink.received) == 1)
        packet = sink.received[0]
        assert packet.pixel_format == "RGB"
        assert packet.layout == "HWC"
        assert packet.dtype == "uint8"
        assert packet.value_range == "[0,255]"
        assert packet.num_color_channels == 3
        assert packet.bits_per_channel == 8
        assert packet.packing == "tightly_packed"
        assert len(packet.image_bytes) == packet.width * packet.height * 3
    finally:
        gateway.stop()


def test_gray8_expands_channels_equally() -> None:
    sink = CapturingSink()
    gateway, transport = start_gateway(sink)
    try:
        msg = make_message("GRAY8", bytes([42] * 16))
        transport.inject_message(msg)
        assert wait_until(lambda: len(sink.received) == 1)
        arr = np.frombuffer(sink.received[0].image_bytes, dtype=np.uint8).reshape(4, 4, 3)
        np.testing.assert_array_equal(arr[:, :, 0], arr[:, :, 1])
        np.testing.assert_array_equal(arr[:, :, 1], arr[:, :, 2])
    finally:
        gateway.stop()


def test_jpeg_declared_dimension_mismatch_rejected() -> None:
    sink = CapturingSink()
    gateway, transport = start_gateway(sink)
    try:
        msg = make_message("JPEG", make_jpeg(4, 4, (0, 0, 255)), width=5, height=5)
        transport.inject_message(msg)
        time.sleep(0.05)
        assert not sink.received
        assert gateway.health().frames_rejected_total >= 1
    finally:
        gateway.stop()


@pytest.mark.parametrize(
    "message",
    [
        make_message("rgb", bytes(48)),
        make_message("H264", bytes(48)),
        make_message("JPEG", make_jpeg(4, 4, (1, 2, 3)), source_layout="HWC"),
        make_message("RGB", bytes(48), source_bits_per_channel=16),
        make_message("RGB", bytes(10)),
        make_message("YUV420", bytes(3 * 4 * 3 // 2), width=3, height=4),
        make_message("NV12", bytes(4 * 3 * 3 // 2), width=4, height=3),
        make_message("RGB", bytes(48), width=4000, height=4),
        make_message("RGB", bytes(48), width=4, height=4000),
        make_message("RGB", bytes(11_000_000), width=4, height=4),
    ],
)
def test_invalid_messages_rejected(message: IngressFrameMessage) -> None:
    sink = CapturingSink()
    gateway, transport = start_gateway(sink)
    try:
        transport.inject_message(message)
        time.sleep(0.05)
        assert not sink.received
        assert gateway.health().frames_rejected_total >= 1
    finally:
        gateway.stop()


def test_unknown_camera_rejected_before_worker_dispatch() -> None:
    sink = CapturingSink()
    gateway, transport = start_gateway(sink)
    try:
        msg = make_message("RGB", bytes(48), camera_id="cam-unknown")
        transport.inject_message(msg)
        time.sleep(0.05)
        assert not sink.received
        assert gateway.health().unknown_camera_rejected_total >= 1
    finally:
        gateway.stop()


def test_duplicate_frame_id_allowed_by_default() -> None:
    sink = CapturingSink()
    gateway, transport = start_gateway(sink)
    try:
        msg1 = make_message("RGB", bytes(48), frame_id="dup")
        msg2 = make_message("RGB", bytes(48), frame_id="dup", timestamp_ms=1010)
        transport.inject_message(msg1)
        transport.inject_message(msg2)
        assert wait_until(lambda: len(sink.received) == 2)
        assert gateway.health().duplicate_frame_id_observed_total == 0
    finally:
        gateway.stop()


def test_duplicate_tracking_metric_increments_when_enabled() -> None:
    sink = CapturingSink()
    overrides: dict[str, object] = {
        "duplicate_frame_tracking_enabled": True,
        "duplicate_frame_tracking_window_size": 8,
    }
    gateway, transport = start_gateway(sink, config_overrides=overrides)
    try:
        msg1 = make_message("RGB", bytes(48), frame_id="dup")
        msg2 = make_message("RGB", bytes(48), frame_id="dup", timestamp_ms=1020)
        transport.inject_message(msg1)
        transport.inject_message(msg2)
        assert wait_until(lambda: len(sink.received) == 2)
        assert gateway.health().duplicate_frame_id_observed_total >= 1
    finally:
        gateway.stop()


def test_timestamp_anomaly_metric_and_strict_mode() -> None:
    sink = CapturingSink()
    gateway, transport = start_gateway(sink)
    try:
        now_ms = int(time.time() * 1000)
        first = make_message("RGB", bytes(48), timestamp_ms=now_ms)
        older = make_message(
            "RGB",
            bytes(48),
            timestamp_ms=now_ms - 100,
            frame_id="f-2",
        )
        transport.inject_message(first)
        transport.inject_message(older)
        assert wait_until(lambda: len(sink.received) == 2)
        assert gateway.health().timestamp_anomaly_total >= 1
    finally:
        gateway.stop()

    strict_sink = CapturingSink()
    strict_gateway, strict_transport = start_gateway(
        strict_sink,
        config_overrides={"strict_timestamp_validation_enabled": True},
    )
    try:
        now_ms = int(time.time() * 1000)
        strict_transport.inject_message(
            make_message("RGB", bytes(48), timestamp_ms=now_ms)
        )
        strict_transport.inject_message(
            make_message(
                "RGB",
                bytes(48),
                timestamp_ms=now_ms - 100,
                frame_id="f-2",
            )
        )
        assert wait_until(lambda: len(strict_sink.received) == 1)
        assert strict_gateway.health().frames_rejected_total >= 1
    finally:
        strict_gateway.stop()


def test_sink_reject_increments_counter_and_reason() -> None:
    sink = CapturingSink()
    sink.reject_mode = True
    sink.reject_reason = EnqueueRejectReason.QUEUE_FULL_REJECT
    gateway, transport = start_gateway(sink)
    try:
        transport.inject_message(make_message("RGB", bytes(48)))
        time.sleep(0.05)
        health = gateway.health()
        assert health.sink_enqueue_rejected_total >= 1
        assert health.last_sink_reject_reason == EnqueueRejectReason.QUEUE_FULL_REJECT.value
    finally:
        gateway.stop()


def test_per_camera_fifo_and_worker_isolation() -> None:
    sink = CapturingSink()
    gateway, transport = start_gateway(sink)
    try:
        msgs = [
            make_message("RGB", bytes(48), frame_id="a-1", camera_id="cam-a"),
            make_message("RGB", bytes(48), frame_id="a-2", camera_id="cam-a"),
            make_message("RGB", bytes(48), frame_id="b-1", camera_id="cam-b"),
        ]
        for msg in msgs:
            transport.inject_message(msg)
        assert wait_until(lambda: len(sink.received) == 3)
        a_ids = [packet.frame_id for packet in sink.received if packet.camera_id == "cam-a"]
        b_ids = [packet.frame_id for packet in sink.received if packet.camera_id == "cam-b"]
        assert a_ids == ["a-1", "a-2"]
        assert b_ids == ["b-1"]
    finally:
        gateway.stop()


def test_slow_camera_does_not_block_other_camera() -> None:
    sink = CapturingSink()

    def slow_enqueue(frame_packet: FramePacket) -> EnqueueResult:
        if frame_packet.camera_id == "cam-a":
            time.sleep(0.2)
        sink.received.append(frame_packet)
        return {"accepted": True, "reason": None}

    sink.enqueue = slow_enqueue  # type: ignore[method-assign]
    gateway, transport = start_gateway(sink)
    try:
        transport.inject_message(make_message("RGB", bytes(48), camera_id="cam-a"))
        transport.inject_message(make_message("RGB", bytes(48), camera_id="cam-b", frame_id="b-fast"))
        assert wait_until(lambda: any(p.frame_id == "b-fast" for p in sink.received), timeout_s=0.5)
    finally:
        gateway.stop()


def test_stop_unblocks_receive_and_timeout_paths() -> None:
    sink = CapturingSink()
    transport = InMemoryFrameIngressTransport(block_receive_until_stop=True)
    gateway, _ = start_gateway(sink, transport=transport)
    gateway.stop()
    assert gateway.health().state == "STOPPED"


class CancelFailureTransport(InMemoryFrameIngressTransport):
    """Transport double that simulates cancellation failure on stop."""

    def stop(self) -> None:
        raise RuntimeError("cancel failed")


def test_transport_cancel_failure_increments_metric() -> None:
    sink = CapturingSink()
    transport = CancelFailureTransport()
    gateway, _ = start_gateway(sink, transport=transport)
    gateway.stop()
    assert gateway.health().gateway_receive_cancel_failed_total >= 1


def test_stop_timeout_increments_metric_and_degrades_health() -> None:
    sink = CapturingSink()
    transport = CancelFailureTransport(block_receive_until_stop=True)
    gateway, _ = start_gateway(
        sink,
        transport=transport,
        config_overrides={"stop_timeout_ms": 10},
    )
    gateway.stop()
    health = gateway.health()
    assert health.gateway_stop_timeout_total >= 1
    assert health.state in {"DEGRADED", "ERROR"}


def test_worker_exception_containment_keeps_other_workers_running() -> None:
    sink = CapturingSink()

    def flaky_enqueue(frame_packet: FramePacket) -> EnqueueResult:
        if frame_packet.camera_id == "cam-a":
            raise RuntimeError("sink crash for cam-a")
        sink.received.append(frame_packet)
        return {"accepted": True, "reason": None}

    sink.enqueue = flaky_enqueue  # type: ignore[method-assign]
    gateway, transport = start_gateway(sink)
    try:
        transport.inject_message(make_message("RGB", bytes(48), camera_id="cam-a"))
        transport.inject_message(make_message("RGB", bytes(48), camera_id="cam-b", frame_id="b-1"))
        assert wait_until(lambda: len(sink.received) == 1)
        assert sink.received[0].camera_id == "cam-b"
        assert gateway.health().state in {"DEGRADED", "ERROR"}
    finally:
        gateway.stop()


def test_concurrent_metric_updates_thread_safe() -> None:
    sink = CapturingSink()
    gateway, transport = start_gateway(sink)
    try:
        done = threading.Event()

        def producer(camera_id: str, start: int) -> None:
            for idx in range(50):
                msg = make_message(
                    "RGB",
                    bytes(48),
                    camera_id=camera_id,
                    frame_id=f"{camera_id}-{start + idx}",
                    timestamp_ms=1_000 + idx,
                )
                transport.inject_message(msg)
            done.set()

        thread_a = threading.Thread(target=producer, args=("cam-a", 0), daemon=True)
        thread_b = threading.Thread(target=producer, args=("cam-b", 1000), daemon=True)
        thread_a.start()
        thread_b.start()
        assert done.wait(timeout=1.5)
        assert wait_until(lambda: len(sink.received) == 100, timeout_s=2.0)
        health = gateway.health()
        assert health.frames_in_total >= 100
        assert health.frames_published_total >= 100
    finally:
        gateway.stop()


def test_rate_limited_logging_suppresses_repeated_errors() -> None:
    sink = CapturingSink()
    gateway, transport = start_gateway(sink)
    try:
        for idx in range(5):
            transport.inject_message(
                make_message("RGB", bytes(10), frame_id=f"bad-{idx}")
            )
        time.sleep(0.1)
        assert gateway.health().frames_rejected_total >= 5
        assert gateway.health().suppressed_log_total >= 1
    finally:
        gateway.stop()


def test_public_api_excludes_legacy_pull_queue_methods() -> None:
    assert not hasattr(FrameIngestionGateway, "get_next_frame")
    assert not hasattr(FrameIngestionGateway, "enqueue")


def test_frame_packet_still_defined_once_in_shared_contracts() -> None:
    duplicates = []
    src_root = SRC_DIR / "image_processing"
    for path in src_root.rglob("*.py"):
        content = path.read_text(encoding="utf-8")
        if "\nclass FramePacket(" in content or "\nclass FramePacket:" in content:
            duplicates.append(path)
    assert len(duplicates) == 1
    assert duplicates[0].as_posix().endswith("src/image_processing/shared/contracts.py")


def test_lifecycle_configure_once_and_before_start() -> None:
    sink = CapturingSink()
    gateway = FrameIngestionGateway(transport=InMemoryFrameIngressTransport())
    gateway.configure(make_config(sink))
    with pytest.raises(GatewayLifecycleError):
        gateway.configure(make_config(sink))
    gateway.start()
    with pytest.raises(GatewayLifecycleError):
        gateway.configure(make_config(sink))
    gateway.stop()
