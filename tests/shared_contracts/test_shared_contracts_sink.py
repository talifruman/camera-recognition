"""Shared contract tests for FramePacketSink enqueue semantics."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_processing.shared.contracts import (  # type: ignore[import-not-found]
    EnqueueRejectReason,
    EnqueueResult,
    FramePacket,
    FramePacketSink,
)


def _assert_enqueue_result_contract(result: EnqueueResult) -> None:
    """Assert enqueue result invariants required by the shared contract."""
    accepted = result.get("accepted", False)
    reason = result.get("reason")
    if accepted:
        assert reason is None, "accepted=true requires reason=None"
    else:
        assert reason is not None, "accepted=false requires reason to be set"


def _classify_result(result: dict[str, object]) -> str:
    """Classify enqueue outcome using only programmatic fields.

    The optional freeform message is ignored by design.
    """
    accepted = bool(result.get("accepted", False))
    reason = result.get("reason")
    if accepted:
        return "accepted"
    if isinstance(reason, EnqueueRejectReason):
        return reason.value
    return "rejected"


class FakeSink(FramePacketSink):
    """Simple sink implementation for shared contract tests."""

    def __init__(self, accept: bool = True) -> None:
        self.accept = accept
        self.received: list[FramePacket] = []

    def enqueue(self, frame_packet: FramePacket) -> EnqueueResult:
        if self.accept:
            self.received.append(frame_packet)
            return {"accepted": True, "reason": None}
        return {
            "accepted": False,
            "reason": EnqueueRejectReason.QUEUE_FULL_REJECT,
        }


def _make_packet(frame_id: str = "f-1") -> FramePacket:
    """Create a canonical FramePacket for sink tests."""
    return FramePacket(
        frame_id=frame_id,
        camera_id="cam-a",
        timestamp_ms=1000,
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


def test_enqueue_reject_reason_contains_frozen_values() -> None:
    expected = {
        "STOPPING",
        "QUEUE_FULL_DROP_NEWEST",
        "QUEUE_FULL_REJECT",
        "GLOBAL_MEMORY_LIMIT",
        "SINK_UNAVAILABLE",
        "UNKNOWN_CAMERA",
        "BOUNDARY_VIOLATION",
        "INTERNAL_ERROR",
    }
    actual = {member.value for member in EnqueueRejectReason}
    assert actual == expected


def test_enqueue_result_accepted_true_requires_reason_none() -> None:
    valid_result: EnqueueResult = {"accepted": True, "reason": None}
    _assert_enqueue_result_contract(valid_result)

    invalid_result: EnqueueResult = {
        "accepted": True,
        "reason": EnqueueRejectReason.INTERNAL_ERROR,
    }
    with pytest.raises(AssertionError):
        _assert_enqueue_result_contract(invalid_result)


def test_enqueue_result_accepted_false_requires_reason_set() -> None:
    valid_result: EnqueueResult = {
        "accepted": False,
        "reason": EnqueueRejectReason.SINK_UNAVAILABLE,
    }
    _assert_enqueue_result_contract(valid_result)

    invalid_result: EnqueueResult = {"accepted": False, "reason": None}
    with pytest.raises(AssertionError):
        _assert_enqueue_result_contract(invalid_result)


def test_message_is_optional_and_non_programmatic() -> None:
    without_message = {
        "accepted": False,
        "reason": EnqueueRejectReason.QUEUE_FULL_REJECT,
    }
    with_message = {
        "accepted": False,
        "reason": EnqueueRejectReason.QUEUE_FULL_REJECT,
        "message": "human readable only",
    }
    assert _classify_result(without_message) == "QUEUE_FULL_REJECT"
    assert _classify_result(with_message) == "QUEUE_FULL_REJECT"


def test_frame_packet_sink_can_be_implemented_by_simple_fake() -> None:
    sink: FramePacketSink = FakeSink(accept=True)
    packet = _make_packet()
    result = sink.enqueue(packet)
    _assert_enqueue_result_contract(result)
    assert result["accepted"] is True

    rejecting_sink: FramePacketSink = FakeSink(accept=False)
    reject_result = rejecting_sink.enqueue(_make_packet(frame_id="f-2"))
    _assert_enqueue_result_contract(reject_result)
    assert reject_result["accepted"] is False


def test_no_local_frame_packet_definition_introduced() -> None:
    src_root = SRC_DIR / "image_processing"
    pattern = re.compile(r"^class\s+FramePacket\b", re.MULTILINE)
    matches: list[Path] = []

    for path in src_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            matches.append(path)

    assert len(matches) == 1
    assert matches[0].as_posix().endswith("src/image_processing/shared/contracts.py")
