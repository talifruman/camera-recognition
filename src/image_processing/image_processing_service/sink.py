"""Service-owned FramePacketSink implementation."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from image_processing.shared.contracts import EnqueueRejectReason, EnqueueResult, FramePacket, FramePacketSink

from .config import OverflowPolicy
from .metrics import ThreadSafeServiceMetrics
from .queues import FrameQueueRegistry, PerCameraFrameQueue, QueueMutationResult


@dataclass(slots=True)
class SinkState:
    """Shared sink acceptance gate state."""

    accepting: bool = True


class ServiceFramePacketSink(FramePacketSink):
    """IPS-owned sink that routes canonical frames into per-camera queues."""

    def __init__(
        self,
        registry: FrameQueueRegistry,
        overflow_policy: OverflowPolicy,
        metrics: ThreadSafeServiceMetrics,
        state: SinkState,
        total_bytes_getter,
        total_bytes_setter,
        reserved_bytes_per_camera: int,
        max_total_queued_bytes: int,
    ) -> None:
        self._registry = registry
        self._overflow_policy = overflow_policy
        self._metrics = metrics
        self._state = state
        self._lock = threading.Lock()
        self._total_bytes_getter = total_bytes_getter
        self._total_bytes_setter = total_bytes_setter
        self._reserved_bytes_per_camera = reserved_bytes_per_camera
        self._max_total_queued_bytes = max_total_queued_bytes

    def close_acceptance(self) -> None:
        """Close the enqueue acceptance gate."""
        with self._lock:
            self._state.accepting = False

    def open_acceptance(self) -> None:
        """Open the enqueue acceptance gate."""
        with self._lock:
            self._state.accepting = True

    def enqueue(self, frame_packet: FramePacket) -> EnqueueResult:
        """Publish a frame packet into the configured per-camera queue."""
        with self._lock:
            if not self._state.accepting:
                return self._rejected(EnqueueRejectReason.STOPPING)
        queue = self._registry.get_queue(frame_packet.camera_id)
        if queue is None:
            self._metrics.incr("unknown_camera_rejected_total")
            return self._rejected(EnqueueRejectReason.UNKNOWN_CAMERA)
        mutation = queue.enqueue_local(frame_packet, self._overflow_policy)
        if not mutation.accepted:
            self._record_rejection(mutation.reason)
            return self._result_from_reason(mutation.reason)
        if not self._can_accept_globally(frame_packet.camera_id, queue, mutation):
            queue.rollback_enqueue(mutation)
            self._record_rejection(EnqueueRejectReason.GLOBAL_MEMORY_LIMIT)
            return self._rejected(EnqueueRejectReason.GLOBAL_MEMORY_LIMIT)
        if mutation.dropped_entry is not None:
            self._metrics.incr("frames_dropped_oldest_total")
            self._metrics.incr("frames_dropped_total")
        self._total_bytes_setter(self._total_bytes_getter() + len(frame_packet.image_bytes) - self._dropped_bytes(mutation))
        self._metrics.incr("frames_enqueued_total")
        self._metrics.incr_camera("accepted_per_camera", frame_packet.camera_id)
        return {"accepted": True, "reason": None}

    def _can_accept_globally(
        self,
        camera_id: str,
        queue: PerCameraFrameQueue,
        mutation: QueueMutationResult,
    ) -> bool:
        packet_bytes = len(mutation.entry.frame_packet.image_bytes) if mutation.entry else 0
        current_total = self._total_bytes_getter()
        current_camera_bytes = queue.queued_bytes()
        other_reserved = 0
        for other_camera_id, other_queue in self._registry.items():
            if other_camera_id == camera_id:
                continue
            deficit = self._reserved_bytes_per_camera - other_queue.queued_bytes()
            if deficit > 0:
                other_reserved += deficit
        effective_limit = self._max_total_queued_bytes - other_reserved
        prospective = current_total + packet_bytes - self._dropped_bytes(mutation)
        return prospective <= effective_limit and current_camera_bytes <= queue.max_queue_bytes

    def _dropped_bytes(self, mutation: QueueMutationResult) -> int:
        if mutation.dropped_entry is None:
            return 0
        return len(mutation.dropped_entry.frame_packet.image_bytes)

    def _record_rejection(self, reason: EnqueueRejectReason | None) -> None:
        self._metrics.incr("frames_dropped_total")
        if reason is EnqueueRejectReason.QUEUE_FULL_DROP_NEWEST:
            self._metrics.incr("frames_dropped_newest_total")
        elif reason is EnqueueRejectReason.QUEUE_FULL_REJECT:
            self._metrics.incr("frames_dropped_newest_total")
        elif reason is EnqueueRejectReason.GLOBAL_MEMORY_LIMIT:
            self._metrics.incr("frames_dropped_total")

    def _result_from_reason(self, reason: EnqueueRejectReason | None) -> EnqueueResult:
        return self._rejected(reason or EnqueueRejectReason.INTERNAL_ERROR)

    def _rejected(self, reason: EnqueueRejectReason) -> EnqueueResult:
        return {"accepted": False, "reason": reason}