"""Per-camera bounded queues and registry for the Image Processing Service."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

from image_processing.shared.contracts import EnqueueRejectReason, FramePacket

from .config import OverflowPolicy


@dataclass(frozen=True, slots=True)
class QueueEntry:
    """Immutable queue entry with enqueue timestamp metadata."""

    frame_packet: FramePacket
    enqueued_at_ms: int


@dataclass(frozen=True, slots=True)
class QueueMutationResult:
    """Result of an atomic queue enqueue mutation."""

    accepted: bool
    reason: EnqueueRejectReason | None
    entry: QueueEntry | None
    dropped_entry: QueueEntry | None
    queue_depth: int
    queue_bytes: int


class PerCameraFrameQueue:
    """Thread-safe bounded FIFO queue for one camera lane."""

    def __init__(self, camera_id: str, max_queue_size: int, max_queue_bytes: int) -> None:
        self.camera_id = camera_id
        self.max_queue_size = max_queue_size
        self.max_queue_bytes = max_queue_bytes
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._entries: deque[QueueEntry] = deque()
        self._queued_bytes = 0

    def enqueue_local(
        self,
        frame_packet: FramePacket,
        overflow_policy: OverflowPolicy,
    ) -> QueueMutationResult:
        """Mutate queue state for a local enqueue attempt."""
        now_ms = int(time.time() * 1000)
        entry = QueueEntry(frame_packet=frame_packet, enqueued_at_ms=now_ms)
        packet_bytes = len(frame_packet.image_bytes)
        with self._condition:
            if self._can_fit(entry, packet_bytes):
                self._append_entry(entry, packet_bytes)
                return self._accepted(entry, None)
            return self._handle_overflow(entry, packet_bytes, overflow_policy)

    def rollback_enqueue(self, result: QueueMutationResult) -> None:
        """Undo a previously accepted enqueue mutation."""
        with self._condition:
            if result.dropped_entry is not None:
                self._entries.appendleft(result.dropped_entry)
                self._queued_bytes += len(result.dropped_entry.frame_packet.image_bytes)
            if result.entry is not None:
                self._entries.pop()
                self._queued_bytes -= len(result.entry.frame_packet.image_bytes)
            self._condition.notify_all()

    def dequeue(self, timeout_s: float | None = None) -> QueueEntry | None:
        """Remove and return the oldest queued frame entry."""
        with self._condition:
            if timeout_s is None:
                if not self._entries:
                    return None
                return self._pop_left()
            deadline = time.monotonic() + timeout_s
            while not self._entries:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)
            return self._pop_left()

    def clear(self) -> list[QueueEntry]:
        """Drop every queued frame and return the dropped entries."""
        with self._condition:
            dropped = list(self._entries)
            self._entries.clear()
            self._queued_bytes = 0
            self._condition.notify_all()
            return dropped

    def depth(self) -> int:
        """Return the current queue depth."""
        with self._lock:
            return len(self._entries)

    def queued_bytes(self) -> int:
        """Return the current queued byte count."""
        with self._lock:
            return self._queued_bytes

    def oldest_frame_age_ms(self) -> int:
        """Return the age of the oldest queued frame in milliseconds."""
        with self._lock:
            if not self._entries:
                return 0
            return int(time.time() * 1000) - self._entries[0].frame_packet.timestamp_ms

    def queue_wait_age_ms(self) -> int:
        """Return the queue wait time of the oldest entry in milliseconds."""
        with self._lock:
            if not self._entries:
                return 0
            return int(time.time() * 1000) - self._entries[0].enqueued_at_ms

    def _can_fit(self, entry: QueueEntry, packet_bytes: int) -> bool:
        return (
            len(self._entries) < self.max_queue_size
            and self._queued_bytes + packet_bytes <= self.max_queue_bytes
        )

    def _append_entry(self, entry: QueueEntry, packet_bytes: int) -> None:
        self._entries.append(entry)
        self._queued_bytes += packet_bytes
        self._condition.notify_all()

    def _handle_overflow(
        self,
        entry: QueueEntry,
        packet_bytes: int,
        overflow_policy: OverflowPolicy,
    ) -> QueueMutationResult:
        if overflow_policy is OverflowPolicy.DROP_NEWEST:
            return self._rejected(EnqueueRejectReason.QUEUE_FULL_DROP_NEWEST)
        if overflow_policy is OverflowPolicy.REJECT:
            return self._rejected(EnqueueRejectReason.QUEUE_FULL_REJECT)
        if not self._entries:
            return self._rejected(EnqueueRejectReason.GLOBAL_MEMORY_LIMIT)
        dropped_entry = self._entries.popleft()
        self._queued_bytes -= len(dropped_entry.frame_packet.image_bytes)
        if self._queued_bytes + packet_bytes > self.max_queue_bytes:
            self._entries.appendleft(dropped_entry)
            self._queued_bytes += len(dropped_entry.frame_packet.image_bytes)
            return self._rejected(EnqueueRejectReason.GLOBAL_MEMORY_LIMIT)
        self._append_entry(entry, packet_bytes)
        return QueueMutationResult(True, None, entry, dropped_entry, len(self._entries), self._queued_bytes)

    def _accepted(self, entry: QueueEntry, dropped: QueueEntry | None) -> QueueMutationResult:
        return QueueMutationResult(True, None, entry, dropped, len(self._entries), self._queued_bytes)

    def _rejected(self, reason: EnqueueRejectReason) -> QueueMutationResult:
        return QueueMutationResult(False, reason, None, None, len(self._entries), self._queued_bytes)

    def _pop_left(self) -> QueueEntry:
        entry = self._entries.popleft()
        self._queued_bytes -= len(entry.frame_packet.image_bytes)
        return entry


class FrameQueueRegistry:
    """Read-mostly mapping from camera_id to per-camera queues."""

    def __init__(self, queues: dict[str, PerCameraFrameQueue]) -> None:
        self._queues = dict(queues)

    def get_queue(self, camera_id: str) -> PerCameraFrameQueue | None:
        """Return the queue for the requested camera, if present."""
        return self._queues.get(camera_id)

    def camera_ids(self) -> list[str]:
        """Return configured camera identifiers."""
        return sorted(self._queues)

    def items(self) -> list[tuple[str, PerCameraFrameQueue]]:
        """Return camera/queue pairs for iteration."""
        return [(camera_id, self._queues[camera_id]) for camera_id in sorted(self._queues)]