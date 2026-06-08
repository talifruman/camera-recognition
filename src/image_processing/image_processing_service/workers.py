"""Processing workers for per-camera Image Processing Service lanes."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

from image_processing.shared.contracts import FramePacket

from .config import ImageProcessingServiceConfig
from .metrics import ThreadSafeServiceMetrics
from .queues import PerCameraFrameQueue


class ResultHandler(Protocol):
    """Non-blocking processing result handler abstraction."""

    def handle(self, camera_id: str, frame_packet: FramePacket, output: Any) -> None:
        """Handle a processing result."""


class NoOpResultHandler:
    """Default result handler used when no downstream handler is configured."""

    def handle(self, camera_id: str, frame_packet: FramePacket, output: Any) -> None:
        """Ignore outputs while preserving worker flow."""


@dataclass(slots=True)
class WorkerSnapshot:
    """Health snapshot for one processing worker."""

    worker_alive: bool
    last_frame_started_at_ms: int
    last_frame_completed_at_ms: int
    last_error_code: str


class ProcessingWorker:
    """Per-camera worker that consumes from a matching queue and invokes RPM."""

    def __init__(
        self,
        camera_id: str,
        queue: PerCameraFrameQueue,
        rpm: Any,
        metrics: ThreadSafeServiceMetrics,
        config: ImageProcessingServiceConfig,
        result_handler: ResultHandler | None = None,
    ) -> None:
        self.camera_id = camera_id
        self._queue = queue
        self._rpm = rpm
        self._metrics = metrics
        self._config = config
        self._result_handler = result_handler or NoOpResultHandler()
        self._run_event = threading.Event()
        self._run_event.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._lock = threading.Lock()
        self._last_frame_started_at_ms = 0
        self._last_frame_completed_at_ms = 0
        self._last_error_code = ""
        self._warmup_state = "COLD"
        self._warmup_completed_at_ms = 0
        self._cold_start_frame_total = 0

    def start(self) -> None:
        """Start the worker thread."""
        self._thread.start()

    def stop(self) -> None:
        """Request worker shutdown."""
        self._run_event.clear()

    def join(self, timeout_s: float) -> None:
        """Join the worker thread."""
        self._thread.join(timeout_s)

    def is_alive(self) -> bool:
        """Return whether the worker thread is alive."""
        return self._thread.is_alive()

    def snapshot(self) -> WorkerSnapshot:
        """Return the current worker state snapshot."""
        with self._lock:
            return WorkerSnapshot(
                worker_alive=self.is_alive(),
                last_frame_started_at_ms=self._last_frame_started_at_ms,
                last_frame_completed_at_ms=self._last_frame_completed_at_ms,
                last_error_code=self._last_error_code,
            )

    def warmup_state(self) -> str:
        """Return the current warmup state."""
        with self._lock:
            return self._warmup_state

    def warmup_completed_at_ms(self) -> int:
        """Return the warmup completion timestamp."""
        with self._lock:
            return self._warmup_completed_at_ms

    def cold_start_frame_total(self) -> int:
        """Return the number of cold-start frames observed by the worker."""
        with self._lock:
            return self._cold_start_frame_total

    def _run(self) -> None:
        while self._run_event.is_set() or self._queue.depth() > 0:
            entry = self._queue.dequeue(timeout_s=0.05)
            if entry is None:
                continue
            self._process_entry(entry)

    def _process_entry(self, entry) -> None:
        now_ms = int(time.time() * 1000)
        frame_packet = entry.frame_packet
        wait_ms = now_ms - entry.enqueued_at_ms
        self._metrics.record_queue_wait(self.camera_id, wait_ms)
        self._metrics.add_latency("queue_dequeue_latency_ms", 0.0)
        with self._lock:
            self._last_frame_started_at_ms = now_ms
        if self._is_stale(frame_packet.timestamp_ms):
            self._metrics.incr("stale_frames_dropped_total")
            self._metrics.incr("stale_preprocessing_drop_total")
            return
        try:
            if self._is_stale(frame_packet.timestamp_ms):
                self._metrics.incr("stale_frames_dropped_total")
                self._metrics.incr("stale_preprocessing_drop_total")
                return
            start_ms = time.perf_counter()
            output = self._rpm.process_frame(frame_packet)
            elapsed_ms = (time.perf_counter() - start_ms) * 1000.0
            self._metrics.add_latency("processing_worker_latency_ms", elapsed_ms)
            self._metrics.add_latency("frame_processing_duration_ms", elapsed_ms)
            self._metrics.incr("frames_dequeued_total")
            self._metrics.incr_camera("dequeued_per_camera", self.camera_id)
            self._metrics.incr_camera("processed_per_camera", self.camera_id)
            if self._mark_cold_start():
                self._metrics.incr("cold_start_frame_total")
            self._result_handler.handle(self.camera_id, frame_packet, output)
            self._maybe_mark_warmup(frame_packet.timestamp_ms)
            if self._is_stale(frame_packet.timestamp_ms):
                self._metrics.incr("stale_completed_processing_total")
            with self._lock:
                self._last_frame_completed_at_ms = int(time.time() * 1000)
        except Exception as exc:  # pragma: no cover
            with self._lock:
                self._last_error_code = type(exc).__name__
            self._metrics.incr("worker_error_total")

    def _is_stale(self, timestamp_ms: int) -> bool:
        if not self._config.stale_frame_drop_enabled:
            return False
        now_ms = int(time.time() * 1000)
        return now_ms - timestamp_ms > self._config.max_frame_age_ms

    def _mark_cold_start(self) -> bool:
        with self._lock:
            if self._warmup_state == "COLD":
                self._warmup_state = "WARMING"
                self._cold_start_frame_total += 1
                return True
        return False

    def _maybe_mark_warmup(self, timestamp_ms: int) -> None:
        with self._lock:
            if self._warmup_state == "WARMING":
                self._warmup_state = "WARMED"
                self._warmup_completed_at_ms = int(time.time() * 1000)
            self._last_frame_completed_at_ms = max(self._last_frame_completed_at_ms, timestamp_ms)