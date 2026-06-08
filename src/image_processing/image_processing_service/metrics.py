"""Non-blocking in-memory metrics for the Image Processing Service."""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from statistics import quantiles


class ThreadSafeServiceMetrics:
    """Thread-safe metric accumulator used by IPS runtime hot paths."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._latencies: dict[str, list[float]] = defaultdict(list)
        self._queue_wait_samples: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=4096)
        )
        self._per_camera_counters: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def incr(self, name: str, amount: int = 1) -> None:
        """Increment a service-wide counter."""
        with self._lock:
            self._counters[name] += amount

    def incr_camera(self, name: str, camera_id: str, amount: int = 1) -> None:
        """Increment a per-camera counter."""
        with self._lock:
            self._per_camera_counters[camera_id][name] += amount

    def add_latency(self, name: str, value_ms: float) -> None:
        """Record a latency sample in milliseconds."""
        with self._lock:
            self._latencies[name].append(float(value_ms))

    def record_queue_wait(self, camera_id: str, value_ms: float) -> None:
        """Record queue residence time for a camera lane."""
        with self._lock:
            self._queue_wait_samples[camera_id].append(float(value_ms))

    def get_counter(self, name: str) -> int:
        """Return the current service-wide counter value."""
        with self._lock:
            return int(self._counters.get(name, 0))

    def get_camera_counter(self, camera_id: str, name: str) -> int:
        """Return the current per-camera counter value."""
        with self._lock:
            return int(self._per_camera_counters.get(camera_id, {}).get(name, 0))

    def get_latency_average(self, name: str) -> float:
        """Return the average of the recorded latency samples."""
        with self._lock:
            samples = list(self._latencies.get(name, []))
        if not samples:
            return 0.0
        return float(sum(samples) / len(samples))

    def get_latency_max(self, name: str) -> float:
        """Return the maximum recorded latency sample."""
        with self._lock:
            samples = list(self._latencies.get(name, []))
        return float(max(samples)) if samples else 0.0

    def get_queue_wait_average(self) -> float:
        """Return the average queue wait time across all cameras."""
        samples = self._all_queue_wait_samples()
        return float(sum(samples) / len(samples)) if samples else 0.0

    def get_queue_wait_max(self) -> float:
        """Return the maximum queue wait time across all cameras."""
        samples = self._all_queue_wait_samples()
        return float(max(samples)) if samples else 0.0

    def get_queue_wait_p95(self, camera_id: str) -> float:
        """Return the p95 queue wait time for one camera lane."""
        with self._lock:
            samples = list(self._queue_wait_samples.get(camera_id, []))
        if not samples:
            return 0.0
        if len(samples) < 20:
            return float(max(samples))
        return float(quantiles(samples, n=20)[18])

    def get_all_camera_ids(self) -> list[str]:
        """Return all cameras that have recorded metrics."""
        with self._lock:
            return sorted(self._per_camera_counters.keys() | self._queue_wait_samples.keys())

    def _all_queue_wait_samples(self) -> list[float]:
        with self._lock:
            samples = [value for queue in self._queue_wait_samples.values() for value in queue]
        return samples