"""Thread-safe metrics utilities for gateway runtime."""

from __future__ import annotations

import threading
from collections import defaultdict


class RunningAverage:
    """Thread-safe running average accumulator."""

    def __init__(self) -> None:
        self._total = 0.0
        self._count = 0

    def add(self, value: float) -> None:
        """Add one measurement to running average."""
        self._total += value
        self._count += 1

    def value(self) -> float:
        """Return current average value."""
        if self._count == 0:
            return 0.0
        return self._total / self._count


class ThreadSafeMetrics:
    """Central thread-safe counters and latency metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._latency: dict[str, RunningAverage] = defaultdict(RunningAverage)

    def incr(self, key: str, amount: int = 1) -> None:
        """Increment a named counter."""
        with self._lock:
            self._counters[key] += amount

    def add_latency(self, key: str, value_ms: float) -> None:
        """Record one latency value in milliseconds."""
        with self._lock:
            self._latency[key].add(value_ms)

    def get_counter(self, key: str) -> int:
        """Read one counter value."""
        with self._lock:
            return self._counters.get(key, 0)

    def get_latency(self, key: str) -> float:
        """Read one average latency value."""
        with self._lock:
            if key not in self._latency:
                return 0.0
            return self._latency[key].value()
