"""Transport abstractions and in-memory ingress adapter."""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Protocol

from .config import FrameIngestionGatewayConfig
from .contracts import IngressFrameMessage


class FrameIngressTransport(Protocol):
    """Ingress transport contract owned by the gateway."""

    def bind_and_start(self, config: FrameIngestionGatewayConfig) -> None:
        """Bind transport resources and start receiving."""

    def stop(self) -> None:
        """Stop transport and unblock receive loops."""

    def receive_message(self, camera_id: str) -> IngressFrameMessage | None:
        """Receive next message for one camera worker."""


class InMemoryFrameIngressTransport:
    """Thread-safe in-memory transport used by tests."""

    def __init__(self, block_receive_until_stop: bool = False) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._queues: dict[str, deque[IngressFrameMessage]] = defaultdict(deque)
        self._running = False
        self._configured_cameras: set[str] = set()
        self._on_unknown_camera: Callable[[str], None] | None = None
        self._block_receive_until_stop = block_receive_until_stop

    def bind_and_start(self, config: FrameIngestionGatewayConfig) -> None:
        """Initialize queues and start in-memory transport."""
        with self._lock:
            self._configured_cameras = set(config.configured_camera_ids)
            self._queues = defaultdict(deque)
            self._running = True
            self._on_unknown_camera = config.on_unknown_camera_rejected

    def inject_message(self, message: IngressFrameMessage) -> None:
        """Inject a message into transport for tests."""
        with self._condition:
            if not self._running:
                return
            if message.camera_id not in self._configured_cameras:
                if self._on_unknown_camera is not None:
                    self._on_unknown_camera(message.camera_id)
                return
            self._queues[message.camera_id].append(message)
            self._condition.notify_all()

    def receive_message(self, camera_id: str) -> IngressFrameMessage | None:
        """Block until message for camera is available or transport stops."""
        with self._condition:
            while self._running:
                queue = self._queues.get(camera_id)
                if queue:
                    return queue.popleft()
                if self._block_receive_until_stop:
                    self._condition.wait(timeout=0.05)
                    continue
                self._condition.wait(timeout=0.05)
            return None

    def stop(self) -> None:
        """Stop transport and wake all blocked workers."""
        with self._condition:
            self._running = False
            self._condition.notify_all()
