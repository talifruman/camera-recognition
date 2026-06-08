"""Frame Ingestion Gateway runtime orchestration."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import replace

from .config import FrameIngestionGatewayConfig
from .contracts import IngressFrameMessage
from .errors import GatewayLifecycleError
from .health import GatewayHealth
from .metrics import ThreadSafeMetrics
from .normalizer import FramePayloadNormalizer
from .packet_builder import FramePacketBuilder
from .transport import FrameIngressTransport
from .validators import FrameIngestionInputValidator, FrameValidator, detect_timestamp_anomaly
from .workers import CameraIngestionWorker


class _RateLimiter:
    """Simple per-key fixed-window rate limiter for logs."""

    def __init__(self, window_ms: int, max_per_key: int) -> None:
        self._window_ms = window_ms
        self._max_per_key = max_per_key
        self._state: dict[str, tuple[int, int]] = {}
        self._lock = threading.Lock()
        self.suppressed_total = 0

    def allow(self, key: str) -> bool:
        """Return true when one log event is allowed for key."""
        now_ms = int(time.time() * 1000)
        with self._lock:
            start_ms, count = self._state.get(key, (now_ms, 0))
            if now_ms - start_ms >= self._window_ms:
                self._state[key] = (now_ms, 1)
                return True
            if count < self._max_per_key:
                self._state[key] = (start_ms, count + 1)
                return True
            self.suppressed_total += 1
            return False


class FrameIngestionGateway:
    """Ingress-only runtime that publishes canonical FramePacket objects."""

    def __init__(self, transport: FrameIngressTransport) -> None:
        self._transport = transport
        self._config: FrameIngestionGatewayConfig | None = None
        self._configured = False
        self._running = False
        self._accepting_frames = False
        self._state = "STOPPED"
        self._degraded = False
        self._last_error_code = ""
        self._last_error_message = ""
        self._last_error_timestamp_ms = 0
        self._last_sink_reject_reason = ""
        self._lock = threading.Lock()
        self._workers: list[CameraIngestionWorker] = []
        self._metrics = ThreadSafeMetrics()
        self._last_timestamp_by_camera: dict[str, int] = {}
        self._dup_windows: dict[str, deque[str]] = defaultdict(deque)
        self._dup_sets: dict[str, set[str]] = defaultdict(set)
        self._log_limiter: _RateLimiter | None = None

    def configure(self, config: FrameIngestionGatewayConfig) -> None:
        """Configure gateway exactly once and before start."""
        with self._lock:
            if self._running:
                raise GatewayLifecycleError("configure() is not allowed after start()")
            if self._configured:
                raise GatewayLifecycleError("configure() may be called only once")
            self._config = config
            self._configured = True

    def start(self) -> None:
        """Start transport and one worker per configured camera."""
        config = self._require_configured()
        with self._lock:
            if self._running:
                return
            self._running = True
            self._accepting_frames = True
            self._state = "RUNNING"
            self._log_limiter = _RateLimiter(
                window_ms=config.log_rate_limit_window_ms,
                max_per_key=config.log_rate_limit_max_per_key,
            )
        runtime_config = replace(config, on_unknown_camera_rejected=self._on_unknown_camera)
        self._transport.bind_and_start(runtime_config)
        self._workers = self._build_workers(config)
        for worker in self._workers:
            worker.start()

    def stop(self) -> None:
        """Stop transport and workers while honoring stop timeout."""
        config = self._require_configured()
        with self._lock:
            if not self._running:
                self._state = "STOPPED"
                self._accepting_frames = False
                return
            self._accepting_frames = False
        for worker in self._workers:
            worker.stop()
        try:
            self._transport.stop()
        except Exception as exc:  # pragma: no cover
            self._metrics.incr("gateway_receive_cancel_failed_total")
            self._set_error("TRANSPORT_STOP_ERROR", str(exc), state="ERROR")
        timeout_s = config.stop_timeout_ms / 1000.0
        for worker in self._workers:
            worker.join(timeout_s)
        if any(worker.is_alive() for worker in self._workers):
            self._metrics.incr("gateway_stop_timeout_total")
            self._set_error("WORKER_STOP_TIMEOUT", "worker stop timeout", state="DEGRADED")
        with self._lock:
            self._running = False
            if self._state not in {"DEGRADED", "ERROR"}:
                self._state = "STOPPED"

    def health(self) -> GatewayHealth:
        """Return a thread-safe health snapshot for gateway runtime."""
        config = self._config
        assert config is not None
        with self._lock:
            state = self._state
            accepting = self._accepting_frames
            degraded = self._degraded
            last_error_code = self._last_error_code
            last_error_message = self._last_error_message
            last_error_ts = self._last_error_timestamp_ms
            reject_reason = self._last_sink_reject_reason
            active_workers = sum(1 for worker in self._workers if worker.is_alive())
        suppressed = 0 if self._log_limiter is None else self._log_limiter.suppressed_total
        return GatewayHealth(
            state=state,
            transport_state="RUNNING" if self._running else "STOPPED",
            accepting_frames=accepting,
            degraded=degraded,
            last_error_code=last_error_code,
            last_error_message=last_error_message,
            last_error_timestamp_ms=last_error_ts,
            active_ingress_workers=active_workers,
            configured_camera_count=len(config.configured_camera_ids),
            frames_in_total=self._metrics.get_counter("frames_in_total"),
            frames_accepted_total=self._metrics.get_counter("frames_accepted_total"),
            frames_rejected_total=self._metrics.get_counter("frames_rejected_total"),
            frames_published_total=self._metrics.get_counter("frames_published_total"),
            sink_enqueue_rejected_total=self._metrics.get_counter("sink_enqueue_rejected_total"),
            decode_failed_total=self._metrics.get_counter("decode_failed_total"),
            normalization_failed_total=self._metrics.get_counter("normalization_failed_total"),
            unsupported_source_format_total=self._metrics.get_counter(
                "unsupported_source_format_total"
            ),
            unknown_camera_rejected_total=self._metrics.get_counter(
                "unknown_camera_rejected_total"
            ),
            duplicate_frame_id_observed_total=self._metrics.get_counter(
                "duplicate_frame_id_observed_total"
            ),
            timestamp_anomaly_total=self._metrics.get_counter("timestamp_anomaly_total"),
            gateway_stop_timeout_total=self._metrics.get_counter("gateway_stop_timeout_total"),
            gateway_receive_cancel_failed_total=self._metrics.get_counter(
                "gateway_receive_cancel_failed_total"
            ),
            ingest_latency_ms_avg=self._metrics.get_latency("total_ingest_latency_ms"),
            decode_latency_ms_avg=self._metrics.get_latency("decode_latency_ms"),
            normalize_latency_ms_avg=self._metrics.get_latency("normalize_latency_ms"),
            worker_error_total=self._metrics.get_counter("worker_error_total"),
            suppressed_log_total=suppressed,
            last_sink_reject_reason=reject_reason,
        )

    def _build_workers(self, config: FrameIngestionGatewayConfig) -> list[CameraIngestionWorker]:
        input_validator = FrameIngestionInputValidator()
        frame_validator = FrameValidator(config)
        normalizer = FramePayloadNormalizer()
        packet_builder = FramePacketBuilder()
        return [
            CameraIngestionWorker(
                camera_id=camera_id,
                transport=self._transport,
                sink=config.sink,
                input_validator=input_validator,
                frame_validator=frame_validator,
                normalizer=normalizer,
                packet_builder=packet_builder,
                metrics=self._metrics,
                should_reject_message=self._should_reject_message,
                on_sink_rejected=self._on_sink_rejected,
                on_loggable_error=self._on_loggable_error,
                on_worker_exception=self._on_worker_exception,
            )
            for camera_id in config.configured_camera_ids
        ]

    def _should_reject_message(self, message: IngressFrameMessage) -> bool:
        config = self._require_configured()
        anomaly = detect_timestamp_anomaly(
            message=message,
            previous_timestamp_ms=self._last_timestamp_by_camera.get(message.camera_id),
            config=config,
        )
        self._last_timestamp_by_camera[message.camera_id] = message.timestamp_ms
        if anomaly:
            self._metrics.incr("timestamp_anomaly_total")
            self._on_loggable_error("timestamp_anomaly", message.camera_id)
            if config.strict_timestamp_validation_enabled:
                return True
        if not config.duplicate_frame_tracking_enabled:
            return False
        window = self._dup_windows[message.camera_id]
        seen = self._dup_sets[message.camera_id]
        if message.frame_id in seen:
            self._metrics.incr("duplicate_frame_id_observed_total")
        window.append(message.frame_id)
        seen.add(message.frame_id)
        while len(window) > config.duplicate_frame_tracking_window_size:
            evicted = window.popleft()
            if evicted not in window:
                seen.discard(evicted)
        return False

    def _on_unknown_camera(self, _camera_id: str) -> None:
        self._metrics.incr("unknown_camera_rejected_total")
        self._metrics.incr("frames_rejected_total")
        self._on_loggable_error("unknown_camera", "gateway")

    def _on_sink_rejected(self, reason_code: str) -> None:
        with self._lock:
            self._last_sink_reject_reason = reason_code

    def _on_worker_exception(self, camera_id: str, exc: Exception) -> None:
        self._metrics.incr("worker_error_total")
        self._set_error(
            code="WORKER_EXCEPTION",
            message=f"worker {camera_id} exception: {exc}",
            state="DEGRADED",
        )

    def _on_loggable_error(self, error_type: str, camera_id: str) -> None:
        limiter = self._log_limiter
        if limiter is None:
            return
        key = f"{error_type}:{camera_id}"
        limiter.allow(key)

    def _set_error(self, code: str, message: str, state: str) -> None:
        with self._lock:
            self._last_error_code = code
            self._last_error_message = message
            self._last_error_timestamp_ms = int(time.time() * 1000)
            self._state = state
            self._degraded = state in {"DEGRADED", "ERROR"}

    def _require_configured(self) -> FrameIngestionGatewayConfig:
        if self._config is None:
            raise GatewayLifecycleError("configure() must be called before start()")
        return self._config
