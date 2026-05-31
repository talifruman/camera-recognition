"""Main Image Processing Service orchestration."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from image_processing.frame_ingestion_gateway import (
    FrameIngestionGateway,
    FrameIngestionGatewayConfig,
    InMemoryFrameIngressTransport,
)
from image_processing.recognition_pipeline_manager import RecognitionPipelineManager
from image_processing.shared.contracts import FramePacket

from .config import ImageProcessingServiceConfig
from .errors import ImageProcessingServiceLifecycleError
from .health import ImageProcessingServiceHealth, LaneHealth, WorkerStatus
from .lifecycle import ServiceState
from .metrics import ThreadSafeServiceMetrics
from .queues import FrameQueueRegistry, PerCameraFrameQueue
from .sink import ServiceFramePacketSink, SinkState
from .workers import ProcessingWorker, ResultHandler


@dataclass(slots=True)
class _ServiceRuntime:
    """Runtime wiring held after configure()."""

    config: ImageProcessingServiceConfig
    registry: FrameQueueRegistry
    sink: ServiceFramePacketSink
    gateway: FrameIngestionGateway
    workers: dict[str, ProcessingWorker]


class ImageProcessingService:
    """Top-level lifecycle owner for gateway ingestion and RPM processing."""

    def __init__(
        self,
        rpm: RecognitionPipelineManager,
        transport: InMemoryFrameIngressTransport | None = None,
        result_handler: ResultHandler | None = None,
    ) -> None:
        self._rpm = rpm
        self._transport = transport or InMemoryFrameIngressTransport()
        self._result_handler = result_handler
        self._metrics = ThreadSafeServiceMetrics()
        self._state = ServiceState.STOPPED
        self._configured = False
        self._runtime: _ServiceRuntime | None = None
        self._total_queued_bytes = 0
        self._state_lock = threading.Lock()
        self._sink_state = SinkState()

    def configure(self, config: ImageProcessingServiceConfig) -> None:
        """Validate configuration and prepare runtime structures."""
        if self._configured:
            raise ImageProcessingServiceLifecycleError("configure() may only be called once")
        config.validate()
        registry = self._build_registry(config)
        sink = ServiceFramePacketSink(
            registry=registry,
            overflow_policy=config.overflow_policy,
            metrics=self._metrics,
            state=self._sink_state,
            total_bytes_getter=self._get_total_queued_bytes,
            total_bytes_setter=self._set_total_queued_bytes,
            reserved_bytes_per_camera=config.reserved_queue_bytes_per_camera,
            max_total_queued_bytes=config.max_total_queued_bytes,
        )
        gateway = FrameIngestionGateway(transport=self._transport)
        gateway.configure(self._build_gateway_config(config, sink))
        workers = self._build_workers(config, registry)
        self._runtime = _ServiceRuntime(config, registry, sink, gateway, workers)
        self._configured = True
        self._state = ServiceState.INITIALIZING

    def start(self) -> None:
        """Start the gateway and per-camera processing workers."""
        runtime = self._require_runtime()
        with self._state_lock:
            if self._state is ServiceState.RUNNING:
                return
        runtime.gateway.start()
        for worker in runtime.workers.values():
            worker.start()
        with self._state_lock:
            self._state = ServiceState.RUNNING

    def stop(self, drain: bool) -> None:
        """Stop ingestion and processing according to the drain policy."""
        runtime = self._require_runtime()
        with self._state_lock:
            self._state = ServiceState.STOPPING
        runtime.sink.close_acceptance()
        runtime.gateway.stop()
        if drain:
            self._drain_queues(runtime, self._runtime.config.max_drain_timeout_ms)
        else:
            self._drop_queues(runtime)
        self._stop_workers(runtime)
        with self._state_lock:
            self._state = ServiceState.STOPPED

    def health(self) -> ImageProcessingServiceHealth:
        """Return a consistent service health snapshot."""
        runtime = self._require_runtime()
        queue_depths: dict[str, int] = {}
        queue_bytes: dict[str, int] = {}
        oldest_ages: dict[str, int] = {}
        p95s: dict[str, float] = {}
        lane_health: dict[str, LaneHealth] = {}
        worker_alive: dict[str, bool] = {}
        last_started: dict[str, int] = {}
        last_completed: dict[str, int] = {}
        last_errors: dict[str, str] = {}
        warmup_state: dict[str, str] = {}
        warmup_completed: dict[str, int] = {}
        cold_start_total = 0
        for camera_id, queue in runtime.registry.items():
            queue_depths[camera_id] = queue.depth()
            queue_bytes[camera_id] = queue.queued_bytes()
            oldest_ages[camera_id] = queue.oldest_frame_age_ms()
            p95s[camera_id] = self._metrics.get_queue_wait_p95(camera_id)
            worker = runtime.workers[camera_id]
            snapshot = worker.snapshot()
            worker_alive[camera_id] = snapshot.worker_alive
            last_started[camera_id] = snapshot.last_frame_started_at_ms
            last_completed[camera_id] = snapshot.last_frame_completed_at_ms
            last_errors[camera_id] = snapshot.last_error_code
            warmup_state[camera_id] = worker.warmup_state()
            warmup_completed[camera_id] = worker.warmup_completed_at_ms()
            cold_start_total += worker.cold_start_frame_total()
            lane_health[camera_id] = LaneHealth(
                queue_depth=queue_depths[camera_id],
                queue_bytes=queue_bytes[camera_id],
                oldest_frame_age_ms=oldest_ages[camera_id],
                queue_age_p95_ms=p95s[camera_id],
                worker_status=WorkerStatus(
                    worker_alive=snapshot.worker_alive,
                    last_frame_started_at_ms=snapshot.last_frame_started_at_ms,
                    last_frame_completed_at_ms=snapshot.last_frame_completed_at_ms,
                    last_error_code=snapshot.last_error_code,
                ),
                warmup_state=warmup_state[camera_id],
                warmup_completed_at_ms=warmup_completed[camera_id],
            )
        state = self._state.value
        return ImageProcessingServiceHealth(
            state=state,
            configured_camera_count=len(runtime.registry.camera_ids()),
            active_processing_workers=sum(1 for alive in worker_alive.values() if alive),
            total_queued_frames=sum(queue_depths.values()),
            total_queued_bytes=sum(queue_bytes.values()),
            queue_depth_per_camera=queue_depths,
            queue_bytes_per_camera=queue_bytes,
            frames_dropped_oldest_total=self._metrics.get_counter("frames_dropped_oldest_total"),
            frames_dropped_newest_total=self._metrics.get_counter("frames_dropped_newest_total"),
            unknown_camera_rejected_total=self._metrics.get_counter("unknown_camera_rejected_total"),
            stale_frames_dropped_total=self._metrics.get_counter("stale_frames_dropped_total"),
            worker_error_total=self._metrics.get_counter("worker_error_total"),
            worker_timeout_total=self._metrics.get_counter("worker_timeout_total"),
            degraded_reason_code="" if state != ServiceState.DEGRADED.value else "WORKER_TIMEOUT",
            degraded_reason_message="" if state != ServiceState.DEGRADED.value else "service degraded",
            lane_health_by_camera_id=lane_health,
            queue_age_p95_per_camera=p95s,
            average_queue_wait_ms=self._metrics.get_queue_wait_average(),
            max_queue_wait_ms=self._metrics.get_queue_wait_max(),
            oldest_frame_age_ms_per_camera=oldest_ages,
            worker_alive=worker_alive,
            last_frame_started_at_ms=last_started,
            last_frame_completed_at_ms=last_completed,
            last_error_code=last_errors,
            cold_start_frame_total=cold_start_total,
            camera_warmup_state=warmup_state,
            warmup_completed_at_ms=warmup_completed,
        )

    def _build_registry(self, config: ImageProcessingServiceConfig) -> FrameQueueRegistry:
        queues = {
            camera_id: PerCameraFrameQueue(
                camera_id=camera_id,
                max_queue_size=config.max_queue_size_per_camera,
                max_queue_bytes=config.max_queue_bytes_per_camera,
            )
            for camera_id in config.camera_ids
        }
        return FrameQueueRegistry(queues)

    def _build_workers(
        self,
        config: ImageProcessingServiceConfig,
        registry: FrameQueueRegistry,
    ) -> dict[str, ProcessingWorker]:
        workers: dict[str, ProcessingWorker] = {}
        for camera_id in registry.camera_ids():
            workers[camera_id] = ProcessingWorker(
                camera_id=camera_id,
                queue=registry.get_queue(camera_id),
                rpm=self._rpm,
                metrics=self._metrics,
                config=config,
                result_handler=self._result_handler,
            )
        return workers

    def _build_gateway_config(
        self,
        config: ImageProcessingServiceConfig,
        sink: ServiceFramePacketSink,
    ) -> FrameIngestionGatewayConfig:
        return FrameIngestionGatewayConfig(
            bind_address="127.0.0.1:0",
            service_name="image-processing-service",
            max_concurrent_streams=max(1, len(config.camera_ids)),
            configured_camera_ids=list(config.camera_ids),
            supported_source_formats=["RGB", "BGR", "GRAY8", "YUV420", "NV12", "YUY2", "JPEG", "MJPEG"],
            max_frame_width=8192,
            max_frame_height=8192,
            max_payload_bytes=64 * 1024 * 1024,
            allowed_timestamp_skew_ms=500,
            allowed_timestamp_lag_ms=5_000,
            strict_timestamp_validation_enabled=False,
            duplicate_frame_tracking_enabled=False,
            duplicate_frame_tracking_window_size=32,
            stop_timeout_ms=5_000,
            log_rate_limit_window_ms=5_000,
            log_rate_limit_max_per_key=5,
            sink=sink,
        )

    def _drain_queues(self, runtime: _ServiceRuntime, timeout_ms: int) -> None:
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            if all(runtime.registry.get_queue(camera_id).depth() == 0 for camera_id in runtime.registry.camera_ids()):
                break
            time.sleep(0.01)
        for camera_id in runtime.registry.camera_ids():
            dropped = runtime.registry.get_queue(camera_id).clear()
            self._metrics.incr("shutdown_dropped_frames_total", len(dropped))

    def _drop_queues(self, runtime: _ServiceRuntime) -> None:
        for camera_id in runtime.registry.camera_ids():
            dropped = runtime.registry.get_queue(camera_id).clear()
            self._metrics.incr("shutdown_dropped_frames_total", len(dropped))

    def _stop_workers(self, runtime: _ServiceRuntime) -> None:
        for worker in runtime.workers.values():
            worker.stop()
        timeout_s = self._runtime.config.worker_shutdown_timeout_ms / 1000.0
        for worker in runtime.workers.values():
            worker.join(timeout_s)
        if any(worker.is_alive() for worker in runtime.workers.values()):
            self._metrics.incr("service_worker_stop_timeout_total")
            self._metrics.incr("worker_timeout_total")
            with self._state_lock:
                self._state = ServiceState.DEGRADED

    def _get_total_queued_bytes(self) -> int:
        return self._total_queued_bytes

    def _set_total_queued_bytes(self, value: int) -> None:
        self._total_queued_bytes = value

    def _require_runtime(self) -> _ServiceRuntime:
        if self._runtime is None:
            raise ImageProcessingServiceLifecycleError("configure() must be called before use")
        return self._runtime