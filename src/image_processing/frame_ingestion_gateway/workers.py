"""Per-camera ingestion worker implementation."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from image_processing.shared.contracts import FramePacketSink

from .contracts import IngressFrameMessage
from .errors import (
    FrameDecodeError,
    FrameNormalizationError,
    PayloadSizeMismatchError,
    StructuralValidationError,
    UnsupportedSourceFormatError,
)
from .metrics import ThreadSafeMetrics
from .normalizer import FramePayloadNormalizer
from .packet_builder import FramePacketBuilder
from .transport import FrameIngressTransport
from .validators import FrameIngestionInputValidator, FrameValidator


class CameraIngestionWorker:
    """Single-camera worker that validates, normalizes, and publishes frames."""

    def __init__(
        self,
        camera_id: str,
        transport: FrameIngressTransport,
        sink: FramePacketSink,
        input_validator: FrameIngestionInputValidator,
        frame_validator: FrameValidator,
        normalizer: FramePayloadNormalizer,
        packet_builder: FramePacketBuilder,
        metrics: ThreadSafeMetrics,
        should_reject_message: Callable[[IngressFrameMessage], bool],
        on_sink_rejected: Callable[[str], None],
        on_loggable_error: Callable[[str, str], None],
        on_worker_exception: Callable[[str, Exception], None],
    ) -> None:
        self._camera_id = camera_id
        self._transport = transport
        self._sink = sink
        self._input_validator = input_validator
        self._frame_validator = frame_validator
        self._normalizer = normalizer
        self._packet_builder = packet_builder
        self._metrics = metrics
        self._should_reject_message = should_reject_message
        self._on_sink_rejected = on_sink_rejected
        self._on_loggable_error = on_loggable_error
        self._on_worker_exception = on_worker_exception
        self._run_flag = threading.Event()
        self._run_flag.set()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        """Start worker thread."""
        self._thread.start()

    def stop(self) -> None:
        """Request worker shutdown."""
        self._run_flag.clear()

    def join(self, timeout_s: float) -> None:
        """Join worker thread for at most timeout seconds."""
        self._thread.join(timeout_s)

    def is_alive(self) -> bool:
        """Return true when worker thread is alive."""
        return self._thread.is_alive()

    def _run(self) -> None:
        while self._run_flag.is_set():
            ingest_start = time.perf_counter()
            message = self._transport.receive_message(self._camera_id)
            if message is None:
                continue
            self._metrics.incr("frames_in_total")
            try:
                self._process_message(message)
            except Exception as exc:  # pragma: no cover
                self._on_worker_exception(self._camera_id, exc)
            finally:
                elapsed_ms = (time.perf_counter() - ingest_start) * 1000.0
                self._metrics.add_latency("total_ingest_latency_ms", elapsed_ms)

    def _process_message(self, message: IngressFrameMessage) -> None:
        if message.camera_id != self._camera_id:
            self._metrics.incr("unknown_camera_rejected_total")
            self._metrics.incr("frames_rejected_total")
            self._on_loggable_error("unknown_camera", self._camera_id)
            return
        try:
            structural_start = time.perf_counter()
            self._input_validator.validate(message)
            structural_ms = (time.perf_counter() - structural_start) * 1000.0
            self._metrics.add_latency("structural_validation_latency_ms", structural_ms)

            format_start = time.perf_counter()
            self._frame_validator.validate(message)
            format_ms = (time.perf_counter() - format_start) * 1000.0
            self._metrics.add_latency("format_validation_latency_ms", format_ms)

            if self._should_reject_message(message):
                self._metrics.incr("frames_rejected_total")
                return
            decode_start = time.perf_counter()
            normalized = self._normalizer.normalize(message)
            decode_ms = (time.perf_counter() - decode_start) * 1000.0
            self._metrics.add_latency("decode_latency_ms", decode_ms)
            self._metrics.add_latency("normalize_latency_ms", decode_ms)
            self._metrics.add_latency("receive_latency_ms", structural_ms + format_ms)

            build_start = time.perf_counter()
            packet = self._packet_builder.build(message, normalized)
            build_ms = (time.perf_counter() - build_start) * 1000.0
            self._metrics.add_latency("build_packet_latency_ms", build_ms)

            self._metrics.incr("frames_accepted_total")

            enqueue_start = time.perf_counter()
            result = self._sink.enqueue(packet)
            enqueue_ms = (time.perf_counter() - enqueue_start) * 1000.0
            self._metrics.add_latency("sink_enqueue_latency_ms", enqueue_ms)
            if result.get("accepted", False):
                self._metrics.incr("frames_published_total")
                return
            self._metrics.incr("sink_enqueue_rejected_total")
            self._metrics.incr("frames_rejected_total")
            reason = result.get("reason")
            reason_code = "REJECTED" if reason is None else reason.value
            self._on_sink_rejected(reason_code)
        except StructuralValidationError:
            self._metrics.incr("frames_rejected_total")
            self._on_loggable_error("structural_validation", self._camera_id)
        except UnsupportedSourceFormatError:
            self._metrics.incr("unsupported_source_format_total")
            self._metrics.incr("frames_rejected_total")
            self._on_loggable_error("unsupported_format", self._camera_id)
        except PayloadSizeMismatchError:
            self._metrics.incr("frames_rejected_total")
            self._on_loggable_error("payload_size_mismatch", self._camera_id)
        except FrameDecodeError:
            self._metrics.incr("decode_failed_total")
            self._metrics.incr("frames_rejected_total")
            self._on_loggable_error("decode_failed", self._camera_id)
        except FrameNormalizationError:
            self._metrics.incr("normalization_failed_total")
            self._metrics.incr("frames_rejected_total")
            self._on_loggable_error("normalization_failed", self._camera_id)
