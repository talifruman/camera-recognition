from __future__ import annotations

import queue
import threading
from collections import deque
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Errors — spec §11
# ---------------------------------------------------------------------------


class GatewayLifecycleError(Exception):
    """Raised for illegal lifecycle transitions (e.g., configure after start)."""


class StructuralValidationError(Exception):
    """Raised when a required field is missing, empty, or has a non-positive dimension."""


class UnsupportedSourceFormatError(Exception):
    """Raised when source_format is not in the supported list."""


class PayloadSizeMismatchError(Exception):
    """Raised when raw payload size is inconsistent with declared metadata."""


class FrameDecodeError(Exception):
    """Raised when an encoded payload cannot be decoded."""


class FrameNormalizationError(Exception):
    """Raised when normalization fails or decoded dimensions differ from declared."""


class FrameStoreError(Exception):
    """Raised for internal FrameStore failures during enqueue."""


class CameraSlotExhaustedError(Exception):
    """Raised when max_camera_slots is reached and a new camera_id arrives."""


# ---------------------------------------------------------------------------
# Data structures — spec §2.2, §3.1, §9.1, §10
# ---------------------------------------------------------------------------


@dataclass
class IngressFrameMessage:
    """Raw transport message — spec §2.2."""

    frame_id: str
    camera_id: str
    timestamp_ms: int
    width: int
    height: int
    source_format: str
    payload_bytes: bytes
    source_layout: str | None = None
    source_num_color_channels: int | None = None
    source_bits_per_channel: int | None = None


@dataclass
class NormalizedFrameBuffer:
    """Intermediate canonical buffer — spec §10."""

    width: int
    height: int
    color_format: str        # always "RGB"
    layout: str              # always "HWC"
    dtype: str               # always "uint8"
    value_range: str         # always "[0,255]"
    num_color_channels: int  # always 3
    bits_per_channel: int    # always 8
    image_bytes: bytes       # raw RGB pixels


@dataclass(frozen=True)
class FramePacket:
    """Canonical immutable output frame — spec §3.1."""

    frame_id: str
    camera_id: str
    timestamp_ms: int
    width: int
    height: int
    pixel_format: str        # always "RGB"
    layout: str              # always "HWC"
    num_color_channels: int  # always 3
    bits_per_channel: int    # always 8
    image_bytes: bytes       # raw RGB pixels


@dataclass
class FrameIngestionGatewayConfig:
    """Module configuration — spec §9.1."""

    bind_address: str
    service_name: str
    max_concurrent_streams: int
    max_camera_slots: int
    camera_order: list[str]
    supported_source_formats: list[str]


@dataclass
class GatewayHealth:
    """Health status report — spec §10."""

    transport_state: str  # "RUNNING" | "STOPPED" | "ERROR"
    store_state: str      # "READY" | "STOPPED"


@dataclass
class StoreHealth:
    """FrameStore health report — spec §10."""

    state: str                             # "READY" | "STOPPED"
    active_cameras: int
    total_queued_frames: int
    queue_depth_per_camera: dict[str, int]


# ---------------------------------------------------------------------------
# Format constants
# ---------------------------------------------------------------------------

_RAW_FORMATS: frozenset[str] = frozenset({"RGB", "BGR", "GRAY8", "YUV420", "NV12", "YUY2"})
_ENCODED_FORMATS: frozenset[str] = frozenset({"JPEG", "MJPEG"})


# ---------------------------------------------------------------------------
# Transport interface and implementations — spec §6, §8.3
# ---------------------------------------------------------------------------


@runtime_checkable
class FrameIngressTransport(Protocol):
    """Transport reception abstraction — spec §6.1."""

    def bind_and_start(self, config: FrameIngestionGatewayConfig) -> None: ...
    def stop(self) -> None: ...
    def receive_message(self) -> IngressFrameMessage: ...


class GrpcFrameIngressTransport:
    """Default gRPC transport — NOT YET IMPLEMENTED in stub version."""

    def bind_and_start(self, config: FrameIngestionGatewayConfig) -> None:
        raise NotImplementedError(
            "GrpcFrameIngressTransport is not implemented in the stub version"
        )

    def stop(self) -> None:
        pass

    def receive_message(self) -> IngressFrameMessage:
        raise NotImplementedError(
            "GrpcFrameIngressTransport is not implemented in the stub version"
        )


class StubFrameIngressTransport:
    """Test-friendly transport — feeds messages via inject_message().

    inject_message() is not part of the FrameIngressTransport interface;
    it is a stub-specific method used to push messages in tests.
    """

    _STOP_SENTINEL: object = object()

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()

    def bind_and_start(self, config: FrameIngestionGatewayConfig) -> None:
        pass  # no-op in stub

    def stop(self) -> None:
        # Unblock any waiting receive_message() call
        self._queue.put(self._STOP_SENTINEL)

    def receive_message(self) -> IngressFrameMessage:
        msg = self._queue.get()
        if msg is self._STOP_SENTINEL:
            raise RuntimeError("StubFrameIngressTransport: transport stopped")
        return msg  # type: ignore[return-value]

    def inject_message(self, message: IngressFrameMessage) -> None:
        """Push a message into the transport queue (stub/test use only)."""
        self._queue.put(message)


# ---------------------------------------------------------------------------
# FrameIngestionInputValidator — spec §8.2
# ---------------------------------------------------------------------------


class FrameIngestionInputValidator:
    """Structural field presence validation — spec §8.2."""

    def validate(self, message: IngressFrameMessage) -> None:
        """Raise StructuralValidationError if any required field is invalid."""
        if not message.frame_id:
            raise StructuralValidationError("frame_id is missing or empty")
        if not message.camera_id:
            raise StructuralValidationError("camera_id is missing or empty")
        if message.timestamp_ms is None:
            raise StructuralValidationError("timestamp_ms is missing")
        if message.width is None or message.width <= 0:
            raise StructuralValidationError(
                f"width must be a positive integer, got {message.width!r}"
            )
        if message.height is None or message.height <= 0:
            raise StructuralValidationError(
                f"height must be a positive integer, got {message.height!r}"
            )
        if not message.source_format:
            raise StructuralValidationError("source_format is missing or empty")
        if not message.payload_bytes:
            raise StructuralValidationError("payload_bytes is missing or empty")


# ---------------------------------------------------------------------------
# FrameValidator — spec §8.4
# ---------------------------------------------------------------------------


def _expected_payload_bytes(fmt: str, width: int, height: int) -> int:
    """Return expected payload byte count for a raw format — spec §2.4B."""
    if fmt in ("RGB", "BGR"):
        return width * height * 3
    if fmt == "GRAY8":
        return width * height
    if fmt in ("YUV420", "NV12"):
        return width * height * 3 // 2
    if fmt == "YUY2":
        return width * height * 2
    raise ValueError(f"Unknown raw format: {fmt!r}")


class FrameValidator:
    """Source format and payload size validation — spec §8.4."""

    def __init__(self, supported_source_formats: list[str]) -> None:
        self._supported: frozenset[str] = frozenset(supported_source_formats)

    def validate(self, message: IngressFrameMessage) -> None:
        """Raise UnsupportedSourceFormatError or PayloadSizeMismatchError on failure."""
        fmt = message.source_format

        if fmt not in self._supported:
            raise UnsupportedSourceFormatError(
                f"source_format {fmt!r} is not in supported list"
            )

        if fmt in _RAW_FORMATS:
            w, h = message.width, message.height

            # YUV420 and NV12 require even dimensions — spec §2.4B
            if fmt in ("YUV420", "NV12"):
                if w % 2 != 0 or h % 2 != 0:
                    raise PayloadSizeMismatchError(
                        f"{fmt} requires even width and height, got {w}x{h}"
                    )

            expected = _expected_payload_bytes(fmt, w, h)
            actual = len(message.payload_bytes)
            if actual != expected:
                raise PayloadSizeMismatchError(
                    f"{fmt} payload size mismatch: expected {expected} bytes, got {actual}"
                )

        # For encoded formats (JPEG, MJPEG): no byte-size check before decoding


# ---------------------------------------------------------------------------
# FramePayloadNormalizer — spec §8.5
# ---------------------------------------------------------------------------


class FramePayloadNormalizer:
    """Normalizes payload to canonical RGB/HWC/uint8/[0,255] — real implementation.

    Supported source formats and conversion paths (spec §8.5):
      RGB    — pass through; CHW layout is transposed to HWC
      BGR    — cv2.COLOR_BGR2RGB; CHW layout transposed first
      GRAY8  — R=G=B=pixel_value via np.stack
      YUV420 — reshape(h*3//2, w) + cv2.COLOR_YUV2RGB_I420 (full-range BT.601)
      NV12   — reshape(h*3//2, w) + cv2.COLOR_YUV2RGB_NV12 (full-range BT.601)
      YUY2   — reshape(h, w, 2) + cv2.COLOR_YUV2RGB_YUY2  (full-range BT.601)
      JPEG   — cv2.imdecode(IMREAD_UNCHANGED) + channel normalisation + BGR->RGB
      MJPEG  — identical to JPEG (one JPEG-encoded frame per message)
    """

    def normalize(self, message: IngressFrameMessage) -> NormalizedFrameBuffer:
        """Return NormalizedFrameBuffer with canonical RGB output — spec §8.5."""
        fmt = message.source_format
        w, h = message.width, message.height
        layout = message.source_layout  # "CHW" | "HWC" | None

        try:
            rgb = self._to_rgb(fmt, message.payload_bytes, w, h, layout)
        except (FrameDecodeError, FrameNormalizationError):
            raise
        except Exception as exc:
            raise FrameNormalizationError(
                f"Normalization failed for format {fmt!r}: {exc}"
            ) from exc

        # Guarantee contiguous uint8 memory — spec §8.5 output guarantees
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

        return NormalizedFrameBuffer(
            width=w,
            height=h,
            color_format="RGB",
            layout="HWC",
            dtype="uint8",
            value_range="[0,255]",
            num_color_channels=3,
            bits_per_channel=8,
            image_bytes=rgb.tobytes(),
        )

    # ------------------------------------------------------------------
    # Internal per-format conversion helpers
    # ------------------------------------------------------------------

    def _to_rgb(
        self,
        fmt: str,
        payload: bytes,
        w: int,
        h: int,
        layout: str | None,
    ) -> np.ndarray:
        if fmt in ("JPEG", "MJPEG"):
            return self._decode_jpeg(payload, w, h)
        if fmt == "RGB":
            return self._normalize_rgb(payload, w, h, layout)
        if fmt == "BGR":
            return self._normalize_bgr(payload, w, h, layout)
        if fmt == "GRAY8":
            return self._normalize_gray8(payload, w, h)
        if fmt == "YUV420":
            return self._normalize_yuv(payload, w, h, cv2.COLOR_YUV2RGB_I420)
        if fmt == "NV12":
            return self._normalize_yuv(payload, w, h, cv2.COLOR_YUV2RGB_NV12)
        if fmt == "YUY2":
            return self._normalize_yuy2(payload, w, h)
        raise FrameNormalizationError(f"Unhandled source format: {fmt!r}")

    def _decode_jpeg(self, payload: bytes, w: int, h: int) -> np.ndarray:
        """Decode JPEG/MJPEG payload to RGB HWC numpy array — spec §8.5."""
        arr = np.frombuffer(payload, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise FrameDecodeError(
                "cv2.imdecode returned None — payload could not be decoded as JPEG"
            )

        if decoded.ndim == 2:
            # Grayscale 2D array — R=G=B rule (spec §8.5 JPEG rules)
            rgb = np.stack([decoded, decoded, decoded], axis=-1)
        elif decoded.shape[2] == 1:
            ch = decoded[:, :, 0]
            rgb = np.stack([ch, ch, ch], axis=-1)
        elif decoded.shape[2] == 3:
            # OpenCV decodes colour JPEG as BGR — convert to RGB
            rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        elif decoded.shape[2] == 4:
            # BGRA — drop alpha, reorder to RGB (spec §8.5 JPEG rules)
            rgb = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGB)
        else:
            raise FrameNormalizationError(
                f"Decoded JPEG has unexpected channel count: {decoded.shape[2]}"
            )

        # Validate decoded dimensions against declared metadata (spec §8.5)
        dec_h, dec_w = rgb.shape[:2]
        if dec_h != h or dec_w != w:
            raise FrameNormalizationError(
                f"Decoded JPEG dimensions {dec_w}x{dec_h} differ from "
                f"declared {w}x{h}"
            )

        return rgb

    def _normalize_rgb(
        self, payload: bytes, w: int, h: int, layout: str | None
    ) -> np.ndarray:
        if layout == "CHW":
            arr = np.frombuffer(payload, dtype=np.uint8).reshape(3, h, w)
            return arr.transpose(1, 2, 0)  # CHW -> HWC
        return np.frombuffer(payload, dtype=np.uint8).reshape(h, w, 3)

    def _normalize_bgr(
        self, payload: bytes, w: int, h: int, layout: str | None
    ) -> np.ndarray:
        if layout == "CHW":
            arr = np.frombuffer(payload, dtype=np.uint8).reshape(3, h, w)
            bgr = arr.transpose(1, 2, 0)  # CHW -> HWC
        else:
            bgr = np.frombuffer(payload, dtype=np.uint8).reshape(h, w, 3)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _normalize_gray8(self, payload: bytes, w: int, h: int) -> np.ndarray:
        gray = np.frombuffer(payload, dtype=np.uint8).reshape(h, w)
        return np.stack([gray, gray, gray], axis=-1)  # R=G=B rule (spec §8.5)

    def _normalize_yuv(
        self, payload: bytes, w: int, h: int, cv2_code: int
    ) -> np.ndarray:
        # YUV420 / NV12: planar layout requires shape (h*3//2, w) — spec §8.5
        yuv = np.frombuffer(payload, dtype=np.uint8).reshape(h * 3 // 2, w)
        return cv2.cvtColor(yuv, cv2_code)

    def _normalize_yuy2(self, payload: bytes, w: int, h: int) -> np.ndarray:
        # YUY2 packed 4:2:2 — 2 bytes per pixel, shape (h, w, 2) — spec §8.5
        yuyv = np.frombuffer(payload, dtype=np.uint8).reshape(h, w, 2)
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUY2)


# ---------------------------------------------------------------------------
# FramePacketBuilder — spec §8.6
# ---------------------------------------------------------------------------


class FramePacketBuilder:
    """Constructs an immutable FramePacket — spec §8.6."""

    def build(
        self,
        message: IngressFrameMessage,
        buffer: NormalizedFrameBuffer,
    ) -> FramePacket:
        return FramePacket(
            frame_id=message.frame_id,
            camera_id=message.camera_id,
            timestamp_ms=message.timestamp_ms,
            width=buffer.width,
            height=buffer.height,
            pixel_format="RGB",
            layout="HWC",
            num_color_channels=3,
            bits_per_channel=8,
            image_bytes=buffer.image_bytes,
        )


# ---------------------------------------------------------------------------
# FrameStore — spec §8.7
# ---------------------------------------------------------------------------


class FrameStore:
    """Per-camera FIFO queue storage and round-robin retrieval — spec §8.7."""

    def __init__(self, max_camera_slots: int, camera_order: list[str]) -> None:
        self._max_camera_slots = max_camera_slots
        self._camera_order: list[str] = list(camera_order)
        self._queues: dict[str, deque[FramePacket]] = {
            cam_id: deque() for cam_id in camera_order
        }
        self._lock = threading.Lock()
        self._rr_index: int = 0
        self._state: str = "STOPPED"

        # Metrics — spec §12
        self._frames_enqueued_total: int = 0
        self._frames_dequeued_total: int = 0
        self._camera_slot_rejected_total: int = 0
        self._store_error_total: int = 0

    def start(self) -> None:
        with self._lock:
            self._state = "READY"

    def stop(self) -> None:
        with self._lock:
            self._state = "STOPPED"
        # Queued frames are NOT cleared — remain retrievable via get_next_frame()

    def enqueue(self, frame_packet: FramePacket) -> None:
        with self._lock:
            cam_id = frame_packet.camera_id
            if cam_id not in self._queues:
                if len(self._camera_order) >= self._max_camera_slots:
                    self._camera_slot_rejected_total += 1
                    raise CameraSlotExhaustedError(
                        f"max_camera_slots={self._max_camera_slots} reached; "
                        f"camera_id={cam_id!r} rejected"
                    )
                # New camera: append to end of camera_order and create queue
                self._camera_order.append(cam_id)
                self._queues[cam_id] = deque()

            self._queues[cam_id].append(frame_packet)
            self._frames_enqueued_total += 1

    def get_next_frame(self) -> FramePacket | None:
        with self._lock:
            n = len(self._camera_order)
            if n == 0:
                return None
            # Round-robin: scan up to n cameras starting from _rr_index
            for _ in range(n):
                idx = self._rr_index % n
                self._rr_index = (idx + 1) % n
                cam_id = self._camera_order[idx]
                q = self._queues.get(cam_id)
                if q:
                    pkt = q.popleft()
                    self._frames_dequeued_total += 1
                    return pkt
            return None

    def health(self) -> StoreHealth:
        with self._lock:
            return StoreHealth(
                state=self._state,
                active_cameras=len(self._camera_order),
                total_queued_frames=sum(len(q) for q in self._queues.values()),
                queue_depth_per_camera={
                    cam_id: len(q) for cam_id, q in self._queues.items()
                },
            )


# ---------------------------------------------------------------------------
# FrameIngestionGateway — spec §8.1
# ---------------------------------------------------------------------------


class FrameIngestionGateway:
    """Orchestration layer for the frame ingestion pipeline — spec §8.1.

    Public API (spec §4):
        get_next_frame() -> FramePacket | None
        configure(config: FrameIngestionGatewayConfig) -> None
        start() -> None
        stop() -> None
        health() -> GatewayHealth
    """

    def __init__(self, transport: FrameIngressTransport | None = None) -> None:
        self._transport: FrameIngressTransport | None = transport
        self._input_validator = FrameIngestionInputValidator()
        self._normalizer = FramePayloadNormalizer()
        self._builder = FramePacketBuilder()
        self._frame_validator: FrameValidator | None = None
        self._store: FrameStore | None = None
        self._config: FrameIngestionGatewayConfig | None = None
        self._ingestion_thread: threading.Thread | None = None
        self._running: bool = False

        # Metrics — spec §12
        self._frames_in_total: int = 0
        self._bytes_in_total: int = 0
        self._ingress_rejected_total: int = 0
        self._unsupported_source_format_total: int = 0
        self._decode_failed_total: int = 0
        self._normalization_failed_total: int = 0
        self._frames_normalized_total: int = 0

    def configure(self, config: FrameIngestionGatewayConfig) -> None:
        """Wire all internal components with config — spec §13.1.

        Raises GatewayLifecycleError if called after start().
        """
        if self._running:
            raise GatewayLifecycleError("configure() must not be called after start()")

        self._config = config

        if self._transport is None:
            self._transport = GrpcFrameIngressTransport()

        self._frame_validator = FrameValidator(config.supported_source_formats)
        self._store = FrameStore(config.max_camera_slots, list(config.camera_order))

    def start(self) -> None:
        """Start transport and ingestion background thread — spec §13.1.

        Idempotent: second call is a no-op.
        Returns only after transport is bound and ingestion thread is running.
        """
        if self._running:
            return  # idempotent

        if self._config is None:
            raise GatewayLifecycleError("configure() must be called before start()")

        self._store.start()
        self._transport.bind_and_start(self._config)
        self._running = True

        self._ingestion_thread = threading.Thread(
            target=self._ingestion_loop,
            daemon=True,
            name="FrameIngestionGateway-ingestion",
        )
        self._ingestion_thread.start()

    def stop(self) -> None:
        """Stop transport and ingestion thread — spec §13.3.

        Queued frames in FrameStore are preserved.
        """
        self._running = False

        if self._transport is not None:
            self._transport.stop()

        if self._ingestion_thread is not None:
            self._ingestion_thread.join(timeout=5.0)
            self._ingestion_thread = None

        if self._store is not None:
            self._store.stop()

    def health(self) -> GatewayHealth:
        """Return current gateway health — spec §4."""
        transport_state = "RUNNING" if self._running else "STOPPED"
        store_state = "STOPPED"
        if self._store is not None:
            store_state = self._store.health().state
        return GatewayHealth(
            transport_state=transport_state,
            store_state=store_state,
        )

    def get_next_frame(self) -> FramePacket | None:
        """Return next available FramePacket or None — spec §4."""
        if self._store is None:
            return None
        return self._store.get_next_frame()

    # ------------------------------------------------------------------
    # Internal ingestion loop — spec §8.8
    # ------------------------------------------------------------------

    def _ingestion_loop(self) -> None:
        """Background thread: continuously receive and process ingress messages."""
        while self._running:
            # Receive next message (blocking call)
            try:
                message = self._transport.receive_message()
            except Exception:
                # Transport stopped or irrecoverable error
                break

            if not self._running:
                break

            # Track ingress metrics — spec §12
            self._frames_in_total += 1
            self._bytes_in_total += len(message.payload_bytes) if message.payload_bytes else 0

            # Step 1: Structural validation — spec §8.8 steps 2–3
            try:
                self._input_validator.validate(message)
            except StructuralValidationError:
                self._ingress_rejected_total += 1
                continue

            # Step 2: Source format + payload size validation — spec §8.8 steps 4–5
            try:
                self._frame_validator.validate(message)
            except UnsupportedSourceFormatError:
                self._unsupported_source_format_total += 1
                continue
            except PayloadSizeMismatchError:
                self._ingress_rejected_total += 1
                continue

            # Step 3: Normalize payload — spec §8.8 steps 6–7
            try:
                buffer = self._normalizer.normalize(message)
            except FrameDecodeError:
                self._decode_failed_total += 1
                continue
            except FrameNormalizationError:
                self._normalization_failed_total += 1
                continue

            self._frames_normalized_total += 1

            # Step 4: Build FramePacket — spec §8.8 step 8
            packet = self._builder.build(message, buffer)

            # Step 5: Enqueue — spec §8.8 step 9
            try:
                self._store.enqueue(packet)
            except CameraSlotExhaustedError:
                pass  # FrameStore already incremented camera_slot_rejected_total
            except FrameStoreError:
                pass  # FrameStore already incremented store_error_total
