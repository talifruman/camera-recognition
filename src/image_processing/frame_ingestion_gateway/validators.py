"""Validation logic for ingress messages."""

from __future__ import annotations

import time

from .config import FrameIngestionGatewayConfig
from .contracts import IngressFrameMessage
from .errors import (
    PayloadSizeMismatchError,
    StructuralValidationError,
    UnsupportedSourceFormatError,
)

RAW_FORMATS = {"RGB", "BGR", "GRAY8", "YUV420", "NV12", "YUY2"}
ENCODED_FORMATS = {"JPEG", "MJPEG"}


class FrameIngestionInputValidator:
    """Structural validator for ingress frame envelopes."""

    def validate(self, message: IngressFrameMessage) -> None:
        """Validate required fields and primitive constraints."""
        if not message.frame_id:
            raise StructuralValidationError("frame_id is required")
        if not message.camera_id:
            raise StructuralValidationError("camera_id is required")
        if message.timestamp_ms <= 0:
            raise StructuralValidationError("timestamp_ms must be > 0")
        if message.width <= 0 or message.height <= 0:
            raise StructuralValidationError("width and height must be > 0")
        if not message.source_format:
            raise StructuralValidationError("source_format is required")
        if not message.payload_bytes:
            raise StructuralValidationError("payload_bytes is required")


class FrameValidator:
    """Format and guardrail validator for ingress frame payloads."""

    def __init__(self, config: FrameIngestionGatewayConfig) -> None:
        self._config = config

    def validate(self, message: IngressFrameMessage) -> None:
        """Validate source format, metadata, and payload constraints."""
        if message.source_format != message.source_format.upper():
            raise UnsupportedSourceFormatError("source_format must be uppercase")
        if message.source_format not in self._config.supported_source_formats:
            raise UnsupportedSourceFormatError("source_format not supported")
        if message.width > self._config.max_frame_width:
            raise StructuralValidationError("width exceeds max_frame_width")
        if message.height > self._config.max_frame_height:
            raise StructuralValidationError("height exceeds max_frame_height")
        if len(message.payload_bytes) > self._config.max_payload_bytes:
            raise StructuralValidationError("payload exceeds max_payload_bytes")
        self._validate_source_metadata(message)
        self._validate_raw_payload_size(message)

    def _validate_source_metadata(self, message: IngressFrameMessage) -> None:
        is_raw = message.source_format in RAW_FORMATS
        if not is_raw and message.source_layout is not None:
            raise StructuralValidationError("source_layout must be absent for encoded")
        if is_raw and message.source_layout not in {None, "HWC", "CHW"}:
            raise StructuralValidationError("source_layout must be HWC or CHW")
        if message.source_bits_per_channel not in {None, 8}:
            raise StructuralValidationError("source_bits_per_channel must be 8")

    def _validate_raw_payload_size(self, message: IngressFrameMessage) -> None:
        expected = _expected_raw_size(
            source_format=message.source_format,
            width=message.width,
            height=message.height,
        )
        if expected is None:
            return
        if len(message.payload_bytes) != expected:
            raise PayloadSizeMismatchError("raw payload size mismatch")


def _expected_raw_size(source_format: str, width: int, height: int) -> int | None:
    """Compute expected payload size for raw source formats."""
    if source_format in ENCODED_FORMATS:
        return None
    if source_format in {"YUV420", "NV12"}:
        if width % 2 != 0 or height % 2 != 0:
            raise PayloadSizeMismatchError("YUV420/NV12 require even dimensions")
        return width * height * 3 // 2
    if source_format in {"RGB", "BGR"}:
        return width * height * 3
    if source_format == "GRAY8":
        return width * height
    if source_format == "YUY2":
        return width * height * 2
    raise UnsupportedSourceFormatError("source_format not recognized")


def detect_timestamp_anomaly(
    message: IngressFrameMessage,
    previous_timestamp_ms: int | None,
    config: FrameIngestionGatewayConfig,
) -> bool:
    """Return true when timestamp violates skew/lag/order policy."""
    now_ms = int(time.time() * 1000)
    too_future = message.timestamp_ms > now_ms + config.allowed_timestamp_skew_ms
    too_old = message.timestamp_ms < now_ms - config.allowed_timestamp_lag_ms
    out_of_order = (
        previous_timestamp_ms is not None
        and message.timestamp_ms < previous_timestamp_ms
    )
    return too_future or too_old or out_of_order
