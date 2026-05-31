"""Payload decoding and canonical RGB normalization."""

from __future__ import annotations

import numpy as np

from .contracts import IngressFrameMessage, NormalizedFrameBuffer
from .errors import DependencyUnavailableError, FrameDecodeError, FrameNormalizationError

try:
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    _CV2_IMPORT_ERROR: Exception | None = exc
else:
    _CV2_IMPORT_ERROR = None


class FramePayloadNormalizer:
    """Normalize source payloads into canonical RGB/HWC/uint8 bytes."""

    def __init__(self) -> None:
        if cv2 is None:
            raise DependencyUnavailableError(
                f"OpenCV dependency unavailable: {_CV2_IMPORT_ERROR}"
            )

    def normalize(self, message: IngressFrameMessage) -> NormalizedFrameBuffer:
        """Decode/convert message payload into canonical frame buffer."""
        if message.source_format == "RGB" and message.source_layout != "CHW":
            return NormalizedFrameBuffer(
                width=message.width,
                height=message.height,
                color_format="RGB",
                layout="HWC",
                dtype="uint8",
                value_range="[0,255]",
                num_color_channels=3,
                bits_per_channel=8,
                image_bytes=message.payload_bytes,
            )
        rgb = self._to_rgb_array(message)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise FrameNormalizationError("normalized output must be HWC with 3 channels")
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        return NormalizedFrameBuffer(
            width=rgb.shape[1],
            height=rgb.shape[0],
            color_format="RGB",
            layout="HWC",
            dtype="uint8",
            value_range="[0,255]",
            num_color_channels=3,
            bits_per_channel=8,
            image_bytes=rgb.tobytes(),
        )

    def _to_rgb_array(self, message: IngressFrameMessage) -> np.ndarray:
        if message.source_format == "RGB":
            return self._rgb_from_raw(message)
        if message.source_format == "BGR":
            raw = self._reshape_hwc(message.payload_bytes, message.width, message.height, 3)
            return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        if message.source_format == "GRAY8":
            raw = np.frombuffer(message.payload_bytes, dtype=np.uint8).reshape(
                message.height,
                message.width,
            )
            return cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        if message.source_format == "YUV420":
            raw = np.frombuffer(message.payload_bytes, dtype=np.uint8).reshape(
                message.height * 3 // 2,
                message.width,
            )
            return cv2.cvtColor(raw, cv2.COLOR_YUV2RGB_I420)
        if message.source_format == "NV12":
            raw = np.frombuffer(message.payload_bytes, dtype=np.uint8).reshape(
                message.height * 3 // 2,
                message.width,
            )
            return cv2.cvtColor(raw, cv2.COLOR_YUV2RGB_NV12)
        if message.source_format == "YUY2":
            raw = np.frombuffer(message.payload_bytes, dtype=np.uint8).reshape(
                message.height,
                message.width,
                2,
            )
            return cv2.cvtColor(raw, cv2.COLOR_YUV2RGB_YUY2)
        return self._decode_encoded(message)

    def _rgb_from_raw(self, message: IngressFrameMessage) -> np.ndarray:
        if message.source_layout == "CHW":
            raw = np.frombuffer(message.payload_bytes, dtype=np.uint8).reshape(
                3,
                message.height,
                message.width,
            )
            return raw.transpose(1, 2, 0)
        return self._reshape_hwc(message.payload_bytes, message.width, message.height, 3)

    def _reshape_hwc(
        self,
        payload: bytes,
        width: int,
        height: int,
        channels: int,
    ) -> np.ndarray:
        return np.frombuffer(payload, dtype=np.uint8).reshape(height, width, channels)

    def _decode_encoded(self, message: IngressFrameMessage) -> np.ndarray:
        encoded = np.frombuffer(message.payload_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise FrameDecodeError("failed to decode encoded payload")
        if decoded.ndim == 2:
            decoded = cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGB)
        elif decoded.shape[2] == 4:
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGB)
        else:
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        if decoded.shape[1] != message.width or decoded.shape[0] != message.height:
            raise FrameNormalizationError("declared dimensions mismatch decoded image")
        return decoded
