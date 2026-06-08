"""FramePacket builder for canonical sink publication."""

from __future__ import annotations

from image_processing.shared.contracts import FramePacket

from .contracts import IngressFrameMessage, NormalizedFrameBuffer


class FramePacketBuilder:
    """Build immutable shared FramePacket objects from normalized buffers."""

    def build(
        self,
        message: IngressFrameMessage,
        normalized: NormalizedFrameBuffer,
    ) -> FramePacket:
        """Build a canonical shared FramePacket without source metadata."""
        return FramePacket(
            frame_id=message.frame_id,
            camera_id=message.camera_id,
            timestamp_ms=message.timestamp_ms,
            width=normalized.width,
            height=normalized.height,
            pixel_format="RGB",
            layout="HWC",
            dtype="uint8",
            value_range="[0,255]",
            num_color_channels=3,
            bits_per_channel=8,
            packing="tightly_packed",
            image_bytes=normalized.image_bytes,
        )
