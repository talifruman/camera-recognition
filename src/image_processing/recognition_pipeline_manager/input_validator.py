"""RecognitionPipelineInputValidator — validates FramePacket before pipeline execution.

Spec: RecognitionPipelineManager §8.2
"""

from __future__ import annotations

from image_processing.frame_transformation_layer.contracts import FramePacket


class RecognitionPipelineInputValidator:
    """Validates a ``FramePacket`` before the recognition pipeline executes.

    Raises ``ValueError`` on any violation.  Does not ingest frames, invoke
    pipeline stages, or make routing decisions.
    """

    def validate(self, frame_packet: FramePacket) -> None:
        """Validate ``frame_packet`` against all required-field rules.

        Raises:
            ValueError: if ``frame_packet`` is None, or any required field
                is missing, empty, or of the wrong type.
        """
        if frame_packet is None:
            raise ValueError("frame_packet must not be None")

        frame_id = frame_packet.frame_id
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError("frame_packet.frame_id must be a non-empty string")

        camera_id = frame_packet.camera_id
        if not isinstance(camera_id, str) or not camera_id:
            raise ValueError("frame_packet.camera_id must be a non-empty string")

        timestamp_ms = frame_packet.timestamp_ms
        if timestamp_ms is None:
            raise ValueError("frame_packet.timestamp_ms must be present")
        if not isinstance(timestamp_ms, int):
            raise ValueError("frame_packet.timestamp_ms must be an integer")
