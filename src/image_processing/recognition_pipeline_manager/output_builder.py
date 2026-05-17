"""RecognitionPipelineOutputBuilder — assembles RecognitionPipelineOutput.

Spec: RecognitionPipelineManager §8.7
"""

from __future__ import annotations

from image_processing.recognition_pipeline_manager.types import (
    PipelineResult,
    RecognitionPipelineOutput,
)


class RecognitionPipelineOutputBuilder:
    """Assembles the final ``RecognitionPipelineOutput`` from pipeline results.

    Copies traceability metadata unchanged from the original FramePacket
    fields.  Does not invoke pipeline stages, apply projection logic, or
    make routing decisions.
    """

    def build(
        self,
        frame_id: str,
        camera_id: str,
        timestamp_ms: int,
        pipeline_result: PipelineResult,
    ) -> RecognitionPipelineOutput:
        """Build and return the final output struct.

        Args:
            frame_id:        copied unchanged from FramePacket.frame_id
            camera_id:       copied unchanged from FramePacket.camera_id
            timestamp_ms:    copied unchanged from FramePacket.timestamp_ms
            pipeline_result: accumulated persons from PipelineOrchestrator

        Returns:
            A fully populated ``RecognitionPipelineOutput``.
        """
        return RecognitionPipelineOutput(
            frame_id=frame_id,
            camera_id=camera_id,
            timestamp_ms=timestamp_ms,
            persons=list(pipeline_result["persons"]),
        )
