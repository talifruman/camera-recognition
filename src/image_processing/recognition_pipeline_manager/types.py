"""Public and internal data structures for RecognitionPipelineManager.

Public types (exported via __init__.py):
    RecognizedFaceResult  — one recognized face in full-frame coordinates
    PersonResult          — one detected person with their recognized faces
    RecognitionPipelineOutput — final output of process_frame()

Internal types (NOT exported):
    PipelineResult        — accumulated per-frame output inside PipelineOrchestrator
"""

from __future__ import annotations

from typing import TypedDict

from image_processing.shared.contracts import BoundingBox


# ---------------------------------------------------------------------------
# Public output types
# ---------------------------------------------------------------------------


class RecognizedFaceResult(TypedDict):
    """A single recognized face within a person's region.

    All coordinates are in full-frame space.
    Unrecognized faces are never present — they are omitted entirely.
    """

    face_bbox: BoundingBox  # FULL_FRAME coordinates
    person_id: str
    person_name: str
    found: bool


class PersonResult(TypedDict):
    """A single detected person with all their recognized faces.

    ``person_bbox`` is in full-frame coordinates.
    ``recognized_faces`` may be empty when no faces were detected or
    all detected faces failed recognition.  Unrecognized faces are omitted
    entirely — they are never labeled UNKNOWN in this list.
    """

    person_bbox: BoundingBox           # FULL_FRAME coordinates
    recognized_faces: list[RecognizedFaceResult]


class RecognitionPipelineOutput(TypedDict):
    """Final output of ``RecognitionPipelineManager.process_frame``.

    ``frame_id``, ``camera_id``, and ``timestamp_ms`` are copied unchanged
    from the input ``FramePacket`` for traceability.
    ``persons`` may be empty.
    No ROI-local coordinates, scores, embeddings, or landmarks are exposed.
    Unrecognized faces are omitted from ``PersonResult.recognized_faces``—
    they are never labeled UNKNOWN.
    """

    frame_id: str
    camera_id: str
    timestamp_ms: int
    persons: list[PersonResult]


# ---------------------------------------------------------------------------
# Internal accumulator — NOT part of the public API
# ---------------------------------------------------------------------------


class PipelineResult(TypedDict):
    """Internal accumulated result produced by PipelineOrchestrator.execute().

    Does not carry frame_id / camera_id / timestamp_ms — those are copied
    from the original FramePacket by RecognitionPipelineOutputBuilder.
    """

    persons: list[PersonResult]
