"""Fake/stub implementations of all RPM pipeline dependencies.

Each fake:
    - records every call with full argument detail
    - is configurable with a single return value or per-call raise
    - returns realistic data structures (not placeholder Nones)
    - propagates metadata (frame_id, timestamp_ms) faithfully (C3)

Fakes are test-only.  They are NOT exported from the RPM package.

Usage:
    ftl = FakeFTL()
    ftl.configure_ingest_frame_packet(frame_packet)   # record expected packet
    motion = FakeMotionDetection(result=MotionResult(detected=True, bboxes=[...]))
    rpm = RecognitionPipelineManager(ftl=ftl, motion=motion, ...)
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from image_processing.face_detection.module import (
    FaceDetectionInput,
    FaceDetectionOutput,
)
from image_processing.face_recognition.module import (
    FaceRecognitionInput,
    FaceRecognitionOutput,
)
from image_processing.frame_transformation_layer.contracts import (
    FramePacket,
    FrameTemporalSelector,
    PreviousFrameNotAvailableError,
    ProcessedFrame,
    SpatialTransform,
)
from image_processing.motion_detection.module import MotionDetectionInput, MotionResult
from image_processing.object_detection.module import (
    ObjectDetectionInput,
    PersonDetectionResult,
)
from image_processing.person_directory.module import PersonDirectoryOutput
from image_processing.shared.contracts import (
    BoundingBox,
    GeometrySpec,
    Image,
    OutputImageType,
    PipelineStageInputContract,
    ResizePolicy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IDENTITY_SPATIAL = SpatialTransform(
    scale_x=1.0,
    scale_y=1.0,
    pad_left=0,
    pad_top=0,
    output_width=0,  # filled in per-call
    output_height=0,
)

_UNKNOWN_PD_OUTPUT: PersonDirectoryOutput = PersonDirectoryOutput(
    person_id="UNKNOWN",
    person_name="UNKNOWN",
    found=False,
)

_STUB_INPUT_CONTRACT_GRAY: PipelineStageInputContract = PipelineStageInputContract(
    output_image_type=OutputImageType.GRAYSCALE_UINT8_HWC,
    geometry_spec=GeometrySpec(width=0, height=0, resize_policy=ResizePolicy.NONE),
)

_STUB_INPUT_CONTRACT_RGB: PipelineStageInputContract = PipelineStageInputContract(
    output_image_type=OutputImageType.RGB_UINT8_HWC,
    geometry_spec=GeometrySpec(width=0, height=0, resize_policy=ResizePolicy.NONE),
)


def _make_image(width: int, height: int, channels: int, color_format: str) -> Image:
    """Build a minimal valid Image TypedDict with a zero-filled ndarray."""
    data = np.zeros((height, width, channels), dtype=np.uint8)
    return Image(
        data=data,
        width=width,
        height=height,
        color_format=color_format,
        layout="HWC",
        dtype="uint8",
        value_range="[0,255]",
    )


def _make_processed_frame(
    *,
    frame_id: str,
    timestamp_ms: int,
    region_bbox: BoundingBox,
    output_type: OutputImageType,
    spatial_transform: SpatialTransform | None = None,
) -> ProcessedFrame:
    """Build a realistic ProcessedFrame from a requested region bbox.

    The image dimensions match the requested region.
    The source_bbox_full_frame mirrors the requested region.
    Metadata (frame_id, timestamp_ms) is propagated from the last ingest (C3).
    """
    w = max(1, region_bbox["width"])
    h = max(1, region_bbox["height"])

    if output_type == OutputImageType.GRAYSCALE_UINT8_HWC:
        channels = 1
        color_format = "GRAY"
    else:
        channels = 3
        color_format = "RGB"

    image = _make_image(w, h, channels, color_format)

    if spatial_transform is None:
        st = SpatialTransform(
            scale_x=1.0,
            scale_y=1.0,
            pad_left=0,
            pad_top=0,
            output_width=w,
            output_height=h,
        )
    else:
        st = spatial_transform

    source_bbox = BoundingBox(
        x=region_bbox["x"],
        y=region_bbox["y"],
        width=w,
        height=h,
    )

    return ProcessedFrame(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        image=image,
        source_bbox_full_frame=source_bbox,
        spatial_transform=st,
    )


# ---------------------------------------------------------------------------
# FakeFTL
# ---------------------------------------------------------------------------


class FakeFTL:
    """Fake FrameTransformationLayerInterface.

    Attributes:
        ingest_calls:       list of FramePackets passed to ingest_frame()
        get_frame_calls:    list of dicts with call arguments for get_frame()
        raise_on_ingest:    if set, ingest_frame() raises this exception
        raise_on_previous:  if True, get_frame(PREVIOUS, ...) raises
                            PreviousFrameNotAvailableError
        raise_on_get_frame_call_indices: set of 0-based get_frame call
                            indices that should raise an exception (for
                            testing OD/FD/FR region-level failures)
        spatial_transform_overrides: list of SpatialTransform values to
                            return for get_frame calls at matching indices
                            (index-aligned; None means use identity default)
    """

    def __init__(
        self,
        *,
        raise_on_ingest: Exception | None = None,
        raise_on_previous: bool = False,
        raise_on_get_frame_call_indices: set[int] | None = None,
        spatial_transform_overrides: list[SpatialTransform | None] | None = None,
    ) -> None:
        self.raise_on_ingest = raise_on_ingest
        self.raise_on_previous = raise_on_previous
        self.raise_on_get_frame_call_indices: set[int] = (
            raise_on_get_frame_call_indices or set()
        )
        self.spatial_transform_overrides: list[SpatialTransform | None] = (
            spatial_transform_overrides or []
        )
        self.ingest_calls: list[FramePacket] = []
        self.get_frame_calls: list[dict[str, Any]] = []
        # Tracks the most-recently ingested packet for metadata propagation (C3)
        self._last_frame_id: str = "fake-frame-0"
        self._last_timestamp_ms: int = 0

    def ingest_frame(self, frame_packet: FramePacket) -> None:
        self.ingest_calls.append(frame_packet)
        if self.raise_on_ingest is not None:
            raise self.raise_on_ingest
        # Propagate metadata for downstream ProcessedFrame responses (C3)
        self._last_frame_id = frame_packet.frame_id
        self._last_timestamp_ms = frame_packet.timestamp_ms

    def get_frame(
        self,
        camera_id: str,
        temporal_selector: FrameTemporalSelector,
        region_bbox: BoundingBox,
        output_type: OutputImageType,
        geometry_spec: GeometrySpec,
    ) -> ProcessedFrame:
        call_index = len(self.get_frame_calls)
        self.get_frame_calls.append(
            {
                "camera_id": camera_id,
                "temporal_selector": temporal_selector,
                "region_bbox": dict(region_bbox),
                "output_type": output_type,
                "geometry_spec": dict(geometry_spec),
            }
        )

        # Cold-start simulation
        if (
            temporal_selector is FrameTemporalSelector.PREVIOUS
            and self.raise_on_previous
        ):
            raise PreviousFrameNotAvailableError(
                f"FakeFTL: no previous frame for camera_id={camera_id!r}"
            )

        # Per-call-index failure simulation
        if call_index in self.raise_on_get_frame_call_indices:
            raise RuntimeError(
                f"FakeFTL: simulated get_frame failure at call index {call_index}"
            )

        # Spatial transform override (for LETTERBOX projection tests)
        st_override: SpatialTransform | None = None
        if call_index < len(self.spatial_transform_overrides):
            st_override = self.spatial_transform_overrides[call_index]

        return _make_processed_frame(
            frame_id=self._last_frame_id,
            timestamp_ms=self._last_timestamp_ms,
            region_bbox=region_bbox,
            output_type=output_type,
            spatial_transform=st_override,
        )


# ---------------------------------------------------------------------------
# FakeMotionDetection
# ---------------------------------------------------------------------------


class FakeMotionDetection:
    """Fake MotionDetectionInterface.

    Attributes:
        calls:   list of MotionDetectionInput passed to detect()
        result:  MotionResult returned from detect()
    """

    def __init__(self, result: MotionResult | None = None) -> None:
        self.calls: list[MotionDetectionInput] = []
        self.result: MotionResult = result or MotionResult(
            detected=False, bboxes=[]
        )

    def detect(self, input: MotionDetectionInput) -> MotionResult:
        self.calls.append(input)
        return self.result

    def get_input_contract(self) -> PipelineStageInputContract:
        return _STUB_INPUT_CONTRACT_GRAY


# ---------------------------------------------------------------------------
# FakeObjectDetection
# ---------------------------------------------------------------------------


class FakeObjectDetection:
    """Fake ObjectDetectionInterface.

    Attributes:
        calls:   list of ObjectDetectionInput passed to detect()
        result:  PersonDetectionResult returned from detect()
    """

    def __init__(self, result: PersonDetectionResult | None = None) -> None:
        self.calls: list[ObjectDetectionInput] = []
        self.result: PersonDetectionResult = result or PersonDetectionResult(
            frame_id="", person_detected=False, persons=[]
        )

    def detect(self, input: ObjectDetectionInput) -> PersonDetectionResult:
        self.calls.append(input)
        return self.result

    def get_input_contract(self) -> PipelineStageInputContract:
        return _STUB_INPUT_CONTRACT_RGB


# ---------------------------------------------------------------------------
# FakeFaceDetection
# ---------------------------------------------------------------------------


class FakeFaceDetection:
    """Fake FaceDetectionInterface.

    Attributes:
        calls:   list of FaceDetectionInput passed to detect_faces()
        result:  FaceDetectionOutput returned from detect_faces()
    """

    def __init__(self, result: FaceDetectionOutput | None = None) -> None:
        self.calls: list[FaceDetectionInput] = []
        self.result: FaceDetectionOutput = result or FaceDetectionOutput(
            frame_id="", camera_id="", timestamp_ms=0, detections=[]
        )

    def detect_faces(self, input: FaceDetectionInput) -> FaceDetectionOutput:
        self.calls.append(input)
        return self.result

    def get_input_contract(self) -> PipelineStageInputContract:
        return _STUB_INPUT_CONTRACT_RGB


# ---------------------------------------------------------------------------
# FakeFaceRecognition
# ---------------------------------------------------------------------------


class FakeFaceRecognition:
    """Fake FaceRecognitionInterface.

    Attributes:
        calls:   list of FaceRecognitionInput passed to recognize()
        result:  FaceRecognitionOutput returned from recognize()
    """

    def __init__(self, result: FaceRecognitionOutput | None = None) -> None:
        self.calls: list[FaceRecognitionInput] = []
        self.result: FaceRecognitionOutput = result or FaceRecognitionOutput(
            frame_id="", camera_id="", timestamp_ms=0,
            person_found=False, person_id="UNKNOWN",
        )

    def recognize(self, face_input: FaceRecognitionInput) -> FaceRecognitionOutput:
        self.calls.append(face_input)
        return self.result

    def get_input_contract(self) -> PipelineStageInputContract:
        return _STUB_INPUT_CONTRACT_RGB


# ---------------------------------------------------------------------------
# FakePersonDirectory
# ---------------------------------------------------------------------------


class FakePersonDirectory:
    """Fake PersonDirectoryInterface.

    Attributes:
        calls:       list of person_id strings passed to get_person()
        lookup_map:  maps person_id → PersonDirectoryOutput; unknown IDs
                     return canonical UNKNOWN output
    """

    def __init__(
        self,
        lookup_map: dict[str, PersonDirectoryOutput] | None = None,
        raise_on_get_person: Exception | None = None,
    ) -> None:
        self.calls: list[str] = []
        self.lookup_map: dict[str, PersonDirectoryOutput] = lookup_map or {}
        self.raise_on_get_person = raise_on_get_person

    def get_person(self, person_id: str) -> PersonDirectoryOutput:
        self.calls.append(person_id)
        if self.raise_on_get_person is not None:
            raise self.raise_on_get_person
        return self.lookup_map.get(person_id, _UNKNOWN_PD_OUTPUT)
