from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Public data structures — match Face Detection markdown spec §2.2, §7.1
# ---------------------------------------------------------------------------


class BoundingBox(TypedDict):
    x: int
    y: int
    width: int
    height: int


class Point(TypedDict):
    x: int
    y: int


class FaceLandmarks(TypedDict):
    left_eye: Point
    right_eye: Point
    nose: Point
    mouth_left: Point
    mouth_right: Point


class DetectedFace(TypedDict):
    face_bbox_frame: BoundingBox
    landmarks: FaceLandmarks


class FaceDetectionInput(TypedDict):
    frame_id: int
    camera_id: str
    timestamp_ms: int
    roi_image: Any  # numpy.ndarray at runtime
    roi_bbox_frame: BoundingBox


class FaceDetectionOutput(TypedDict):
    frame_id: int
    camera_id: str
    timestamp_ms: int
    detections: list[DetectedFace]


# ---------------------------------------------------------------------------
# Configuration — match Face Detection markdown spec §3.1
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DetectorInputContract:
    color_format: str = "RGB"
    layout: str = "HWC"
    dtype: str = "uint8"
    value_range: str = "[0, 255]"


@dataclass(slots=True)
class FaceDetectionConfig:
    confidence_threshold: float = 0.5
    detector_input_contract: DetectorInputContract = field(
        default_factory=DetectorInputContract,
    )


# ---------------------------------------------------------------------------
# Internal data structures — spec §6 (never exposed through public API)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RawFaceDetection:
    bbox: BoundingBox
    confidence: float
    landmarks: list[Point]


@dataclass(slots=True)
class AcceptedDetection:
    bbox: BoundingBox
    confidence: float
    landmarks: list[Point]


@dataclass(slots=True)
class ProjectedDetection:
    bbox: BoundingBox
    confidence: float
    landmarks: list[Point]


# ---------------------------------------------------------------------------
# FaceDetectorEngine — spec §5.3 / §9.2
#
# Abstract detector interface.  The STUB or real detector (e.g. SCRFD)
# implements this protocol.  The rest of the module depends only on this
# abstraction, so the detector can be replaced without touching the
# public API, pipeline flow, or result format.
# ---------------------------------------------------------------------------


@runtime_checkable
class FaceDetectorEngine(Protocol):
    def detect(self, roi_image: np.ndarray) -> list[RawFaceDetection]: ...


# ---------------------------------------------------------------------------
# FaceDetectionInputValidator — spec §5.2
# ---------------------------------------------------------------------------


class FaceDetectionInputValidator:
    def __init__(self, detector_input_contract: DetectorInputContract) -> None:
        self._contract = detector_input_contract

    def validate(self, face_input: FaceDetectionInput) -> None:
        if "frame_id" not in face_input:
            raise ValueError("frame_id is required")
        if not face_input.get("camera_id"):
            raise ValueError("camera_id is required and must be non-empty")
        if "timestamp_ms" not in face_input:
            raise ValueError("timestamp_ms is required")

        roi_image = face_input.get("roi_image")
        if roi_image is None:
            raise ValueError("roi_image is required")
        if not isinstance(roi_image, np.ndarray):
            raise TypeError("roi_image must be a numpy.ndarray")

        roi_bbox = face_input.get("roi_bbox_frame")
        if roi_bbox is None:
            raise ValueError("roi_bbox_frame is required")
        if roi_bbox["width"] <= 0 or roi_bbox["height"] <= 0:
            raise ValueError("roi_bbox_frame width and height must be greater than zero")

        self._validate_roi_contract(roi_image)

    def _validate_roi_contract(self, roi_image: np.ndarray) -> None:
        if self._contract.layout == "HWC" and roi_image.ndim not in (2, 3):
            raise ValueError("roi_image must be a 2D or 3D array for HWC layout")

        expected_dtype = np.dtype(self._contract.dtype)
        if roi_image.dtype != expected_dtype:
            raise ValueError(
                f"roi_image dtype {roi_image.dtype} does not match "
                f"detector contract dtype {expected_dtype}"
            )


# ---------------------------------------------------------------------------
# FaceDetectionPostprocessor — spec §5.4
#
# The only place inside the module that decides whether a raw detector
# result becomes an accepted face detection.
# ---------------------------------------------------------------------------


class FaceDetectionPostprocessor:
    def __init__(self, confidence_threshold: float) -> None:
        self._confidence_threshold = confidence_threshold

    def accept(self, raw_detections: list[RawFaceDetection]) -> list[AcceptedDetection]:
        accepted: list[AcceptedDetection] = []
        for det in raw_detections:
            if det.confidence < self._confidence_threshold:
                continue
            if det.bbox["width"] <= 0 or det.bbox["height"] <= 0:
                continue
            accepted.append(
                AcceptedDetection(
                    bbox=det.bbox,
                    confidence=det.confidence,
                    landmarks=list(det.landmarks),
                )
            )
        return accepted


# ---------------------------------------------------------------------------
# FaceCoordinateProjector — spec §5.5
#
# Offsets ROI-local coordinates into full-frame coordinates using the
# roi_bbox_frame origin.
# ---------------------------------------------------------------------------


class FaceCoordinateProjector:
    def project(
        self,
        detections: list[AcceptedDetection],
        roi_bbox_frame: BoundingBox,
    ) -> list[ProjectedDetection]:
        ox = roi_bbox_frame["x"]
        oy = roi_bbox_frame["y"]
        projected: list[ProjectedDetection] = []
        for det in detections:
            projected_bbox: BoundingBox = {
                "x": det.bbox["x"] + ox,
                "y": det.bbox["y"] + oy,
                "width": det.bbox["width"],
                "height": det.bbox["height"],
            }
            projected_landmarks: list[Point] = [
                {"x": lm["x"] + ox, "y": lm["y"] + oy} for lm in det.landmarks
            ]
            projected.append(
                ProjectedDetection(
                    bbox=projected_bbox,
                    confidence=det.confidence,
                    landmarks=projected_landmarks,
                )
            )
        return projected


# ---------------------------------------------------------------------------
# FaceDetectionOutputBuilder — spec §5.6
#
# Constructs the final FaceDetectionOutput from projected detections.
# Maps raw landmark lists into the canonical 5-point FaceLandmarks
# structure.  Model-specific landmark formats do not escape this component.
# ---------------------------------------------------------------------------

_LANDMARK_KEYS = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")


class FaceDetectionOutputBuilder:
    def build(
        self,
        frame_id: int,
        camera_id: str,
        timestamp_ms: int,
        projected_detections: list[ProjectedDetection],
    ) -> FaceDetectionOutput:
        detections: list[DetectedFace] = []
        for det in projected_detections:
            landmarks = self._build_landmarks(det.landmarks)
            detections.append(
                DetectedFace(
                    face_bbox_frame=det.bbox,
                    landmarks=landmarks,
                )
            )
        return FaceDetectionOutput(
            frame_id=frame_id,
            camera_id=camera_id,
            timestamp_ms=timestamp_ms,
            detections=detections,
        )

    @staticmethod
    def _build_landmarks(raw_landmarks: list[Point]) -> FaceLandmarks:
        if len(raw_landmarks) < 5:
            zero: Point = {"x": 0, "y": 0}
            padded = list(raw_landmarks) + [zero] * (5 - len(raw_landmarks))
        else:
            padded = raw_landmarks[:5]
        return FaceLandmarks(
            left_eye=padded[0],
            right_eye=padded[1],
            nose=padded[2],
            mouth_left=padded[3],
            mouth_right=padded[4],
        )


# ---------------------------------------------------------------------------
# FaceDetectionModule — spec §5.1
#
# Orchestration layer only.  Owns no detection logic.
# Wires subcomponents and invokes them in order:
#   validation → inference → postprocessing → coordinate projection → output
#
# Error handling (spec §10): all errors are caught internally; the caller
# never receives exceptions.  In every failure scenario the module returns
# a structurally valid FaceDetectionOutput with empty detections.
# ---------------------------------------------------------------------------


class FaceDetectionModule:
    def __init__(
        self,
        config: FaceDetectionConfig | None = None,
        detector_engine: FaceDetectorEngine | None = None,
    ) -> None:
        self._config = config or FaceDetectionConfig()

        if detector_engine is None:
            raise ValueError(
                "detector_engine is required — pass a FaceDetectorEngine "
                "implementation (e.g. StubFaceDetectorEngine for testing)"
            )
        self._detector_engine: FaceDetectorEngine = detector_engine

        # Wire subcomponents with configuration (spec §3.2, §11.1)
        self._input_validator = FaceDetectionInputValidator(
            self._config.detector_input_contract,
        )
        self._postprocessor = FaceDetectionPostprocessor(
            self._config.confidence_threshold,
        )
        self._coordinate_projector = FaceCoordinateProjector()
        self._output_builder = FaceDetectionOutputBuilder()

    # ---- public API (spec §9.1) ------------------------------------------

    def detect_faces(self, face_input: FaceDetectionInput) -> FaceDetectionOutput:
        try:
            return self._detect_faces_internal(face_input)
        except Exception:
            # Spec §10: return valid output with empty detections on any error
            return self._empty_output(face_input)

    # ---- internal pipeline (spec §5.7) -----------------------------------

    def _detect_faces_internal(
        self, face_input: FaceDetectionInput
    ) -> FaceDetectionOutput:
        # Step 1: validate
        self._input_validator.validate(face_input)

        # Step 2: inference — send validated roi_image to detector engine
        raw_detections = self._detector_engine.detect(face_input["roi_image"])

        # Step 3: postprocessing / acceptance
        accepted = self._postprocessor.accept(raw_detections)

        # Step 4: coordinate projection (ROI-local → frame coordinates)
        projected = self._coordinate_projector.project(
            accepted, face_input["roi_bbox_frame"]
        )

        # Step 5: output construction
        return self._output_builder.build(
            frame_id=face_input["frame_id"],
            camera_id=face_input["camera_id"],
            timestamp_ms=face_input["timestamp_ms"],
            projected_detections=projected,
        )

    @staticmethod
    def _empty_output(face_input: FaceDetectionInput) -> FaceDetectionOutput:
        return FaceDetectionOutput(
            frame_id=face_input.get("frame_id", 0),
            camera_id=face_input.get("camera_id", ""),
            timestamp_ms=face_input.get("timestamp_ms", 0),
            detections=[],
        )
