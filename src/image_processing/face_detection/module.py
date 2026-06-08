from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TypedDict, runtime_checkable

import numpy as np

from image_processing.shared.contracts import (
    BoundingBox,
    FaceLandmarks,
    GeometrySpec,
    Image,
    OutputImageType,
    PipelineStageInputContract,
    Point,
    ResizePolicy,
)


# ---------------------------------------------------------------------------
# Public data structures — match Face Detection markdown spec §2.2, §7.1
# (BoundingBox, Point, FaceLandmarks are imported from image_processing.shared.contracts)
# ---------------------------------------------------------------------------


class DetectedFace(TypedDict):
    face_bbox: BoundingBox  # ROI-local, relative to roi_image
    landmarks: FaceLandmarks  # ROI-local, relative to roi_image


class FaceDetectionInput(TypedDict):
    frame_id: str
    camera_id: str
    timestamp_ms: int
    roi_image: Image


class FaceDetectionOutput(TypedDict):
    frame_id: str
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
    geometry_spec: GeometrySpec = field(
        default_factory=lambda: GeometrySpec(
            width=0, height=0, resize_policy=ResizePolicy.NONE
        )
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
        if not isinstance(roi_image, dict):
            raise TypeError("roi_image must be an Image struct (dict)")

        data = roi_image.get("data")
        if data is None:
            raise ValueError("roi_image.data is required")
        if not isinstance(data, np.ndarray):
            raise TypeError("roi_image.data must be a numpy.ndarray")
        if roi_image.get("width", 0) <= 0:
            raise ValueError("roi_image.width must be positive")
        if roi_image.get("height", 0) <= 0:
            raise ValueError("roi_image.height must be positive")

        self._validate_roi_contract(roi_image)

    def _validate_roi_contract(self, roi_image: Image) -> None:
        color_format = roi_image.get("color_format")
        if color_format != self._contract.color_format:
            raise ValueError(
                f"roi_image.color_format must be '{self._contract.color_format}', "
                f"got {color_format!r}"
            )
        layout = roi_image.get("layout")
        if layout != self._contract.layout:
            raise ValueError(
                f"roi_image.layout must be '{self._contract.layout}', "
                f"got {layout!r}"
            )
        dtype = roi_image.get("dtype")
        if dtype != self._contract.dtype:
            raise ValueError(
                f"roi_image.dtype must be '{self._contract.dtype}', "
                f"got {dtype!r}"
            )
        value_range = roi_image.get("value_range")
        # Normalize expected value_range for comparison (contract uses "[0, 255]" with space)
        # shared_contracts.md §6 uses "[0,255]" without space; DetectorInputContract uses "[0, 255]"
        # Accept both forms by stripping spaces from both sides
        expected_vr = self._contract.value_range.replace(" ", "")
        actual_vr = value_range.replace(" ", "") if isinstance(value_range, str) else value_range
        if actual_vr != expected_vr:
            raise ValueError(
                f"roi_image.value_range must be '{self._contract.value_range}', "
                f"got {value_range!r}"
            )
        data = roi_image["data"]
        if self._contract.layout == "HWC" and data.ndim not in (2, 3):
            raise ValueError("roi_image.data must be a 2D or 3D array for HWC layout")


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
# FaceDetectionOutputBuilder — spec §5.6
#
# Constructs the final FaceDetectionOutput from projected detections.
# Maps raw landmark lists into the canonical 5-point FaceLandmarks
# structure.  Model-specific landmark formats do not escape this component.
# ---------------------------------------------------------------------------

_LANDMARK_KEYS = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")

# Landmark configuration constants
EXPECTED_LANDMARK_COUNT = 5
DEFAULT_LANDMARK_COORDINATE = 0

# Landmark indices (matches SCRFD keypoint order)
LANDMARK_LEFT_EYE_INDEX = 0
LANDMARK_RIGHT_EYE_INDEX = 1
LANDMARK_NOSE_INDEX = 2
LANDMARK_MOUTH_LEFT_INDEX = 3
LANDMARK_MOUTH_RIGHT_INDEX = 4


class FaceDetectionOutputBuilder:
    def build(
        self,
        frame_id: str,
        camera_id: str,
        timestamp_ms: int,
        accepted_detections: list[AcceptedDetection],
    ) -> FaceDetectionOutput:
        detections: list[DetectedFace] = []
        for det in accepted_detections:
            landmarks = self._build_landmarks(det.landmarks)
            detections.append(
                DetectedFace(
                    face_bbox=det.bbox,
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
        """Build canonical 5-point landmarks structure from raw detection.
        
        Pads with zero landmarks if fewer than expected landmarks are provided.
        
        Args:
            raw_landmarks: List of detected landmark points
        
        Returns:
            FaceLandmarks: Canonical 5-point landmark structure
        """
        if len(raw_landmarks) < EXPECTED_LANDMARK_COUNT:
            # Pad with zero landmarks
            zero: Point = {
                "x": DEFAULT_LANDMARK_COORDINATE,
                "y": DEFAULT_LANDMARK_COORDINATE,
            }
            padded = list(raw_landmarks) + [zero] * (
                EXPECTED_LANDMARK_COUNT - len(raw_landmarks)
            )
        else:
            padded = raw_landmarks[:EXPECTED_LANDMARK_COUNT]
        
        return FaceLandmarks(
            left_eye=padded[LANDMARK_LEFT_EYE_INDEX],
            right_eye=padded[LANDMARK_RIGHT_EYE_INDEX],
            nose=padded[LANDMARK_NOSE_INDEX],
            mouth_left=padded[LANDMARK_MOUTH_LEFT_INDEX],
            mouth_right=padded[LANDMARK_MOUTH_RIGHT_INDEX],
        )


# ---------------------------------------------------------------------------
# FaceDetectionModule — spec §5.1
#
# Orchestration layer only.  Owns no detection logic.
# Wires subcomponents and invokes them in order:
#   validation → inference → postprocessing → output
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
        self._output_builder = FaceDetectionOutputBuilder()

    # ---- public API (spec §9.1) ------------------------------------------

    def get_input_contract(self) -> PipelineStageInputContract:
        return PipelineStageInputContract(
            output_image_type=OutputImageType.RGB_UINT8_HWC,
            geometry_spec=self._config.geometry_spec,
        )

    def get_backend_info(self) -> dict[str, str]:
        """Return detector backend diagnostics for runtime trace reporting."""
        detector_getter = getattr(self._detector_engine, "get_backend_info", None)
        detector_info = detector_getter() if callable(detector_getter) else {}
        if not isinstance(detector_info, dict):
            detector_info = {}
        geometry_spec = self._config.geometry_spec
        width = getattr(geometry_spec, "width", 0)
        height = getattr(geometry_spec, "height", 0)
        input_size = str(detector_info.get("input_size", "unknown"))
        if width > 0 and height > 0:
            input_size = f"{int(width)}x{int(height)}"
        return {
            "backend": str(detector_info.get("backend", "unknown")),
            "device_provider": str(detector_info.get("device_provider", "unknown")),
            "providers": str(detector_info.get("providers", "unknown")),
            "available_providers": str(detector_info.get("available_providers", "unknown")),
            "selected_provider": str(detector_info.get("selected_provider", "unknown")),
            "fallback_provider": str(detector_info.get("fallback_provider", "none")),
            "cuda_execution_provider_available": str(
                detector_info.get("cuda_execution_provider_available", "unknown")
            ),
            "cpu_forced": str(detector_info.get("cpu_forced", "unknown")),
            "gpu_not_used_reason": str(detector_info.get("gpu_not_used_reason", "")),
            "onnxruntime_package_variant": str(
                detector_info.get("onnxruntime_package_variant", "unknown")
            ),
            "model_path": str(detector_info.get("model_path", "unknown")),
            "model_name": str(detector_info.get("model_name", "unknown")),
            "input_size": input_size,
            "batch_mode": str(detector_info.get("batch_mode", "unknown")),
            "inference_precision": str(detector_info.get("inference_precision", "unknown")),
            "confidence_threshold": str(self._config.confidence_threshold),
        }

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

        # Step 2: inference — extract raw ndarray from Image struct and send to detector engine
        raw_detections = self._detector_engine.detect(face_input["roi_image"]["data"])

        # Step 3: postprocessing / acceptance
        accepted = self._postprocessor.accept(raw_detections)

        # Step 4: output construction — coordinates remain ROI-local (spec §7.2)
        return self._output_builder.build(
            frame_id=face_input["frame_id"],
            camera_id=face_input["camera_id"],
            timestamp_ms=face_input["timestamp_ms"],
            accepted_detections=accepted,
        )

    @staticmethod
    def _empty_output(face_input: FaceDetectionInput) -> FaceDetectionOutput:
        return FaceDetectionOutput(
            frame_id=face_input.get("frame_id", ""),
            camera_id=face_input.get("camera_id", ""),
            timestamp_ms=face_input.get("timestamp_ms", 0),
            detections=[],
        )
