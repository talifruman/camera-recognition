from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict, runtime_checkable

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Public data structures — match Motion Detection markdown spec §2.2 / §3.1
# ---------------------------------------------------------------------------


class BoundingBox(TypedDict):
    x: int
    y: int
    width: int
    height: int


class FramePacket(TypedDict):
    frame_id: str
    camera_id: str
    timestamp_ms: int
    image: Any  # numpy.ndarray at runtime


class MotionDetectionInput(TypedDict):
    current_frame: FramePacket
    previous_frame: FramePacket


class MotionResult(TypedDict):
    detected: bool
    bboxes: list[BoundingBox]


# ---------------------------------------------------------------------------
# Configuration — match Motion Detection markdown spec §9.1
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MotionDetectionConfig:
    motion_threshold: int = 25
    motion_fraction_threshold: float = 0.05
    min_bbox_area: int = 100


# ---------------------------------------------------------------------------
# Internal data structures — spec §10 (never exposed through public API)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MotionMeasurementResult:
    """Raw motion measurement; produced by algorithm, consumed by policy."""
    motion_fraction: float
    bboxes: list[BoundingBox] = field(default_factory=list)


@dataclass(slots=True)
class MotionDetectionResultInternal:
    """Binary detection decision; produced by policy, consumed by output builder."""
    detected: bool
    bboxes: list[BoundingBox] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MotionDetectionAlgorithm — spec §6.1 / §8.3
#
# Abstract motion measurement interface.  The module depends only on this
# abstraction, so any implementation (stub or real) can be substituted
# without touching the public API, pipeline flow, or result format.
# ---------------------------------------------------------------------------


@runtime_checkable
class MotionDetectionAlgorithm(Protocol):
    def measure(
        self,
        previous_image: np.ndarray,
        current_image: np.ndarray,
    ) -> MotionMeasurementResult: ...


# ---------------------------------------------------------------------------
# StubMotionDetectionAlgorithm — spec §5 "Option A: fixed deterministic output"
#
# Does NOT perform real frame differencing, thresholding, contour detection,
# connected-component analysis, or real bbox extraction.  All of that belongs
# to the REAL implementation (FrameDifferencingMotionDetector) which will
# replace this class later without touching any other component.
#
# Stub behavior:
#   - motion_fraction is always 0.2
#   - bboxes always contain one centered box (w = img_w // 4, h = img_h // 4)
#   - both values are constant across all inputs and runs
#
# motion_threshold and min_bbox_area are accepted at construction so the
# wiring structure is identical to what the real algorithm will expect.
# They are intentionally unused in this stub.
# ---------------------------------------------------------------------------


class StubMotionDetectionAlgorithm:
    _STUB_MOTION_FRACTION: float = 0.2

    def __init__(self, motion_threshold: int, min_bbox_area: int) -> None:
        # Accepted but unused in the stub — kept for clean real-algorithm swap
        self._motion_threshold = motion_threshold  # noqa: F841
        self._min_bbox_area = min_bbox_area  # noqa: F841

    def measure(
        self,
        previous_image: np.ndarray,
        current_image: np.ndarray,
    ) -> MotionMeasurementResult:
        img_h, img_w = current_image.shape[0], current_image.shape[1]

        box_w = max(1, img_w // 4)
        box_h = max(1, img_h // 4)
        box_x = max(0, img_w // 2 - box_w // 2)
        box_y = max(0, img_h // 2 - box_h // 2)

        # Clamp to stay strictly inside image bounds
        box_x = min(box_x, img_w - box_w)
        box_y = min(box_y, img_h - box_h)

        bbox: BoundingBox = {
            "x": box_x,
            "y": box_y,
            "width": box_w,
            "height": box_h,
        }

        return MotionMeasurementResult(
            motion_fraction=self._STUB_MOTION_FRACTION,
            bboxes=[bbox],
        )


# ---------------------------------------------------------------------------
# FrameDifferencingMotionDetector — spec §6.2 / §8.3
#
# Real implementation of MotionDetectionAlgorithm using frame differencing.
# Pipeline: absdiff → binary threshold → contour extraction → bbox filtering.
# Does not decide whether motion occurred — that belongs to MotionDecisionPolicy.
# ---------------------------------------------------------------------------


class FrameDifferencingMotionDetector:
    def __init__(self, motion_threshold: int, min_bbox_area: int) -> None:
        self._motion_threshold = motion_threshold
        self._min_bbox_area = min_bbox_area

    def measure(
        self,
        previous_image: np.ndarray,
        current_image: np.ndarray,
    ) -> MotionMeasurementResult:
        # Normalise both inputs to 2-D (H, W) — handles (H,W) and (H,W,1)
        prev2d = previous_image[:, :, 0] if previous_image.ndim == 3 else previous_image
        curr2d = current_image[:, :, 0] if current_image.ndim == 3 else current_image

        h, w = curr2d.shape[0], curr2d.shape[1]
        total_pixels = h * w

        # Step A — absolute difference
        diff = cv2.absdiff(prev2d, curr2d)

        # Step B — binary threshold: pixel is "changed" if abs_diff >= motion_threshold.
        # cv2.threshold marks values strictly greater than thresh, so subtract 1.
        _, mask = cv2.threshold(diff, self._motion_threshold - 1, 255, cv2.THRESH_BINARY)

        # Step F — motion fraction (computed from full mask before bbox filtering)
        changed_pixel_count = int(np.count_nonzero(mask))
        motion_fraction = changed_pixel_count / total_pixels

        # Step C — extract external contours from binary mask
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Steps D + E — convert each contour to a BoundingBox; discard small ones
        bboxes: list[BoundingBox] = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw * bh >= self._min_bbox_area:
                bboxes.append(
                    BoundingBox(x=int(x), y=int(y), width=int(bw), height=int(bh))
                )

        # Step G — return measurement (no detected decision here)
        return MotionMeasurementResult(
            motion_fraction=motion_fraction,
            bboxes=bboxes,
        )


# ---------------------------------------------------------------------------
# InputValidator — spec §2.4 / §8.2
#
# Validates all fields of MotionDetectionInput before any processing.
# Raises ValueError on any contract violation.
# Must not modify input or perform any preprocessing.
# ---------------------------------------------------------------------------

_VALID_GRAY_DTYPES = {np.dtype("uint8")}


class InputValidator:
    def validate(self, motion_input: MotionDetectionInput) -> None:
        current = motion_input.get("current_frame")
        previous = motion_input.get("previous_frame")

        if current is None:
            raise ValueError("current_frame is required and must be non-null")
        if previous is None:
            raise ValueError("previous_frame is required and must be non-null")

        self._validate_frame(current, "current_frame")
        self._validate_frame(previous, "previous_frame")

        self._validate_cross_frame(current, previous)

    def _validate_frame(self, frame: FramePacket, name: str) -> None:
        if frame.get("frame_id") is None:
            raise ValueError(f"{name}.frame_id is required")
        if not frame.get("camera_id"):
            raise ValueError(f"{name}.camera_id is required and must be non-empty")
        if frame.get("timestamp_ms") is None:
            raise ValueError(f"{name}.timestamp_ms is required")

        image = frame.get("image")
        if image is None:
            raise ValueError(f"{name}.image is required and must be non-null")
        if not isinstance(image, np.ndarray):
            raise TypeError(f"{name}.image must be a numpy.ndarray")

        self._validate_image_contract(image, name)

    def _validate_image_contract(self, image: np.ndarray, frame_name: str) -> None:
        """Enforce color_format=GRAY, layout=HWC, dtype=uint8, positive dims."""
        # dtype must be uint8
        if image.dtype not in _VALID_GRAY_DTYPES:
            raise ValueError(
                f"{frame_name}.image dtype must be uint8, got {image.dtype}"
            )

        # Layout HWC + color GRAY: accept (H, W) or (H, W, 1)
        if image.ndim == 2:
            pass  # (H, W) — valid grayscale HWC representation
        elif image.ndim == 3:
            if image.shape[2] != 1:
                raise ValueError(
                    f"{frame_name}.image must have color_format=GRAY; "
                    f"expected 1 channel, got {image.shape[2]}"
                )
        else:
            raise ValueError(
                f"{frame_name}.image must be 2D (H, W) or 3D (H, W, 1) for "
                f"GRAY HWC layout, got ndim={image.ndim}"
            )

        h, w = image.shape[0], image.shape[1]
        if h <= 0 or w <= 0:
            raise ValueError(
                f"{frame_name}.image dimensions must be greater than zero, "
                f"got height={h}, width={w}"
            )

    def _validate_cross_frame(
        self, current: FramePacket, previous: FramePacket
    ) -> None:
        if current.get("camera_id") != previous.get("camera_id"):
            raise ValueError(
                "current_frame.camera_id must equal previous_frame.camera_id"
            )

        if previous.get("timestamp_ms") > current.get("timestamp_ms"):  # type: ignore[operator]
            raise ValueError(
                "previous_frame.timestamp_ms must be <= current_frame.timestamp_ms "
                "(temporal ordering violation)"
            )

        # Both images must exist and have matching dimensions
        current_image = current.get("image")
        previous_image = previous.get("image")
        if (
            isinstance(current_image, np.ndarray)
            and isinstance(previous_image, np.ndarray)
        ):
            c_h, c_w = current_image.shape[0], current_image.shape[1]
            p_h, p_w = previous_image.shape[0], previous_image.shape[1]
            if c_h != p_h or c_w != p_w:
                raise ValueError(
                    f"current_frame.image dimensions ({c_w}x{c_h}) must match "
                    f"previous_frame.image dimensions ({p_w}x{p_h})"
                )


# ---------------------------------------------------------------------------
# MotionDecisionPolicy — spec §7 / §8.4
#
# The ONLY component that sets detected.  Applies motion_fraction_threshold.
# detected = True iff motion_fraction >= threshold AND at least one bbox.
# Does not access frame pixel data or perform any measurement.
# ---------------------------------------------------------------------------


class MotionDecisionPolicy:
    def __init__(self, motion_fraction_threshold: float) -> None:
        self._threshold = motion_fraction_threshold

    def decide(
        self, measurement: MotionMeasurementResult
    ) -> MotionDetectionResultInternal:
        if (
            measurement.motion_fraction >= self._threshold
            and len(measurement.bboxes) > 0
        ):
            return MotionDetectionResultInternal(
                detected=True,
                bboxes=list(measurement.bboxes),
            )
        return MotionDetectionResultInternal(detected=False, bboxes=[])


# ---------------------------------------------------------------------------
# MotionOutputBuilder — spec §8.5
#
# Builds the final MotionResult from the internal decision.
# Must not contain threshold logic, measurement logic, or coordinate
# transformation.  Ensures bboxes is empty when detected=False.
# ---------------------------------------------------------------------------


class MotionOutputBuilder:
    def build(
        self, result_internal: MotionDetectionResultInternal
    ) -> MotionResult:
        if result_internal.detected:
            return MotionResult(
                detected=True,
                bboxes=list(result_internal.bboxes),
            )
        return MotionResult(detected=False, bboxes=[])


# ---------------------------------------------------------------------------
# MotionDetectionManager — spec §8.1 / §8.6
#
# Orchestration layer only.  Owns no detection or measurement logic.
# Wires subcomponents and invokes them in pipeline order:
#   InputValidator → MotionDetectionAlgorithm → MotionDecisionPolicy
#   → MotionOutputBuilder
#
# Error handling (spec §11): validation failure and algorithm runtime failure
# both result in MotionResult(detected=False, bboxes=[]).
# ---------------------------------------------------------------------------


class MotionDetectionManager:
    def __init__(
        self,
        config: MotionDetectionConfig | None = None,
        algorithm: MotionDetectionAlgorithm | None = None,
    ) -> None:
        self._config = config or MotionDetectionConfig()

        # Default to the REAL frame-differencing implementation.
        # StubMotionDetectionAlgorithm is retained only for isolated test use.
        self._algorithm: MotionDetectionAlgorithm = algorithm or FrameDifferencingMotionDetector(
            motion_threshold=self._config.motion_threshold,
            min_bbox_area=self._config.min_bbox_area,
        )

        self._input_validator = InputValidator()
        self._decision_policy = MotionDecisionPolicy(
            self._config.motion_fraction_threshold
        )
        self._output_builder = MotionOutputBuilder()

    # ---- public API (spec §4) -------------------------------------------

    def process(self, motion_input: MotionDetectionInput) -> MotionResult:
        try:
            return self._process_internal(motion_input)
        except Exception:
            # Spec §11: all unhandled exceptions → detected=False
            return self._output_builder.build(
                MotionDetectionResultInternal(detected=False, bboxes=[])
            )

    # ---- internal pipeline (spec §8.6) ----------------------------------

    def _process_internal(
        self, motion_input: MotionDetectionInput
    ) -> MotionResult:
        # Step 1: validate input
        try:
            self._input_validator.validate(motion_input)
        except (ValueError, TypeError):
            return self._output_builder.build(
                MotionDetectionResultInternal(detected=False, bboxes=[])
            )

        # Step 2: measure motion (algorithm may raise — caught by outer try)
        measurement = self._algorithm.measure(
            motion_input["previous_frame"]["image"],
            motion_input["current_frame"]["image"],
        )

        # Step 3: apply decision policy
        result_internal = self._decision_policy.decide(measurement)

        # Step 4: build and return MotionResult
        return self._output_builder.build(result_internal)
