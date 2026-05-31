from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict, runtime_checkable

import cv2
import numpy as np

try:
    from src.shared.logger import logger
except ImportError:
    from shared.logger import logger

from image_processing.shared.contracts import (
    BoundingBox,
    GeometrySpec,
    Image,
    OutputImageType,
    PipelineStageInputContract,
    ResizePolicy,
)


# ---------------------------------------------------------------------------
# Public data structures — match Motion Detection markdown spec §2.2 / §3.1
# ---------------------------------------------------------------------------


class MotionInputFrame(TypedDict):
    """Per-frame input container for Motion Detection.

    Uses the shared Image struct instead of raw np.ndarray so that the public
    boundary is fully typed.  The name avoids ambiguity with the FTL-level
    FramePacket which carries raw image_bytes.
    """

    frame_id: str
    camera_id: str
    timestamp_ms: int
    image: Image  # shared Image struct; must be GRAYSCALE_UINT8_HWC


class MotionDetectionInput(TypedDict):
    current_frame: MotionInputFrame
    previous_frame: MotionInputFrame


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
    blur_kernel_size: int = 0
    morph_open_iterations: int = 0
    morph_close_iterations: int = 0
    dilation_iterations: int = 0
    min_aspect_ratio: float = 0.0
    max_aspect_ratio: float = 1000.0
    enable_global_motion_compensation: bool = False
    global_motion_method: str = "gftt_lk_affine"
    max_features: int = 300
    min_feature_matches: int = 20
    max_transform_shift: float = 25.0
    global_motion_changed_ratio_threshold: float = 0.30
    fallback_on_alignment_failure: bool = True
    enable_bbox_merging: bool = False
    bbox_merge_iou_threshold: float = 0.35
    bbox_merge_distance_threshold: float = 12.0
    enable_temporal_persistence: bool = False
    min_persistence_frames: int = 2
    persistence_iou_threshold: float = 0.35
    max_history_frames: int = 8
    camera_state_ttl_ms: int = 0
    debug_capture_enabled: bool = False
    debug_max_images: int = 2


# ---------------------------------------------------------------------------
# Internal data structures — spec §10 (never exposed through public API)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MotionMeasurementResult:
    """Raw motion measurement; produced by algorithm, consumed by policy."""
    motion_fraction: float
    bboxes: list[BoundingBox] = field(default_factory=list)
    debug_info: dict[str, Any] = field(default_factory=dict)
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(slots=True)
class MotionDetectionResultInternal:
    """Binary detection decision; produced by policy, consumed by output builder."""
    detected: bool
    bboxes: list[BoundingBox] = field(default_factory=list)


@dataclass(slots=True)
class _PersistentTrack:
    bbox: BoundingBox
    consecutive_hits: int


@dataclass(slots=True)
class _CameraMotionState:
    tracks: list[_PersistentTrack] = field(default_factory=list)
    history: list[list[BoundingBox]] = field(default_factory=list)
    last_seen_timestamp_ms: int = 0


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
# MotionDetectionInterface — public stage Protocol; spec §8.1
#
# RPM depends on this abstraction.  MotionDetectionManager is the concrete
# implementation.  Defined here (not in a separate file) so that it travels
# with the module and tests can import it directly.
# ---------------------------------------------------------------------------


@runtime_checkable
class MotionDetectionInterface(Protocol):
    def detect(self, input: MotionDetectionInput) -> MotionResult: ...
    def get_input_contract(self) -> PipelineStageInputContract: ...


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
    def __init__(
        self,
        motion_threshold: int,
        min_bbox_area: int,
        blur_kernel_size: int = 0,
        morph_open_iterations: int = 0,
        morph_close_iterations: int = 0,
        dilation_iterations: int = 0,
        min_aspect_ratio: float = 0.0,
        max_aspect_ratio: float = 1000.0,
        enable_global_motion_compensation: bool = False,
        global_motion_method: str = "gftt_lk_affine",
        max_features: int = 300,
        min_feature_matches: int = 20,
        max_transform_shift: float = 25.0,
        global_motion_changed_ratio_threshold: float = 0.30,
        fallback_on_alignment_failure: bool = True,
        debug_capture_enabled: bool = False,
        debug_max_images: int = 2,
    ) -> None:
        self._motion_threshold = motion_threshold
        self._min_bbox_area = min_bbox_area
        self._blur_kernel_size = self._normalize_kernel_size(blur_kernel_size)
        self._morph_open_iterations = max(0, int(morph_open_iterations))
        self._morph_close_iterations = max(0, int(morph_close_iterations))
        self._dilation_iterations = max(0, int(dilation_iterations))
        self._min_aspect_ratio = max(0.0, float(min_aspect_ratio))
        self._max_aspect_ratio = max(self._min_aspect_ratio, float(max_aspect_ratio))
        self._enable_global_motion_compensation = bool(enable_global_motion_compensation)
        self._global_motion_method = str(global_motion_method)
        self._max_features = max(1, int(max_features))
        self._min_feature_matches = max(1, int(min_feature_matches))
        self._max_transform_shift = max(0.0, float(max_transform_shift))
        self._global_motion_changed_ratio_threshold = max(0.0, float(global_motion_changed_ratio_threshold))
        self._fallback_on_alignment_failure = bool(fallback_on_alignment_failure)
        self._debug_capture_enabled = bool(debug_capture_enabled)
        self._debug_max_images = max(0, int(debug_max_images))
        self._morphology_kernel = np.ones((3, 3), dtype=np.uint8)

    def measure(
        self,
        previous_image: np.ndarray,
        current_image: np.ndarray,
    ) -> MotionMeasurementResult:
        # Normalise both inputs to 2-D (H, W) — handles (H,W) and (H,W,1)
        prev2d = previous_image[:, :, 0] if previous_image.ndim == 3 else previous_image
        curr2d = current_image[:, :, 0] if current_image.ndim == 3 else current_image

        h, w = curr2d.shape[0], curr2d.shape[1]

        prev_work = prev2d
        curr_work = curr2d
        if self._blur_kernel_size > 1:
            prev_work = cv2.GaussianBlur(prev_work, (self._blur_kernel_size, self._blur_kernel_size), 0)
            curr_work = cv2.GaussianBlur(curr_work, (self._blur_kernel_size, self._blur_kernel_size), 0)

        debug_images: dict[str, np.ndarray] = {}
        self._capture_debug_image(debug_images, "raw_previous_frame", prev2d)
        self._capture_debug_image(debug_images, "current_frame", curr2d)
        debug_info = self._make_debug_info()

        diff_before = cv2.absdiff(prev_work, curr_work)
        self._capture_debug_image(debug_images, "diff_before_alignment", diff_before)
        changed_ratio_before = 0.0

        aligned_previous = prev_work
        suppress_motion = False

        if self._enable_global_motion_compensation:
            _, changed_ratio_before = self._threshold_and_ratio(diff_before)
            debug_info["changed_pixel_ratio_before_alignment"] = changed_ratio_before
            alignment = self._estimate_global_alignment(prev_work, curr_work)
            debug_info["detected_feature_count"] = alignment["detected_feature_count"]
            debug_info["feature_match_count"] = alignment["feature_match_count"]
            debug_info["alignment_succeeded"] = alignment["alignment_succeeded"]
            debug_info["estimated_transform_matrix"] = alignment["estimated_transform_matrix"]
            debug_info["estimated_global_shift"] = alignment["estimated_global_shift"]
            debug_info["alignment_failure_reason"] = alignment["failure_reason"]

            if alignment["alignment_succeeded"]:
                aligned_previous = alignment["aligned_previous"]
                self._capture_debug_image(debug_images, "aligned_previous_frame", aligned_previous)
                if alignment["estimated_global_shift"] > self._max_transform_shift:
                    debug_info["camera_shake_detected"] = True
                    if not self._fallback_on_alignment_failure:
                        suppress_motion = True
                else:
                    debug_info["global_motion_compensation_applied"] = True
            else:
                debug_info["alignment_fallback_used"] = self._fallback_on_alignment_failure
                if not self._fallback_on_alignment_failure:
                    debug_info["camera_shake_detected"] = (
                        changed_ratio_before > self._global_motion_changed_ratio_threshold
                    )
                    suppress_motion = True

        if suppress_motion:
            zero_mask = np.zeros((h, w), dtype=np.uint8)
            self._capture_debug_image(debug_images, "aligned_previous_frame", aligned_previous)
            self._capture_debug_image(debug_images, "diff_after_alignment", diff_before)
            self._capture_debug_image(debug_images, "final_cleaned_motion_mask", zero_mask)
            debug_info["changed_pixel_ratio_after_alignment"] = 0.0
            return MotionMeasurementResult(
                motion_fraction=0.0,
                bboxes=[],
                debug_info=debug_info,
                debug_images=debug_images,
            )

        diff = cv2.absdiff(aligned_previous, curr_work)
        self._capture_debug_image(debug_images, "aligned_previous_frame", aligned_previous)
        self._capture_debug_image(debug_images, "diff_after_alignment", diff)
        mask, changed_ratio_after = self._threshold_and_ratio(diff)
        if not self._enable_global_motion_compensation:
            debug_info["changed_pixel_ratio_before_alignment"] = changed_ratio_after
        debug_info["changed_pixel_ratio_after_alignment"] = changed_ratio_after

        # Step C — denoise / reconnect before contour extraction.
        if self._morph_open_iterations > 0:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                self._morphology_kernel,
                iterations=self._morph_open_iterations,
            )
        if self._morph_close_iterations > 0:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                self._morphology_kernel,
                iterations=self._morph_close_iterations,
            )
        if self._dilation_iterations > 0:
            mask = cv2.dilate(mask, self._morphology_kernel, iterations=self._dilation_iterations)
        self._capture_debug_image(debug_images, "final_cleaned_motion_mask", mask)

        # Step F — motion fraction (computed from full mask before bbox filtering)
        changed_pixel_count = int(np.count_nonzero(mask))
        motion_fraction = changed_pixel_count / max(1, mask.size)

        # Step D — extract external contours from binary mask
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        debug_info["motion_bboxes_raw_count"] = int(len(contours))

        # Steps E + F — convert each contour to BoundingBox with area/shape filtering.
        bboxes: list[BoundingBox] = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if area < self._min_bbox_area:
                continue

            aspect_ratio = bw / max(1, bh)
            if aspect_ratio < self._min_aspect_ratio:
                continue
            if aspect_ratio > self._max_aspect_ratio:
                continue

            bboxes.append(BoundingBox(x=int(x), y=int(y), width=int(bw), height=int(bh)))

        debug_info["motion_bboxes_after_filter_count"] = int(len(bboxes))

        # Step G — return measurement (no detected decision here)
        return MotionMeasurementResult(
            motion_fraction=motion_fraction,
            bboxes=bboxes,
            debug_info=debug_info,
            debug_images=debug_images,
        )

    def _estimate_global_alignment(
        self,
        previous_image: np.ndarray,
        current_image: np.ndarray,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "alignment_succeeded": False,
            "detected_feature_count": 0,
            "feature_match_count": 0,
            "estimated_transform_matrix": None,
            "estimated_global_shift": 0.0,
            "failure_reason": None,
            "aligned_previous": previous_image,
        }

        if self._global_motion_method != "gftt_lk_affine":
            result["failure_reason"] = "unsupported_global_motion_method"
            return result

        features = cv2.goodFeaturesToTrack(
            previous_image,
            maxCorners=self._max_features,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
        )
        if features is None or len(features) == 0:
            result["failure_reason"] = "no_features_detected"
            return result

        result["detected_feature_count"] = int(len(features))
        tracked_features = features.astype(np.float32)
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(previous_image, current_image, tracked_features, None)
        if next_points is None or status is None:
            result["failure_reason"] = "optical_flow_failed"
            return result

        valid_mask = status.reshape(-1) == 1
        prev_points = features.reshape(-1, 2)[valid_mask]
        curr_points = next_points.reshape(-1, 2)[valid_mask]
        result["feature_match_count"] = int(len(prev_points))
        if len(prev_points) < self._min_feature_matches:
            result["failure_reason"] = "insufficient_feature_matches"
            return result

        transform, _ = cv2.estimateAffinePartial2D(prev_points, curr_points, method=cv2.RANSAC)
        if transform is None:
            result["failure_reason"] = "transform_estimation_failed"
            return result

        shift_x = float(transform[0, 2])
        shift_y = float(transform[1, 2])
        estimated_shift = float(np.hypot(shift_x, shift_y))
        aligned_previous = cv2.warpAffine(
            previous_image,
            transform,
            (current_image.shape[1], current_image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        result["alignment_succeeded"] = True
        result["estimated_transform_matrix"] = transform.tolist()
        result["estimated_global_shift"] = estimated_shift
        result["aligned_previous"] = aligned_previous
        return result

    def _make_debug_info(self) -> dict[str, Any]:
        return {
            "enable_global_motion_compensation": self._enable_global_motion_compensation,
            "global_motion_method": self._global_motion_method,
            "max_features": self._max_features,
            "min_feature_matches": self._min_feature_matches,
            "max_transform_shift": self._max_transform_shift,
            "global_motion_changed_ratio_threshold": self._global_motion_changed_ratio_threshold,
            "fallback_on_alignment_failure": self._fallback_on_alignment_failure,
            "detected_feature_count": 0,
            "feature_match_count": 0,
            "alignment_succeeded": False,
            "alignment_fallback_used": False,
            "global_motion_compensation_applied": False,
            "estimated_transform_matrix": None,
            "estimated_global_shift": 0.0,
            "camera_shake_detected": False,
            "alignment_failure_reason": None,
            "changed_pixel_ratio_before_alignment": 0.0,
            "changed_pixel_ratio_after_alignment": 0.0,
        }

    def _capture_debug_image(
        self,
        debug_images: dict[str, np.ndarray],
        key: str,
        image: np.ndarray,
    ) -> None:
        if not self._debug_capture_enabled:
            return
        if key in debug_images:
            return
        if len(debug_images) >= self._debug_max_images:
            return
        debug_images[key] = image.copy()

    def _threshold_and_ratio(self, diff: np.ndarray) -> tuple[np.ndarray, float]:
        _, mask = cv2.threshold(diff, self._motion_threshold - 1, 255, cv2.THRESH_BINARY)
        ratio = float(np.count_nonzero(mask)) / float(max(1, mask.size))
        return mask, ratio

    @staticmethod
    def _normalize_kernel_size(value: int) -> int:
        size = max(0, int(value))
        if size <= 1:
            return 0
        # Gaussian blur kernel size must be odd.
        return size if size % 2 == 1 else size + 1


# ---------------------------------------------------------------------------
# InputValidator — spec §2.4 / §8.2
#
# Validates all fields of MotionDetectionInput before any processing.
# Raises ValueError / TypeError on any contract violation.
# Must not modify input or perform any preprocessing.
# ---------------------------------------------------------------------------


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

    def _validate_frame(self, frame: MotionInputFrame, name: str) -> None:
        if frame.get("frame_id") is None:
            raise ValueError(f"{name}.frame_id is required")
        if not frame.get("camera_id"):
            raise ValueError(f"{name}.camera_id is required and must be non-empty")

        ts = frame.get("timestamp_ms")
        if ts is None:
            raise ValueError(f"{name}.timestamp_ms is required")
        if ts < 0:
            raise ValueError(
                f"{name}.timestamp_ms must be >= 0, got {ts}"
            )

        image = frame.get("image")
        if image is None:
            raise ValueError(f"{name}.image is required and must be non-null")
        if not isinstance(image, dict):
            raise TypeError(
                f"{name}.image must be an Image struct (dict), "
                f"got {type(image).__name__}"
            )

        self._validate_image_contract(image, name)

    def _validate_image_contract(self, image: Image, frame_name: str) -> None:
        """Enforce Image struct metadata: GRAY, HWC, uint8, [0,255], positive dims, shape."""
        data = image.get("data")
        if data is None or not isinstance(data, np.ndarray):
            raise ValueError(
                f"{frame_name}.image.data must be a non-null numpy.ndarray"
            )

        color_format = image.get("color_format")
        if color_format != "GRAY":
            raise ValueError(
                f"{frame_name}.image.color_format must be 'GRAY', got {color_format!r}"
            )

        layout = image.get("layout")
        if layout != "HWC":
            raise ValueError(
                f"{frame_name}.image.layout must be 'HWC', got {layout!r}"
            )

        dtype_str = image.get("dtype")
        if dtype_str != "uint8":
            raise ValueError(
                f"{frame_name}.image.dtype must be 'uint8', got {dtype_str!r}"
            )

        value_range = image.get("value_range")
        if value_range != "[0,255]":
            raise ValueError(
                f"{frame_name}.image.value_range must be '[0,255]', got {value_range!r}"
            )

        width = image.get("width", 0)
        height = image.get("height", 0)
        if width <= 0:
            raise ValueError(
                f"{frame_name}.image.width must be > 0, got {width}"
            )
        if height <= 0:
            raise ValueError(
                f"{frame_name}.image.height must be > 0, got {height}"
            )

        # Verify data.shape is consistent with metadata (HWC GRAY: (H,W) or (H,W,1))
        shape = data.shape
        if data.ndim == 2:
            if shape[0] != height or shape[1] != width:
                raise ValueError(
                    f"{frame_name}.image.data.shape {shape} is inconsistent with "
                    f"declared width={width}, height={height}"
                )
        elif data.ndim == 3:
            if shape[0] != height or shape[1] != width or shape[2] != 1:
                raise ValueError(
                    f"{frame_name}.image.data.shape {shape} is inconsistent with "
                    f"declared width={width}, height={height} and GRAY HWC layout"
                )
        else:
            raise ValueError(
                f"{frame_name}.image.data must be 2D (H,W) or 3D (H,W,1) for "
                f"GRAY HWC layout, got ndim={data.ndim}"
            )

    def _validate_cross_frame(
        self, current: MotionInputFrame, previous: MotionInputFrame
    ) -> None:
        if current.get("camera_id") != previous.get("camera_id"):
            raise ValueError(
                "current_frame.camera_id must equal previous_frame.camera_id"
            )

        prev_ts = previous.get("timestamp_ms", 0)
        curr_ts = current.get("timestamp_ms", 0)
        if prev_ts > curr_ts:
            raise ValueError(
                "previous_frame.timestamp_ms must be <= current_frame.timestamp_ms "
                "(temporal ordering violation)"
            )

        # Compare declared width/height from Image struct metadata
        current_image = current.get("image")
        previous_image = previous.get("image")
        if isinstance(current_image, dict) and isinstance(previous_image, dict):
            c_w = current_image.get("width", 0)
            c_h = current_image.get("height", 0)
            p_w = previous_image.get("width", 0)
            p_h = previous_image.get("height", 0)
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
        self._last_debug_info: dict[str, Any] = {}
        self._last_debug_images: dict[str, np.ndarray] = {}
        self._motion_history_by_camera_id: dict[str, _CameraMotionState] = {}

        # Default to the REAL frame-differencing implementation.
        # StubMotionDetectionAlgorithm is retained only for isolated test use.
        self._algorithm: MotionDetectionAlgorithm = algorithm or FrameDifferencingMotionDetector(
            motion_threshold=self._config.motion_threshold,
            min_bbox_area=self._config.min_bbox_area,
            blur_kernel_size=self._config.blur_kernel_size,
            morph_open_iterations=self._config.morph_open_iterations,
            morph_close_iterations=self._config.morph_close_iterations,
            dilation_iterations=self._config.dilation_iterations,
            min_aspect_ratio=self._config.min_aspect_ratio,
            max_aspect_ratio=self._config.max_aspect_ratio,
            enable_global_motion_compensation=self._config.enable_global_motion_compensation,
            global_motion_method=self._config.global_motion_method,
            max_features=self._config.max_features,
            min_feature_matches=self._config.min_feature_matches,
            max_transform_shift=self._config.max_transform_shift,
            global_motion_changed_ratio_threshold=self._config.global_motion_changed_ratio_threshold,
            fallback_on_alignment_failure=self._config.fallback_on_alignment_failure,
            debug_capture_enabled=self._config.debug_capture_enabled,
            debug_max_images=self._config.debug_max_images,
        )

        self._input_validator = InputValidator()
        self._decision_policy = MotionDecisionPolicy(
            self._config.motion_fraction_threshold
        )
        self._output_builder = MotionOutputBuilder()

    # ---- public stage API (spec §4) -------------------------------------

    def detect(self, input: MotionDetectionInput) -> MotionResult:
        """Single RPM-facing public stage API.

        Accepts a MotionDetectionInput whose frames carry shared Image structs.
        Returns MotionResult(detected=False, bboxes=[]) for any invalid input
        or algorithm failure (spec §11).
        """
        try:
            return self._process_internal(input)
        except (ValueError, TypeError) as exc:
            camera_id = self._safe_camera_id(input)
            logger.warning(
                "Motion detection validation failed for camera_id=%s: %s",
                camera_id,
                exc,
            )
            self._last_debug_info = {
                "failure_type": "validation_error",
                "error": str(exc),
            }
            self._last_debug_images = {}
            return self._output_builder.build(
                MotionDetectionResultInternal(detected=False, bboxes=[])
            )
        except Exception as exc:
            camera_id = self._safe_camera_id(input)
            logger.exception(
                "Motion detection runtime failure for camera_id=%s",
                camera_id,
            )
            self._last_debug_info = {
                "failure_type": "runtime_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            self._last_debug_images = {}
            # Spec §11: all unhandled exceptions → detected=False
            return self._output_builder.build(
                MotionDetectionResultInternal(detected=False, bboxes=[])
            )

    def process(self, motion_input: MotionDetectionInput) -> MotionResult:
        """Deprecated wrapper — calls detect().  Do not use for new code."""
        return self.detect(motion_input)

    # ---- get_input_contract (spec §4 / §9) ------------------------------

    def get_input_contract(self) -> PipelineStageInputContract:
        """Return the image format and geometry RPM must request from FTL."""
        return PipelineStageInputContract(
            output_image_type=OutputImageType.GRAYSCALE_UINT8_HWC,
            geometry_spec=GeometrySpec(
                width=0,
                height=0,
                resize_policy=ResizePolicy.NONE,
            ),
        )

    def get_backend_info(self) -> dict[str, str]:
        """Return motion detection backend diagnostics for runtime reporting."""
        algorithm_name = type(self._algorithm).__name__
        implementation = "stub" if "Stub" in algorithm_name else "real"
        return {
            "backend": "cv2_frame_differencing" if implementation == "real" else algorithm_name,
            "device_provider": "cpu",
            "implementation": implementation,
            "algorithm": algorithm_name,
        }

    def get_last_debug_info(self) -> dict[str, Any]:
        return dict(self._last_debug_info)

    def get_last_debug_images(self) -> dict[str, np.ndarray]:
        return {key: value.copy() for key, value in self._last_debug_images.items()}

    def reset_camera_history(self, camera_id: str) -> None:
        self._motion_history_by_camera_id.pop(camera_id, None)

    def reset_all_history(self) -> None:
        self._motion_history_by_camera_id.clear()

    # ---- internal pipeline (spec §8.6) ----------------------------------

    def _process_internal(
        self, motion_input: MotionDetectionInput
    ) -> MotionResult:
        # Step 1: validate input
        self._input_validator.validate(motion_input)

        # Step 2: measure motion — extract raw numpy arrays from Image struct
        # (algorithm layer works with np.ndarray internally)
        measurement = self._algorithm.measure(
            motion_input["previous_frame"]["image"]["data"],
            motion_input["current_frame"]["image"]["data"],
        )
        debug_info = dict(measurement.debug_info)

        current_frame = motion_input["current_frame"]
        image_meta = current_frame["image"]
        image_width = int(image_meta["width"])
        image_height = int(image_meta["height"])

        merged_bboxes = self._merge_bboxes_for_frame(
            measurement.bboxes,
            image_width,
            image_height,
        )
        debug_info.setdefault("motion_bboxes_raw_count", int(len(measurement.bboxes)))
        debug_info.setdefault("motion_bboxes_after_filter_count", int(len(measurement.bboxes)))
        debug_info["motion_bboxes_after_merge_count"] = int(len(merged_bboxes))

        persisted_bboxes = self._apply_temporal_persistence(
            camera_id=current_frame["camera_id"],
            current_timestamp_ms=int(current_frame["timestamp_ms"]),
            current_bboxes=merged_bboxes,
        )
        debug_info["motion_bboxes_after_persistence_count"] = int(len(persisted_bboxes))

        measurement = MotionMeasurementResult(
            motion_fraction=measurement.motion_fraction,
            bboxes=persisted_bboxes,
            debug_info=debug_info,
            debug_images=measurement.debug_images,
        )

        self._last_debug_info = debug_info
        self._last_debug_images = dict(measurement.debug_images)

        # Step 3: apply decision policy
        result_internal = self._decision_policy.decide(measurement)

        # Step 4: build and return MotionResult
        return self._output_builder.build(result_internal)

    @staticmethod
    def _safe_camera_id(motion_input: Any) -> str:
        if not isinstance(motion_input, dict):
            return "unknown"
        current_frame = motion_input.get("current_frame")
        if not isinstance(current_frame, dict):
            return "unknown"
        camera_id = current_frame.get("camera_id")
        if not camera_id:
            return "unknown"
        return str(camera_id)

    def _prune_temporal_state(self, current_timestamp_ms: int) -> None:
        ttl_ms = max(0, int(self._config.camera_state_ttl_ms))
        if ttl_ms <= 0:
            return
        expiration_threshold = current_timestamp_ms - ttl_ms
        expired_camera_ids = [
            camera_id
            for camera_id, state in self._motion_history_by_camera_id.items()
            if state.last_seen_timestamp_ms < expiration_threshold
        ]
        for camera_id in expired_camera_ids:
            self._motion_history_by_camera_id.pop(camera_id, None)

    def _merge_bboxes_for_frame(
        self,
        bboxes: list[BoundingBox],
        image_width: int,
        image_height: int,
    ) -> list[BoundingBox]:
        normalized = [
            self._clamp_bbox_to_image(bbox, image_width, image_height)
            for bbox in bboxes
        ]
        normalized = [bbox for bbox in normalized if bbox is not None]

        if not self._config.enable_bbox_merging:
            return [bbox for bbox in normalized if bbox is not None]

        merged = [bbox for bbox in normalized if bbox is not None]
        changed = True
        while changed:
            changed = False
            next_boxes: list[BoundingBox] = []
            consumed = [False] * len(merged)

            for i, base_bbox in enumerate(merged):
                if consumed[i]:
                    continue

                candidate = base_bbox
                consumed[i] = True

                for j in range(i + 1, len(merged)):
                    if consumed[j]:
                        continue

                    other = merged[j]
                    if self._should_merge(candidate, other):
                        candidate = self._union_bboxes(candidate, other)
                        consumed[j] = True
                        changed = True

                clamped = self._clamp_bbox_to_image(candidate, image_width, image_height)
                if clamped is not None:
                    next_boxes.append(clamped)

            merged = next_boxes

        return merged

    def _apply_temporal_persistence(
        self,
        *,
        camera_id: str,
        current_timestamp_ms: int,
        current_bboxes: list[BoundingBox],
    ) -> list[BoundingBox]:
        if not self._config.enable_temporal_persistence:
            return list(current_bboxes)

        self._prune_temporal_state(current_timestamp_ms)
        state = self._motion_history_by_camera_id.setdefault(camera_id, _CameraMotionState())
        state.last_seen_timestamp_ms = current_timestamp_ms
        state.history.append(list(current_bboxes))
        max_history = max(1, int(self._config.max_history_frames))
        if len(state.history) > max_history:
            state.history = state.history[-max_history:]

        previous_tracks = list(state.tracks)
        updated_tracks: list[_PersistentTrack] = []
        matched_track_indices: set[int] = set()

        for bbox in current_bboxes:
            best_track_index: int | None = None
            best_iou = 0.0
            for idx, track in enumerate(previous_tracks):
                if idx in matched_track_indices:
                    continue
                overlap = self._bbox_iou(track.bbox, bbox)
                if overlap >= self._config.persistence_iou_threshold and overlap > best_iou:
                    best_iou = overlap
                    best_track_index = idx

            if best_track_index is None:
                updated_tracks.append(_PersistentTrack(bbox=bbox, consecutive_hits=1))
                continue

            matched_track_indices.add(best_track_index)
            previous = previous_tracks[best_track_index]
            updated_tracks.append(
                _PersistentTrack(
                    bbox=bbox,
                    consecutive_hits=previous.consecutive_hits + 1,
                )
            )

        state.tracks = updated_tracks

        minimum = max(1, int(self._config.min_persistence_frames))
        persistent = [
            track.bbox
            for track in updated_tracks
            if track.consecutive_hits >= minimum
        ]
        return persistent

    def _should_merge(self, first: BoundingBox, second: BoundingBox) -> bool:
        if self._bbox_iou(first, second) >= self._config.bbox_merge_iou_threshold:
            return True

        distance = self._bbox_distance(first, second)
        return distance <= self._config.bbox_merge_distance_threshold

    @staticmethod
    def _union_bboxes(first: BoundingBox, second: BoundingBox) -> BoundingBox:
        x1 = min(int(first["x"]), int(second["x"]))
        y1 = min(int(first["y"]), int(second["y"]))
        x2 = max(int(first["x"]) + int(first["width"]), int(second["x"]) + int(second["width"]))
        y2 = max(int(first["y"]) + int(first["height"]), int(second["y"]) + int(second["height"]))
        return BoundingBox(x=x1, y=y1, width=max(1, x2 - x1), height=max(1, y2 - y1))

    @staticmethod
    def _bbox_iou(first: BoundingBox, second: BoundingBox) -> float:
        left = max(int(first["x"]), int(second["x"]))
        top = max(int(first["y"]), int(second["y"]))
        right = min(int(first["x"]) + int(first["width"]), int(second["x"]) + int(second["width"]))
        bottom = min(int(first["y"]) + int(first["height"]), int(second["y"]) + int(second["height"]))

        inter_w = max(0, right - left)
        inter_h = max(0, bottom - top)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0

        first_area = int(first["width"]) * int(first["height"])
        second_area = int(second["width"]) * int(second["height"])
        union_area = max(1, first_area + second_area - inter_area)
        return inter_area / union_area

    @staticmethod
    def _bbox_distance(first: BoundingBox, second: BoundingBox) -> float:
        first_left = int(first["x"])
        first_top = int(first["y"])
        first_right = first_left + int(first["width"])
        first_bottom = first_top + int(first["height"])

        second_left = int(second["x"])
        second_top = int(second["y"])
        second_right = second_left + int(second["width"])
        second_bottom = second_top + int(second["height"])

        dx = max(second_left - first_right, first_left - second_right, 0)
        dy = max(second_top - first_bottom, first_top - second_bottom, 0)
        return float(np.hypot(dx, dy))

    @staticmethod
    def _clamp_bbox_to_image(
        bbox: BoundingBox,
        image_width: int,
        image_height: int,
    ) -> BoundingBox | None:
        if image_width <= 0 or image_height <= 0:
            return None

        x1 = max(0, int(bbox["x"]))
        y1 = max(0, int(bbox["y"]))
        x2 = min(image_width, int(bbox["x"]) + int(bbox["width"]))
        y2 = min(image_height, int(bbox["y"]) + int(bbox["height"]))

        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return None

        return BoundingBox(x=x1, y=y1, width=width, height=height)
