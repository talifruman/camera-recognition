from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_processing.shared.contracts import (  # type: ignore[import-not-found]
    BoundingBox,
    GeometrySpec,
    Image,
    OutputImageType,
    ResizePolicy,
)
from image_processing.motion_detection import (  # type: ignore[import-not-found]
    FrameDifferencingMotionDetector,
    InputValidator,
    MotionDetectionAlgorithm,
    MotionDetectionConfig,
    MotionDetectionInput,
    MotionDetectionInterface,
    MotionDetectionManager,
    MotionDetectionResultInternal,
    MotionInputFrame,
    MotionMeasurementResult,
    MotionResult,
    StubMotionDetectionAlgorithm,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_image(height: int = 64, width: int = 64) -> Image:
    """Return a valid GRAYSCALE_UINT8_HWC Image struct."""
    return Image(
        data=np.zeros((height, width), dtype=np.uint8),
        width=width,
        height=height,
        color_format="GRAY",
        layout="HWC",
        dtype="uint8",
        value_range="[0,255]",
    )


def _make_image_3d(height: int = 64, width: int = 64) -> Image:
    """Return a valid GRAYSCALE_UINT8_HWC Image struct with (H,W,1) data."""
    return Image(
        data=np.zeros((height, width, 1), dtype=np.uint8),
        width=width,
        height=height,
        color_format="GRAY",
        layout="HWC",
        dtype="uint8",
        value_range="[0,255]",
    )


def _make_image_from_array(arr: np.ndarray) -> Image:
    height, width = arr.shape[:2]
    return Image(
        data=arr,
        width=width,
        height=height,
        color_format="GRAY",
        layout="HWC",
        dtype="uint8",
        value_range="[0,255]",
    )


def _make_textured_scene(height: int = 192, width: int = 256, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scene = rng.integers(30, 120, size=(height, width), dtype=np.uint8)

    for y in range(0, height, 24):
        cv2.line(scene, (0, y), (width - 1, y), 140 + (y % 50), 1)
    for x in range(0, width, 24):
        cv2.line(scene, (x, 0), (x, height - 1), 80 + (x % 60), 1)

    cv2.rectangle(scene, (20, 20), (70, 70), 220, -1)
    cv2.rectangle(scene, (width - 90, 25), (width - 35, 85), 15, -1)
    cv2.circle(scene, (width // 2, height // 2), 26, 200, 3)
    cv2.circle(scene, (width // 3, height // 3), 14, 250, -1)
    cv2.putText(scene, "RPM", (width // 2 - 40, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 210, 2, cv2.LINE_AA)
    return scene


def _shift_image(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    height, width = image.shape[:2]
    transform = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _make_motion_input_from_arrays(
    previous_image: np.ndarray,
    current_image: np.ndarray,
    camera_id: str = "cam-a",
) -> MotionDetectionInput:
    return MotionDetectionInput(
        previous_frame=MotionInputFrame(
            frame_id="frame_0001",
            camera_id=camera_id,
            timestamp_ms=1000,
            image=_make_image_from_array(previous_image),
        ),
        current_frame=MotionInputFrame(
            frame_id="frame_0002",
            camera_id=camera_id,
            timestamp_ms=2000,
            image=_make_image_from_array(current_image),
        ),
    )


def _make_frame(
    *,
    frame_id: str = "frame_0001",
    camera_id: str = "cam-a",
    timestamp_ms: int = 1000,
    height: int = 64,
    width: int = 64,
    image: Image | None = None,
) -> MotionInputFrame:
    return MotionInputFrame(
        frame_id=frame_id,
        camera_id=camera_id,
        timestamp_ms=timestamp_ms,
        image=image if image is not None else _make_image(height, width),
    )


def _make_input(
    *,
    current_ts: int = 2000,
    previous_ts: int = 1000,
    camera_id: str = "cam-a",
    height: int = 64,
    width: int = 64,
) -> MotionDetectionInput:
    return MotionDetectionInput(
        current_frame=_make_frame(
            frame_id="frame_0002",
            camera_id=camera_id,
            timestamp_ms=current_ts,
            height=height,
            width=width,
        ),
        previous_frame=_make_frame(
            frame_id="frame_0001",
            camera_id=camera_id,
            timestamp_ms=previous_ts,
            height=height,
            width=width,
        ),
    )


def _make_manager(
    motion_fraction_threshold: float = 0.05,
    algorithm: MotionDetectionAlgorithm | None = None,
    debug_capture_enabled: bool = False,
    debug_max_images: int = 2,
) -> MotionDetectionManager:
    config = MotionDetectionConfig(
        motion_threshold=25,
        motion_fraction_threshold=motion_fraction_threshold,
        min_bbox_area=100,
        debug_capture_enabled=debug_capture_enabled,
        debug_max_images=debug_max_images,
    )
    return MotionDetectionManager(config=config, algorithm=algorithm)


def _make_stub_manager(
    motion_fraction_threshold: float = 0.05,
) -> MotionDetectionManager:
    """Return a manager explicitly wired to StubMotionDetectionAlgorithm."""
    config = MotionDetectionConfig(
        motion_threshold=25,
        motion_fraction_threshold=motion_fraction_threshold,
        min_bbox_area=100,
    )
    stub_algo = StubMotionDetectionAlgorithm(
        motion_threshold=config.motion_threshold,
        min_bbox_area=config.min_bbox_area,
    )
    return MotionDetectionManager(config=config, algorithm=stub_algo)


# ---------------------------------------------------------------------------
# ValidationTests — spec §2.4 / §11
# ---------------------------------------------------------------------------


class ValidationTests(unittest.TestCase):
    """Each validation rule returns MotionResult(detected=False, bboxes=[])."""

    def _assert_no_motion(self, result: MotionResult) -> None:
        self.assertFalse(result["detected"])
        self.assertEqual(result["bboxes"], [])

    def test_missing_current_frame_returns_no_motion(self) -> None:
        manager = _make_manager()
        bad: MotionDetectionInput = {  # type: ignore[typeddict-item]
            "previous_frame": _make_frame(),
        }
        self._assert_no_motion(manager.detect(bad))

    def test_missing_previous_frame_returns_no_motion(self) -> None:
        manager = _make_manager()
        bad: MotionDetectionInput = {  # type: ignore[typeddict-item]
            "current_frame": _make_frame(),
        }
        self._assert_no_motion(manager.detect(bad))

    def test_empty_current_camera_id_returns_no_motion(self) -> None:
        manager = _make_manager()
        inp = _make_input()
        inp["current_frame"]["camera_id"] = ""
        self._assert_no_motion(manager.detect(inp))

    def test_empty_previous_camera_id_returns_no_motion(self) -> None:
        manager = _make_manager()
        inp = _make_input()
        inp["previous_frame"]["camera_id"] = ""
        self._assert_no_motion(manager.detect(inp))

    def test_camera_id_mismatch_returns_no_motion(self) -> None:
        manager = _make_manager()
        inp = _make_input()
        inp["previous_frame"]["camera_id"] = "cam-b"
        self._assert_no_motion(manager.detect(inp))

    def test_timestamp_ordering_violation_returns_no_motion(self) -> None:
        manager = _make_manager()
        # previous_ts > current_ts — violates spec §2.3
        inp = _make_input(current_ts=1000, previous_ts=2000)
        self._assert_no_motion(manager.detect(inp))

    def test_equal_timestamps_are_accepted(self) -> None:
        """previous_ts == current_ts is valid per spec (≤ relationship)."""
        manager = _make_manager()
        inp = _make_input(current_ts=1000, previous_ts=1000)
        result = manager.detect(inp)
        self.assertIn("detected", result)
        # Not checking detected value — just that it doesn't fail validation

    def test_negative_current_timestamp_returns_no_motion(self) -> None:
        manager = _make_manager()
        inp = _make_input(current_ts=-1, previous_ts=-2)
        self._assert_no_motion(manager.detect(inp))

    def test_negative_previous_timestamp_returns_no_motion(self) -> None:
        manager = _make_manager()
        inp = _make_input(current_ts=1000, previous_ts=-1)
        self._assert_no_motion(manager.detect(inp))

    def test_image_dimension_mismatch_returns_no_motion(self) -> None:
        manager = _make_manager()
        inp = MotionDetectionInput(
            current_frame=_make_frame(height=64, width=64),
            previous_frame=_make_frame(height=32, width=32),
        )
        self._assert_no_motion(manager.detect(inp))

    def test_wrong_dtype_current_image_returns_no_motion(self) -> None:
        manager = _make_manager()
        inp = _make_input()
        bad_image = Image(
            data=np.zeros((64, 64), dtype=np.float32),
            width=64, height=64,
            color_format="GRAY", layout="HWC",
            dtype="float32",
            value_range="[0,255]",
        )
        inp["current_frame"]["image"] = bad_image
        self._assert_no_motion(manager.detect(inp))

    def test_wrong_dtype_previous_image_returns_no_motion(self) -> None:
        manager = _make_manager()
        inp = _make_input()
        bad_image = Image(
            data=np.zeros((64, 64), dtype=np.float32),
            width=64, height=64,
            color_format="GRAY", layout="HWC",
            dtype="float32",
            value_range="[0,255]",
        )
        inp["previous_frame"]["image"] = bad_image
        self._assert_no_motion(manager.detect(inp))

    def test_wrong_color_format_returns_no_motion(self) -> None:
        manager = _make_manager()
        inp = _make_input()
        bad_image = Image(
            data=np.zeros((64, 64, 3), dtype=np.uint8),
            width=64, height=64,
            color_format="RGB",
            layout="HWC",
            dtype="uint8",
            value_range="[0,255]",
        )
        inp["current_frame"]["image"] = bad_image
        self._assert_no_motion(manager.detect(inp))

    def test_wrong_layout_returns_no_motion(self) -> None:
        manager = _make_manager()
        inp = _make_input()
        bad_image = Image(
            data=np.zeros((1, 64, 64), dtype=np.uint8),
            width=64, height=64,
            color_format="GRAY",
            layout="CHW",
            dtype="uint8",
            value_range="[0,255]",
        )
        inp["current_frame"]["image"] = bad_image
        self._assert_no_motion(manager.detect(inp))

    def test_wrong_value_range_returns_no_motion(self) -> None:
        manager = _make_manager()
        inp = _make_input()
        bad_image = Image(
            data=np.zeros((64, 64), dtype=np.uint8),
            width=64, height=64,
            color_format="GRAY", layout="HWC", dtype="uint8",
            value_range="[0,1]",
        )
        inp["current_frame"]["image"] = bad_image
        self._assert_no_motion(manager.detect(inp))

    def test_invalid_width_returns_no_motion(self) -> None:
        manager = _make_manager()
        inp = _make_input()
        bad_image = Image(
            data=np.zeros((64, 64), dtype=np.uint8),
            width=0,
            height=64,
            color_format="GRAY", layout="HWC", dtype="uint8",
            value_range="[0,255]",
        )
        inp["current_frame"]["image"] = bad_image
        self._assert_no_motion(manager.detect(inp))

    def test_invalid_height_returns_no_motion(self) -> None:
        manager = _make_manager()
        inp = _make_input()
        bad_image = Image(
            data=np.zeros((64, 64), dtype=np.uint8),
            width=64,
            height=0,
            color_format="GRAY", layout="HWC", dtype="uint8",
            value_range="[0,255]",
        )
        inp["current_frame"]["image"] = bad_image
        self._assert_no_motion(manager.detect(inp))

    def test_data_shape_mismatch_returns_no_motion(self) -> None:
        """Declared width/height don't match data.shape."""
        manager = _make_manager()
        inp = _make_input()
        bad_image = Image(
            data=np.zeros((32, 32), dtype=np.uint8),
            width=64,
            height=64,
            color_format="GRAY", layout="HWC", dtype="uint8",
            value_range="[0,255]",
        )
        inp["current_frame"]["image"] = bad_image
        self._assert_no_motion(manager.detect(inp))

    def test_raw_ndarray_as_image_returns_no_motion(self) -> None:
        """Passing a raw np.ndarray instead of an Image struct must fail."""
        manager = _make_manager()
        inp = _make_input()
        inp["current_frame"]["image"] = np.zeros((64, 64), dtype=np.uint8)  # type: ignore[typeddict-item]
        self._assert_no_motion(manager.detect(inp))

    def test_none_current_image_returns_no_motion(self) -> None:
        manager = _make_manager()
        inp = _make_input()
        inp["current_frame"]["image"] = None  # type: ignore[typeddict-item]
        self._assert_no_motion(manager.detect(inp))

    def test_multichannel_non_gray_image_returns_no_motion(self) -> None:
        """Image with color_format=RGB (not GRAY) must fail validation."""
        manager = _make_manager()
        inp = _make_input()
        bad_image = Image(
            data=np.zeros((64, 64, 3), dtype=np.uint8),
            width=64, height=64,
            color_format="RGB",
            layout="HWC",
            dtype="uint8",
            value_range="[0,255]",
        )
        inp["current_frame"]["image"] = bad_image
        self._assert_no_motion(manager.detect(inp))


# ---------------------------------------------------------------------------
# StubPipelineTests — spec §5 (stub behavior) + §7 (decision policy)
# ---------------------------------------------------------------------------


class StubPipelineTests(unittest.TestCase):
    """Verify stub produces correct deterministic output through the pipeline."""

    def test_valid_input_returns_detected_true_at_default_threshold(self) -> None:
        # Default threshold 0.05; stub fraction 0.2 => detected=True
        manager = _make_stub_manager(motion_fraction_threshold=0.05)
        result = manager.detect(_make_input())
        self.assertTrue(result["detected"])

    def test_valid_input_returns_non_empty_bboxes_when_detected(self) -> None:
        manager = _make_stub_manager(motion_fraction_threshold=0.05)
        result = manager.detect(_make_input())
        self.assertGreater(len(result["bboxes"]), 0)

    def test_bbox_has_strictly_positive_dimensions(self) -> None:
        manager = _make_stub_manager()
        result = manager.detect(_make_input())
        for bbox in result["bboxes"]:
            self.assertGreater(bbox["width"], 0)
            self.assertGreater(bbox["height"], 0)

    def test_bbox_is_inside_current_image_bounds(self) -> None:
        h, w = 64, 80
        manager = _make_stub_manager()
        result = manager.detect(_make_input(height=h, width=w))
        for bbox in result["bboxes"]:
            self.assertGreaterEqual(bbox["x"], 0)
            self.assertGreaterEqual(bbox["y"], 0)
            self.assertLessEqual(bbox["x"] + bbox["width"], w)
            self.assertLessEqual(bbox["y"] + bbox["height"], h)

    def test_bbox_inside_bounds_for_minimal_image(self) -> None:
        """Even a 1×1 image must produce a valid in-bounds bbox."""
        manager = _make_stub_manager()
        result = manager.detect(_make_input(height=1, width=1))
        for bbox in result["bboxes"]:
            self.assertGreaterEqual(bbox["x"], 0)
            self.assertGreaterEqual(bbox["y"], 0)
            self.assertLessEqual(bbox["x"] + bbox["width"], 1)
            self.assertLessEqual(bbox["y"] + bbox["height"], 1)
            self.assertGreater(bbox["width"], 0)
            self.assertGreater(bbox["height"], 0)

    def test_detected_false_when_threshold_exceeds_stub_fraction(self) -> None:
        # Stub fraction is 0.2; threshold 0.5 => not detected
        manager = _make_stub_manager(motion_fraction_threshold=0.5)
        result = manager.detect(_make_input())
        self.assertFalse(result["detected"])

    def test_bboxes_empty_when_not_detected(self) -> None:
        manager = _make_stub_manager(motion_fraction_threshold=0.5)
        result = manager.detect(_make_input())
        self.assertEqual(result["bboxes"], [])

    def test_detected_true_at_exact_threshold(self) -> None:
        # Stub fraction = 0.2; threshold = 0.2 => motion_fraction >= threshold
        manager = _make_stub_manager(motion_fraction_threshold=0.2)
        result = manager.detect(_make_input())
        self.assertTrue(result["detected"])

    def test_result_is_deterministic(self) -> None:
        manager = _make_stub_manager()
        inp = _make_input()
        result1 = manager.detect(inp)
        result2 = manager.detect(inp)
        self.assertEqual(result1, result2)

    def test_result_with_single_channel_3d_image(self) -> None:
        """(H, W, 1) grayscale images must be accepted and processed."""
        manager = _make_stub_manager()
        inp = MotionDetectionInput(
            current_frame=_make_frame(image=_make_image_3d(64, 64)),
            previous_frame=_make_frame(image=_make_image_3d(64, 64)),
        )
        result = manager.detect(inp)
        self.assertIn("detected", result)
        self.assertIn("bboxes", result)


# ---------------------------------------------------------------------------
# ManagerBehaviorTests — spec §8.1 / §11
# ---------------------------------------------------------------------------


class ManagerBehaviorTests(unittest.TestCase):
    """Manager orchestration and error-handling contract."""

    def test_output_has_exactly_detected_and_bboxes_keys(self) -> None:
        manager = _make_manager()
        result = manager.detect(_make_input())
        self.assertEqual(set(result), {"detected", "bboxes"})

    def test_manager_does_not_mutate_input_dict(self) -> None:
        manager = _make_manager()
        inp = _make_input()

        # Snapshot the identity of inner objects
        current_id = id(inp["current_frame"])
        previous_id = id(inp["previous_frame"])
        current_img_id = id(inp["current_frame"]["image"])
        previous_img_id = id(inp["previous_frame"]["image"])

        manager.detect(inp)

        self.assertEqual(id(inp["current_frame"]), current_id)
        self.assertEqual(id(inp["previous_frame"]), previous_id)
        self.assertEqual(id(inp["current_frame"]["image"]), current_img_id)
        self.assertEqual(id(inp["previous_frame"]["image"]), previous_img_id)

    def test_algorithm_exception_returns_no_motion(self) -> None:
        class ExplodingAlgorithm:
            def measure(self, previous_image: np.ndarray, current_image: np.ndarray) -> MotionMeasurementResult:
                raise RuntimeError("simulated algorithm failure")

        manager = _make_manager(algorithm=ExplodingAlgorithm())  # type: ignore[arg-type]
        result = manager.detect(_make_input())
        self.assertFalse(result["detected"])
        self.assertEqual(result["bboxes"], [])

    def test_non_mapping_input_falls_back_without_secondary_exception(self) -> None:
        manager = _make_manager()
        result = manager.detect(None)  # type: ignore[arg-type]
        self.assertFalse(result["detected"])
        self.assertEqual(result["bboxes"], [])
        debug_info = manager.get_last_debug_info()
        self.assertEqual(debug_info.get("failure_type"), "runtime_error")

    def test_runtime_failure_populates_debug_info_and_clears_debug_images(self) -> None:
        manager = _make_manager(debug_capture_enabled=True, debug_max_images=4)
        manager.detect(_make_input())
        self.assertGreater(len(manager.get_last_debug_images()), 0)

        class ExplodingAlgorithm:
            def measure(self, previous_image: np.ndarray, current_image: np.ndarray) -> MotionMeasurementResult:
                raise RuntimeError("simulated runtime failure")

        manager._algorithm = ExplodingAlgorithm()  # type: ignore[assignment]
        result = manager.detect(_make_input())

        self.assertFalse(result["detected"])
        self.assertEqual(result["bboxes"], [])
        self.assertEqual(manager.get_last_debug_images(), {})

        debug_info = manager.get_last_debug_info()
        self.assertEqual(debug_info.get("failure_type"), "runtime_error")
        self.assertEqual(debug_info.get("error_type"), "RuntimeError")
        self.assertIn("simulated runtime failure", debug_info.get("error", ""))

    def test_validation_failure_populates_debug_info_and_clears_debug_images(self) -> None:
        manager = _make_manager(debug_capture_enabled=True, debug_max_images=4)
        manager.detect(_make_input())
        self.assertGreater(len(manager.get_last_debug_images()), 0)

        invalid_input = _make_input()
        invalid_input["current_frame"]["camera_id"] = ""

        result = manager.detect(invalid_input)

        self.assertFalse(result["detected"])
        self.assertEqual(result["bboxes"], [])
        self.assertEqual(manager.get_last_debug_images(), {})

        debug_info = manager.get_last_debug_info()
        self.assertEqual(debug_info.get("failure_type"), "validation_error")
        self.assertIn("camera_id", debug_info.get("error", ""))

    def test_debug_images_disabled_by_default(self) -> None:
        manager = _make_manager()
        manager.detect(_make_input())
        self.assertEqual(manager.get_last_debug_images(), {})

    def test_debug_images_bounded_when_enabled(self) -> None:
        manager = _make_manager(debug_capture_enabled=True, debug_max_images=2)
        manager.detect(_make_input())
        debug_images = manager.get_last_debug_images()
        self.assertGreater(len(debug_images), 0)
        self.assertLessEqual(len(debug_images), 2)

    def test_manager_does_not_recopy_algorithm_debug_images(self) -> None:
        debug_image = np.zeros((8, 8), dtype=np.uint8)

        class StaticDebugAlgorithm:
            def measure(self, previous_image: np.ndarray, current_image: np.ndarray) -> MotionMeasurementResult:
                return MotionMeasurementResult(
                    motion_fraction=0.2,
                    bboxes=[BoundingBox(x=1, y=1, width=4, height=4)],
                    debug_images={"k": debug_image},
                )

        manager = _make_manager(algorithm=StaticDebugAlgorithm())  # type: ignore[arg-type]
        manager.detect(_make_input())
        self.assertIs(manager._last_debug_images["k"], debug_image)

    def test_detected_is_bool(self) -> None:
        manager = _make_manager()
        result = manager.detect(_make_input())
        self.assertIsInstance(result["detected"], bool)

    def test_bboxes_is_list(self) -> None:
        manager = _make_manager()
        result = manager.detect(_make_input())
        self.assertIsInstance(result["bboxes"], list)

    def test_bbox_fields_are_correct_types(self) -> None:
        manager = _make_stub_manager()
        result = manager.detect(_make_input())
        self.assertTrue(result["detected"])
        bbox = result["bboxes"][0]
        for field_name in ("x", "y", "width", "height"):
            self.assertIsInstance(bbox[field_name], int, msg=f"{field_name} must be int")

    def test_process_wrapper_calls_detect(self) -> None:
        """process() deprecated wrapper must return same result as detect()."""
        manager = _make_stub_manager()
        inp = _make_input()
        self.assertEqual(manager.process(inp), manager.detect(inp))


# ---------------------------------------------------------------------------
# DeterminismTests — spec §5.5 (non-functional: deterministic)
# ---------------------------------------------------------------------------


class DeterminismTests(unittest.TestCase):
    """Same input + same config must always produce identical output."""

    def test_same_input_same_config_produces_identical_output(self) -> None:
        config = MotionDetectionConfig(
            motion_threshold=25,
            motion_fraction_threshold=0.05,
            min_bbox_area=100,
        )
        manager1 = MotionDetectionManager(config=config)
        manager2 = MotionDetectionManager(config=config)
        inp = _make_input()
        self.assertEqual(manager1.detect(inp), manager2.detect(inp))

    def test_multiple_invocations_produce_identical_output(self) -> None:
        manager = _make_manager()
        inp = _make_input()
        results = [manager.detect(inp) for _ in range(5)]
        self.assertTrue(all(r == results[0] for r in results))

    def test_different_image_sizes_produce_in_bounds_bbox(self) -> None:
        """Verify deterministic bbox stays in bounds for various sizes."""
        manager = _make_manager()
        for h, w in [(1, 1), (4, 4), (10, 20), (100, 200), (480, 640)]:
            with self.subTest(h=h, w=w):
                result = manager.detect(_make_input(height=h, width=w))
                if result["detected"]:
                    bbox = result["bboxes"][0]
                    self.assertGreaterEqual(bbox["x"], 0)
                    self.assertGreaterEqual(bbox["y"], 0)
                    self.assertLessEqual(bbox["x"] + bbox["width"], w)
                    self.assertLessEqual(bbox["y"] + bbox["height"], h)
                    self.assertGreater(bbox["width"], 0)
                    self.assertGreater(bbox["height"], 0)


# ---------------------------------------------------------------------------
# GetInputContractTests — spec §4 / §9
# ---------------------------------------------------------------------------


class GetInputContractTests(unittest.TestCase):
    """Verify get_input_contract() returns the correct values."""

    def test_returns_grayscale_uint8_hwc(self) -> None:
        manager = MotionDetectionManager()
        contract = manager.get_input_contract()
        self.assertEqual(contract["output_image_type"], OutputImageType.GRAYSCALE_UINT8_HWC)

    def test_returns_resize_policy_none(self) -> None:
        manager = MotionDetectionManager()
        contract = manager.get_input_contract()
        self.assertEqual(contract["geometry_spec"]["resize_policy"], ResizePolicy.NONE)

    def test_geometry_spec_width_height_zero(self) -> None:
        manager = MotionDetectionManager()
        contract = manager.get_input_contract()
        geo = contract["geometry_spec"]
        self.assertEqual(geo["width"], 0)
        self.assertEqual(geo["height"], 0)

    def test_contract_has_required_keys(self) -> None:
        manager = MotionDetectionManager()
        contract = manager.get_input_contract()
        self.assertIn("output_image_type", contract)
        self.assertIn("geometry_spec", contract)


# ---------------------------------------------------------------------------
# ProtocolComplianceTests — spec §8.1
# ---------------------------------------------------------------------------


class ProtocolComplianceTests(unittest.TestCase):
    """Verify MotionDetectionManager satisfies MotionDetectionInterface."""

    def test_manager_satisfies_interface(self) -> None:
        manager = MotionDetectionManager()
        self.assertIsInstance(manager, MotionDetectionInterface)

    def test_manager_has_detect_method(self) -> None:
        manager = MotionDetectionManager()
        self.assertTrue(callable(getattr(manager, "detect", None)))

    def test_manager_has_get_input_contract_method(self) -> None:
        manager = MotionDetectionManager()
        self.assertTrue(callable(getattr(manager, "get_input_contract", None)))


# ---------------------------------------------------------------------------
# RealAlgorithmTests — spec §6.2 (FrameDifferencingMotionDetector)
# ---------------------------------------------------------------------------


class RealAlgorithmTests(unittest.TestCase):
    """Verify FrameDifferencingMotionDetector real algorithm behavior."""

    ASSETS_ROOT = Path(__file__).parent / "assets" / "frames"

    # --- helpers -----------------------------------------------------------

    def _real_manager(
        self,
        motion_threshold: int = 25,
        motion_fraction_threshold: float = 0.05,
        min_bbox_area: int = 100,
    ) -> MotionDetectionManager:
        config = MotionDetectionConfig(
            motion_threshold=motion_threshold,
            motion_fraction_threshold=motion_fraction_threshold,
            min_bbox_area=min_bbox_area,
        )
        return MotionDetectionManager(config=config)

    def _motion_input(
        self,
        prev_img: np.ndarray,
        curr_img: np.ndarray,
    ) -> MotionDetectionInput:
        h, w = curr_img.shape[0], curr_img.shape[1]
        ph, pw = prev_img.shape[0], prev_img.shape[1]
        curr_image = Image(
            data=curr_img, width=w, height=h,
            color_format="GRAY", layout="HWC", dtype="uint8", value_range="[0,255]",
        )
        prev_image = Image(
            data=prev_img, width=pw, height=ph,
            color_format="GRAY", layout="HWC", dtype="uint8", value_range="[0,255]",
        )
        return MotionDetectionInput(
            current_frame=MotionInputFrame(
                frame_id="frame_0002", camera_id="cam-a",
                timestamp_ms=2000, image=curr_image,
            ),
            previous_frame=MotionInputFrame(
                frame_id="frame_0001", camera_id="cam-a",
                timestamp_ms=1000, image=prev_image,
            ),
        )

    # --- identical frames --------------------------------------------------

    def test_identical_frames_produce_zero_motion_fraction(self) -> None:
        algo = FrameDifferencingMotionDetector(motion_threshold=25, min_bbox_area=100)
        img = np.zeros((64, 64), dtype=np.uint8)
        result = algo.measure(img, img)
        self.assertEqual(result.motion_fraction, 0.0)

    def test_identical_frames_not_detected(self) -> None:
        manager = self._real_manager()
        img = np.zeros((64, 64), dtype=np.uint8)
        result = manager.detect(self._motion_input(img, img))
        self.assertFalse(result["detected"])
        self.assertEqual(result["bboxes"], [])

    # --- synthetic changed region ------------------------------------------

    def test_synthetic_changed_region_returns_bbox(self) -> None:
        """A clearly changed 20×20 block must produce at least one bbox."""
        h, w = 64, 64
        prev_img = np.zeros((h, w), dtype=np.uint8)
        curr_img = np.zeros((h, w), dtype=np.uint8)
        curr_img[20:40, 20:40] = 255  # 20×20 = area 400 >= min_bbox_area=100

        algo = FrameDifferencingMotionDetector(motion_threshold=25, min_bbox_area=100)
        result = algo.measure(prev_img, curr_img)
        self.assertGreater(len(result.bboxes), 0)

    def test_bbox_stays_inside_image_bounds(self) -> None:
        h, w = 64, 64
        prev_img = np.zeros((h, w), dtype=np.uint8)
        curr_img = np.zeros((h, w), dtype=np.uint8)
        curr_img[10:50, 10:50] = 200

        algo = FrameDifferencingMotionDetector(motion_threshold=25, min_bbox_area=100)
        result = algo.measure(prev_img, curr_img)
        for bbox in result.bboxes:
            self.assertGreaterEqual(bbox["x"], 0)
            self.assertGreaterEqual(bbox["y"], 0)
            self.assertLessEqual(bbox["x"] + bbox["width"], w)
            self.assertLessEqual(bbox["y"] + bbox["height"], h)
            self.assertGreater(bbox["width"], 0)
            self.assertGreater(bbox["height"], 0)

    # --- min_bbox_area filtering -------------------------------------------

    def test_small_region_below_min_bbox_area_filtered_by_algorithm(self) -> None:
        """A 3×3 changed block (area=9) must be filtered out by min_bbox_area=100."""
        h, w = 64, 64
        prev_img = np.zeros((h, w), dtype=np.uint8)
        curr_img = np.zeros((h, w), dtype=np.uint8)
        curr_img[30:33, 30:33] = 255  # area 9 < min_bbox_area=100

        algo = FrameDifferencingMotionDetector(motion_threshold=25, min_bbox_area=100)
        result = algo.measure(prev_img, curr_img)
        self.assertEqual(result.bboxes, [])

    def test_small_region_produces_no_detection_via_manager(self) -> None:
        h, w = 64, 64
        prev_img = np.zeros((h, w), dtype=np.uint8)
        curr_img = np.zeros((h, w), dtype=np.uint8)
        curr_img[30:33, 30:33] = 255

        manager = self._real_manager(min_bbox_area=100)
        result = manager.detect(self._motion_input(prev_img, curr_img))
        self.assertFalse(result["detected"])

    # --- motion_fraction_threshold -----------------------------------------

    def test_low_motion_fraction_below_threshold_not_detected(self) -> None:
        """A 15×15 block in a 200×200 image: fraction ~0.0056 < 0.05."""
        h, w = 200, 200
        prev_img = np.zeros((h, w), dtype=np.uint8)
        curr_img = np.zeros((h, w), dtype=np.uint8)
        curr_img[90:105, 90:105] = 255  # area 225 >= min_bbox_area=100

        manager = self._real_manager(motion_fraction_threshold=0.05, min_bbox_area=100)
        result = manager.detect(self._motion_input(prev_img, curr_img))
        self.assertFalse(result["detected"])

    def test_high_motion_fraction_above_threshold_detected(self) -> None:
        """A 50×50 block in a 100×100 image: fraction 0.25 >= 0.05."""
        h, w = 100, 100
        prev_img = np.zeros((h, w), dtype=np.uint8)
        curr_img = np.zeros((h, w), dtype=np.uint8)
        curr_img[25:75, 25:75] = 255  # area 2500, fraction 0.25

        manager = self._real_manager(motion_fraction_threshold=0.05, min_bbox_area=100)
        result = manager.detect(self._motion_input(prev_img, curr_img))
        self.assertTrue(result["detected"])
        self.assertGreater(len(result["bboxes"]), 0)

    # --- direct fraction math ----------------------------------------------

    def test_algorithm_motion_fraction_is_correct(self) -> None:
        """Direct test: 100 changed pixels in 10000 total => fraction = 0.01."""
        h, w = 100, 100  # 10 000 pixels
        prev_img = np.zeros((h, w), dtype=np.uint8)
        curr_img = np.zeros((h, w), dtype=np.uint8)
        curr_img[0:10, 0:10] = 255  # exactly 100 pixels changed

        algo = FrameDifferencingMotionDetector(motion_threshold=25, min_bbox_area=1)
        result = algo.measure(prev_img, curr_img)
        self.assertAlmostEqual(result.motion_fraction, 0.01, places=5)

    def test_threshold_mask_is_reused_without_duplicate_thresholding(self) -> None:
        prev_img = np.zeros((64, 64), dtype=np.uint8)
        curr_img = np.zeros((64, 64), dtype=np.uint8)
        curr_img[20:40, 20:40] = 255

        detector = FrameDifferencingMotionDetector(
            motion_threshold=25,
            min_bbox_area=100,
            enable_global_motion_compensation=False,
        )

        with mock.patch.object(cv2, "threshold", wraps=cv2.threshold) as threshold_mock:
            detector.measure(prev_img, curr_img)

        self.assertEqual(threshold_mock.call_count, 1)

    def test_morphology_kernel_created_once_and_reused(self) -> None:
        prev_img = np.zeros((64, 64), dtype=np.uint8)
        curr_img = np.zeros((64, 64), dtype=np.uint8)
        curr_img[20:40, 20:40] = 255

        detector = FrameDifferencingMotionDetector(
            motion_threshold=25,
            min_bbox_area=100,
            morph_open_iterations=1,
        )

        self.assertTrue(hasattr(detector, "_morphology_kernel"))
        kernel_id = id(detector._morphology_kernel)
        detector.measure(prev_img, curr_img)
        detector.measure(prev_img, curr_img)
        self.assertEqual(id(detector._morphology_kernel), kernel_id)

    # --- (H, W, 1) 3-D input -----------------------------------------------

    def test_single_channel_3d_input_handled(self) -> None:
        """(H, W, 1) inputs must be processed correctly without error."""
        h, w = 64, 64
        prev_img = np.zeros((h, w, 1), dtype=np.uint8)
        curr_img = np.zeros((h, w, 1), dtype=np.uint8)
        curr_img[20:50, 20:50] = 255  # area 900 >= min_bbox_area=100

        algo = FrameDifferencingMotionDetector(motion_threshold=25, min_bbox_area=100)
        result = algo.measure(prev_img, curr_img)
        self.assertGreater(len(result.bboxes), 0)

    def test_3d_image_struct_accepted_by_manager(self) -> None:
        """(H,W,1) Image struct must pass validation and reach the algorithm."""
        h, w = 64, 64
        prev_arr = np.zeros((h, w, 1), dtype=np.uint8)
        curr_arr = np.zeros((h, w, 1), dtype=np.uint8)
        curr_arr[10:50, 10:50] = 255
        manager = self._real_manager(motion_fraction_threshold=0.05, min_bbox_area=100)
        result = manager.detect(self._motion_input(prev_arr, curr_arr))
        self.assertIn("detected", result)

    def test_morphology_open_removes_single_pixel_noise(self) -> None:
        """Sparse salt noise should be removed before contour extraction."""
        h, w = 128, 128
        prev_img = np.zeros((h, w), dtype=np.uint8)
        curr_img = np.zeros((h, w), dtype=np.uint8)

        # Place isolated points with at least one empty pixel around each point.
        ys = np.arange(8, h - 8, 8)
        xs = np.arange(8, w - 8, 8)
        for y in ys:
            for x in xs:
                curr_img[y, x] = 255

        manager = MotionDetectionManager(config=MotionDetectionConfig(
            motion_threshold=25,
            motion_fraction_threshold=0.0001,
            min_bbox_area=1,
            blur_kernel_size=3,
            morph_open_iterations=1,
            morph_close_iterations=0,
            dilation_iterations=0,
        ))
        result = manager.detect(self._motion_input(prev_img, curr_img))
        self.assertFalse(result["detected"])
        self.assertEqual(result["bboxes"], [])


class GlobalMotionCompensationTests(unittest.TestCase):
    def test_global_motion_compensation_suppresses_shifted_static_scene(self) -> None:
        prev_img = _make_textured_scene(seed=11)
        curr_img = _shift_image(prev_img, dx=8, dy=5)

        manager = MotionDetectionManager(config=MotionDetectionConfig(
            motion_threshold=25,
            motion_fraction_threshold=0.02,
            min_bbox_area=50,
            enable_global_motion_compensation=True,
            max_features=300,
            min_feature_matches=20,
            max_transform_shift=20.0,
            global_motion_changed_ratio_threshold=0.20,
            fallback_on_alignment_failure=True,
        ))

        result = manager.detect(_make_motion_input_from_arrays(prev_img, curr_img))
        debug = manager.get_last_debug_info()

        self.assertFalse(result["detected"])
        self.assertTrue(debug["enable_global_motion_compensation"])
        self.assertTrue(debug["alignment_succeeded"])
        self.assertGreaterEqual(debug["feature_match_count"], 20)
        self.assertLess(debug["changed_pixel_ratio_after_alignment"], debug["changed_pixel_ratio_before_alignment"])

    def test_global_motion_compensation_disabled_detects_large_shift_motion(self) -> None:
        prev_img = _make_textured_scene(seed=12)
        curr_img = _shift_image(prev_img, dx=8, dy=5)

        manager = MotionDetectionManager(config=MotionDetectionConfig(
            motion_threshold=25,
            motion_fraction_threshold=0.02,
            min_bbox_area=50,
            enable_global_motion_compensation=False,
        ))

        result = manager.detect(_make_motion_input_from_arrays(prev_img, curr_img))
        debug = manager.get_last_debug_info()

        self.assertTrue(result["detected"])
        self.assertFalse(debug["enable_global_motion_compensation"])
        self.assertFalse(debug["alignment_succeeded"])
        self.assertGreater(debug["changed_pixel_ratio_before_alignment"], 0.02)

    def test_alignment_failure_follows_fallback_configuration(self) -> None:
        prev_img = np.tile(np.linspace(120, 136, 256, dtype=np.uint8), (192, 1))
        transform = np.array([[1.0, 0.0, 6.0], [0.0, 1.0, 4.0]], dtype=np.float32)
        curr_img = cv2.warpAffine(
            prev_img,
            transform,
            (prev_img.shape[1], prev_img.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        manager = MotionDetectionManager(config=MotionDetectionConfig(
            motion_threshold=10,
            motion_fraction_threshold=0.001,
            min_bbox_area=1,
            enable_global_motion_compensation=True,
            max_features=50,
            min_feature_matches=12,
            max_transform_shift=15.0,
            global_motion_changed_ratio_threshold=0.05,
            fallback_on_alignment_failure=True,
        ))

        result = manager.detect(_make_motion_input_from_arrays(prev_img, curr_img))
        debug = manager.get_last_debug_info()

        self.assertFalse(debug["alignment_succeeded"])
        self.assertTrue(debug["alignment_fallback_used"])
        self.assertTrue(result["detected"])

    def test_alignment_preserves_local_motion_after_global_compensation(self) -> None:
        prev_img = _make_textured_scene(seed=13)
        curr_img = _shift_image(prev_img, dx=7, dy=4)
        cv2.rectangle(curr_img, (100, 70), (138, 124), 255, -1)

        manager = MotionDetectionManager(config=MotionDetectionConfig(
            motion_threshold=20,
            motion_fraction_threshold=0.01,
            min_bbox_area=150,
            enable_global_motion_compensation=True,
            max_features=300,
            min_feature_matches=20,
            max_transform_shift=20.0,
            global_motion_changed_ratio_threshold=0.20,
            fallback_on_alignment_failure=True,
            morph_close_iterations=1,
        ))

        result = manager.detect(_make_motion_input_from_arrays(prev_img, curr_img))
        debug = manager.get_last_debug_info()

        self.assertTrue(result["detected"])
        self.assertTrue(debug["alignment_succeeded"])
        self.assertGreater(debug["changed_pixel_ratio_after_alignment"], 0.0)
        self.assertTrue(any(bbox["x"] <= 120 <= bbox["x"] + bbox["width"] and bbox["y"] <= 97 <= bbox["y"] + bbox["height"] for bbox in result["bboxes"]))

    def test_camera_shake_detected_for_excessive_global_shift(self) -> None:
        prev_img = _make_textured_scene(seed=14)
        curr_img = _shift_image(prev_img, dx=32, dy=24)

        manager = MotionDetectionManager(config=MotionDetectionConfig(
            motion_threshold=25,
            motion_fraction_threshold=0.02,
            min_bbox_area=50,
            enable_global_motion_compensation=True,
            max_features=300,
            min_feature_matches=20,
            max_transform_shift=10.0,
            global_motion_changed_ratio_threshold=0.20,
            fallback_on_alignment_failure=False,
        ))

        result = manager.detect(_make_motion_input_from_arrays(prev_img, curr_img))
        debug = manager.get_last_debug_info()

        self.assertFalse(result["detected"])
        self.assertTrue(debug["camera_shake_detected"])
        self.assertGreater(debug["estimated_global_shift"], 10.0)
        self.assertIsNotNone(debug["estimated_transform_matrix"])


class _StaticMeasurementAlgorithm:
    def __init__(self, bboxes: list[BoundingBox], motion_fraction: float = 0.2) -> None:
        self._bboxes = list(bboxes)
        self._motion_fraction = motion_fraction

    def measure(self, previous_image: np.ndarray, current_image: np.ndarray) -> MotionMeasurementResult:
        return MotionMeasurementResult(
            motion_fraction=self._motion_fraction,
            bboxes=list(self._bboxes),
            debug_info={
                "motion_bboxes_raw_count": len(self._bboxes),
                "motion_bboxes_after_filter_count": len(self._bboxes),
            },
        )


class BboxMergingAndTemporalPersistenceTests(unittest.TestCase):
    def _manager_for_test(
        self,
        *,
        bboxes: list[BoundingBox],
        enable_bbox_merging: bool = True,
        bbox_merge_iou_threshold: float = 0.2,
        bbox_merge_distance_threshold: float = 8.0,
        enable_temporal_persistence: bool = False,
        min_persistence_frames: int = 2,
        persistence_iou_threshold: float = 0.3,
        max_history_frames: int = 4,
        camera_state_ttl_ms: int = 0,
    ) -> MotionDetectionManager:
        cfg = MotionDetectionConfig(
            motion_threshold=25,
            motion_fraction_threshold=0.05,
            min_bbox_area=1,
            enable_bbox_merging=enable_bbox_merging,
            bbox_merge_iou_threshold=bbox_merge_iou_threshold,
            bbox_merge_distance_threshold=bbox_merge_distance_threshold,
            enable_temporal_persistence=enable_temporal_persistence,
            min_persistence_frames=min_persistence_frames,
            persistence_iou_threshold=persistence_iou_threshold,
            max_history_frames=max_history_frames,
            camera_state_ttl_ms=camera_state_ttl_ms,
        )
        algo = _StaticMeasurementAlgorithm(bboxes=bboxes)
        return MotionDetectionManager(config=cfg, algorithm=algo)

    def test_overlapping_bboxes_merge_correctly(self) -> None:
        manager = self._manager_for_test(
            bboxes=[
                BoundingBox(x=10, y=10, width=20, height=20),
                BoundingBox(x=18, y=18, width=20, height=20),
            ],
            bbox_merge_iou_threshold=0.05,
        )
        result = manager.detect(_make_input(height=80, width=80))
        self.assertTrue(result["detected"])
        self.assertEqual(len(result["bboxes"]), 1)
        merged = result["bboxes"][0]
        self.assertEqual(merged["x"], 10)
        self.assertEqual(merged["y"], 10)
        self.assertEqual(merged["width"], 28)
        self.assertEqual(merged["height"], 28)

    def test_nearby_bboxes_merge_correctly(self) -> None:
        manager = self._manager_for_test(
            bboxes=[
                BoundingBox(x=5, y=10, width=10, height=12),
                BoundingBox(x=18, y=10, width=9, height=12),
            ],
            bbox_merge_iou_threshold=0.5,
            bbox_merge_distance_threshold=4.0,
        )
        result = manager.detect(_make_input(height=80, width=80))
        self.assertEqual(len(result["bboxes"]), 1)
        merged = result["bboxes"][0]
        self.assertEqual(merged["x"], 5)
        self.assertEqual(merged["width"], 22)

    def test_distant_bboxes_stay_separate(self) -> None:
        manager = self._manager_for_test(
            bboxes=[
                BoundingBox(x=5, y=5, width=10, height=10),
                BoundingBox(x=40, y=40, width=10, height=10),
            ],
            bbox_merge_distance_threshold=3.0,
        )
        result = manager.detect(_make_input(height=80, width=80))
        self.assertEqual(len(result["bboxes"]), 2)

    def test_merged_bboxes_stay_inside_image_bounds(self) -> None:
        manager = self._manager_for_test(
            bboxes=[
                BoundingBox(x=-5, y=-5, width=20, height=20),
                BoundingBox(x=12, y=12, width=80, height=80),
            ],
            bbox_merge_distance_threshold=10.0,
        )
        result = manager.detect(_make_input(height=64, width=64))
        self.assertEqual(len(result["bboxes"]), 1)
        bbox = result["bboxes"][0]
        self.assertGreaterEqual(bbox["x"], 0)
        self.assertGreaterEqual(bbox["y"], 0)
        self.assertLessEqual(bbox["x"] + bbox["width"], 64)
        self.assertLessEqual(bbox["y"] + bbox["height"], 64)

    def test_one_frame_noise_is_suppressed(self) -> None:
        manager = self._manager_for_test(
            bboxes=[BoundingBox(x=10, y=10, width=12, height=12)],
            enable_temporal_persistence=True,
            min_persistence_frames=2,
        )
        result = manager.detect(_make_input(camera_id="cam-noise"))
        self.assertFalse(result["detected"])
        self.assertEqual(result["bboxes"], [])

    def test_persistent_motion_across_frames_is_emitted(self) -> None:
        manager = self._manager_for_test(
            bboxes=[BoundingBox(x=10, y=10, width=12, height=12)],
            enable_temporal_persistence=True,
            min_persistence_frames=2,
            persistence_iou_threshold=0.2,
        )
        first = manager.detect(_make_input(camera_id="cam-persist"))
        second = manager.detect(_make_input(camera_id="cam-persist"))
        self.assertFalse(first["detected"])
        self.assertTrue(second["detected"])
        self.assertEqual(len(second["bboxes"]), 1)

    def test_different_cameras_maintain_isolated_histories(self) -> None:
        manager = self._manager_for_test(
            bboxes=[BoundingBox(x=10, y=10, width=12, height=12)],
            enable_temporal_persistence=True,
            min_persistence_frames=2,
        )
        cam_a_first = manager.detect(_make_input(camera_id="cam-A"))
        cam_b_first = manager.detect(_make_input(camera_id="cam-B"))
        cam_a_second = manager.detect(_make_input(camera_id="cam-A"))
        self.assertFalse(cam_a_first["detected"])
        self.assertFalse(cam_b_first["detected"])
        self.assertTrue(cam_a_second["detected"])

    def test_inactive_camera_state_is_pruned_by_ttl(self) -> None:
        manager = self._manager_for_test(
            bboxes=[BoundingBox(x=10, y=10, width=12, height=12)],
            enable_temporal_persistence=True,
            min_persistence_frames=2,
            camera_state_ttl_ms=500,
        )
        manager.detect(_make_input(camera_id="cam-old", current_ts=1000, previous_ts=900))
        self.assertIn("cam-old", manager._motion_history_by_camera_id)

        manager.detect(_make_input(camera_id="cam-new", current_ts=2000, previous_ts=1900))
        self.assertNotIn("cam-old", manager._motion_history_by_camera_id)
        self.assertIn("cam-new", manager._motion_history_by_camera_id)

    def test_disappearing_motion_is_removed_from_history(self) -> None:
        persistent_manager = self._manager_for_test(
            bboxes=[BoundingBox(x=10, y=10, width=12, height=12)],
            enable_temporal_persistence=True,
            min_persistence_frames=2,
        )
        persistent_manager.detect(_make_input(camera_id="cam-drop"))
        persistent_manager.detect(_make_input(camera_id="cam-drop"))

        class EmptyAlgorithm:
            def measure(self, previous_image: np.ndarray, current_image: np.ndarray) -> MotionMeasurementResult:
                return MotionMeasurementResult(motion_fraction=0.2, bboxes=[])

        persistent_manager._algorithm = EmptyAlgorithm()  # type: ignore[assignment]
        dropped = persistent_manager.detect(_make_input(camera_id="cam-drop"))
        self.assertFalse(dropped["detected"])
        self.assertEqual(dropped["bboxes"], [])

    def test_iou_based_persistence_matching_works(self) -> None:
        cfg = MotionDetectionConfig(
            motion_threshold=25,
            motion_fraction_threshold=0.05,
            min_bbox_area=1,
            enable_bbox_merging=False,
            enable_temporal_persistence=True,
            min_persistence_frames=2,
            persistence_iou_threshold=0.3,
            max_history_frames=4,
        )

        class SequenceAlgorithm:
            def __init__(self) -> None:
                self._index = 0

            def measure(self, previous_image: np.ndarray, current_image: np.ndarray) -> MotionMeasurementResult:
                sequence = [
                    [BoundingBox(x=10, y=10, width=12, height=12)],
                    [BoundingBox(x=12, y=11, width=12, height=12)],
                ]
                bboxes = sequence[min(self._index, len(sequence) - 1)]
                self._index += 1
                return MotionMeasurementResult(motion_fraction=0.2, bboxes=bboxes)

        manager = MotionDetectionManager(config=cfg, algorithm=SequenceAlgorithm())
        first = manager.detect(_make_input(camera_id="cam-iou"))
        second = manager.detect(_make_input(camera_id="cam-iou"))
        self.assertFalse(first["detected"])
        self.assertTrue(second["detected"])


# ---------------------------------------------------------------------------
# AssetIntegrationTests — visual test assets smoke test
# ---------------------------------------------------------------------------


class AssetIntegrationTests(unittest.TestCase):
    """Run the real algorithm manager on all visual test assets and verify structure."""

    FRAMES_ROOT = Path(__file__).parent / "assets" / "frames"
    CATEGORIES = ("no_motion", "small_motion", "clear_motion")

    def _iter_case_paths(self):
        for category in self.CATEGORIES:
            cat_path = self.FRAMES_ROOT / category
            if not cat_path.is_dir():
                continue
            for case_path in sorted(cat_path.iterdir()):
                if case_path.is_dir():
                    yield category, case_path.name, case_path

    def _image_struct(self, arr: np.ndarray) -> Image:
        h, w = arr.shape[0], arr.shape[1]
        return Image(
            data=arr, width=w, height=h,
            color_format="GRAY", layout="HWC", dtype="uint8", value_range="[0,255]",
        )

    def test_all_cases_process_without_error(self) -> None:
        manager = MotionDetectionManager()  # uses real algorithm by default
        for category, case_name, case_path in self._iter_case_paths():
            prev_path = case_path / "previous.png"
            curr_path = case_path / "current.png"
            if not prev_path.exists() or not curr_path.exists():
                continue

            prev_arr = cv2.imread(str(prev_path), cv2.IMREAD_GRAYSCALE)
            curr_arr = cv2.imread(str(curr_path), cv2.IMREAD_GRAYSCALE)
            if prev_arr is None or curr_arr is None:
                self.fail(f"Could not load images from {case_path}")

            inp = MotionDetectionInput(
                current_frame=MotionInputFrame(
                    frame_id="frame_0002",
                    camera_id=case_name,
                    timestamp_ms=2000,
                    image=self._image_struct(curr_arr),
                ),
                previous_frame=MotionInputFrame(
                    frame_id="frame_0001",
                    camera_id=case_name,
                    timestamp_ms=1000,
                    image=self._image_struct(prev_arr),
                ),
            )

            with self.subTest(category=category, case=case_name):
                result = manager.detect(inp)
                self.assertIn("detected", result)
                self.assertIsInstance(result["detected"], bool)
                self.assertIn("bboxes", result)
                self.assertIsInstance(result["bboxes"], list)
                # Structural guarantees when motion is detected
                if result["detected"]:
                    self.assertGreater(len(result["bboxes"]), 0)
                    h, w = curr_arr.shape[0], curr_arr.shape[1]
                    for bbox in result["bboxes"]:
                        self.assertGreater(bbox["width"], 0)
                        self.assertGreater(bbox["height"], 0)
                        self.assertGreaterEqual(bbox["x"], 0)
                        self.assertGreaterEqual(bbox["y"], 0)
                        self.assertLessEqual(bbox["x"] + bbox["width"], w)
                        self.assertLessEqual(bbox["y"] + bbox["height"], h)
                else:
                    self.assertEqual(result["bboxes"], [])


if __name__ == "__main__":
    unittest.main()
