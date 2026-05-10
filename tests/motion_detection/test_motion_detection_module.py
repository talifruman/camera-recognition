from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_processing.motion_detection import (  # type: ignore[import-not-found]
    BoundingBox,
    FramePacket,
    FrameDifferencingMotionDetector,
    InputValidator,
    MotionDetectionAlgorithm,
    MotionDetectionConfig,
    MotionDetectionInput,
    MotionDetectionManager,
    MotionDetectionResultInternal,
    MotionMeasurementResult,
    MotionResult,
    StubMotionDetectionAlgorithm,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_gray_image(height: int = 64, width: int = 64) -> np.ndarray:
    """Return a valid GRAY HWC uint8 image with the given dimensions."""
    return np.zeros((height, width), dtype=np.uint8)


def _make_frame(
    *,
    frame_id: str = "frame_0001",
    camera_id: str = "cam-a",
    timestamp_ms: int = 1000,
    height: int = 64,
    width: int = 64,
    image: np.ndarray | None = None,
) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        camera_id=camera_id,
        timestamp_ms=timestamp_ms,
        image=image if image is not None else _make_gray_image(height, width),
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
) -> MotionDetectionManager:
    config = MotionDetectionConfig(
        motion_threshold=25,
        motion_fraction_threshold=motion_fraction_threshold,
        min_bbox_area=100,
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
        self._assert_no_motion(manager.process(bad))

    def test_missing_previous_frame_returns_no_motion(self) -> None:
        manager = _make_manager()
        bad: MotionDetectionInput = {  # type: ignore[typeddict-item]
            "current_frame": _make_frame(),
        }
        self._assert_no_motion(manager.process(bad))

    def test_empty_current_camera_id_returns_no_motion(self) -> None:
        manager = _make_manager()
        inp = _make_input()
        inp["current_frame"]["camera_id"] = ""
        self._assert_no_motion(manager.process(inp))

    def test_empty_previous_camera_id_returns_no_motion(self) -> None:
        manager = _make_manager()
        inp = _make_input()
        inp["previous_frame"]["camera_id"] = ""
        self._assert_no_motion(manager.process(inp))

    def test_camera_id_mismatch_returns_no_motion(self) -> None:
        manager = _make_manager()
        inp = _make_input()
        inp["previous_frame"]["camera_id"] = "cam-b"
        self._assert_no_motion(manager.process(inp))

    def test_timestamp_ordering_violation_returns_no_motion(self) -> None:
        manager = _make_manager()
        # previous_ts > current_ts — violates spec §2.3
        inp = _make_input(current_ts=1000, previous_ts=2000)
        self._assert_no_motion(manager.process(inp))

    def test_equal_timestamps_are_accepted(self) -> None:
        """previous_ts == current_ts is valid per spec (≤ relationship)."""
        manager = _make_manager()
        inp = _make_input(current_ts=1000, previous_ts=1000)
        result = manager.process(inp)
        self.assertIn("detected", result)
        # Not checking detected value — just that it doesn't fail validation

    def test_image_dimension_mismatch_returns_no_motion(self) -> None:
        manager = _make_manager()
        inp = MotionDetectionInput(
            current_frame=_make_frame(height=64, width=64),
            previous_frame=_make_frame(height=32, width=32),
        )
        self._assert_no_motion(manager.process(inp))

    def test_wrong_dtype_current_image_returns_no_motion(self) -> None:
        manager = _make_manager()
        inp = _make_input()
        inp["current_frame"]["image"] = np.zeros((64, 64), dtype=np.float32)
        self._assert_no_motion(manager.process(inp))

    def test_wrong_dtype_previous_image_returns_no_motion(self) -> None:
        manager = _make_manager()
        inp = _make_input()
        inp["previous_frame"]["image"] = np.zeros((64, 64), dtype=np.float32)
        self._assert_no_motion(manager.process(inp))

    def test_multichannel_non_gray_image_returns_no_motion(self) -> None:
        manager = _make_manager()
        inp = _make_input()
        # 3-channel (RGB-like) image — violates GRAY contract
        inp["current_frame"]["image"] = np.zeros((64, 64, 3), dtype=np.uint8)
        self._assert_no_motion(manager.process(inp))

    def test_none_current_image_returns_no_motion(self) -> None:
        manager = _make_manager()
        inp = _make_input()
        inp["current_frame"]["image"] = None  # type: ignore[typeddict-item]
        self._assert_no_motion(manager.process(inp))


# ---------------------------------------------------------------------------
# StubPipelineTests — spec §5 (stub behavior) + §7 (decision policy)
# ---------------------------------------------------------------------------


class StubPipelineTests(unittest.TestCase):
    """Verify stub produces correct deterministic output through the pipeline."""

    def test_valid_input_returns_detected_true_at_default_threshold(self) -> None:
        # Default threshold 0.05; stub fraction 0.2 => detected=True
        manager = _make_stub_manager(motion_fraction_threshold=0.05)
        result = manager.process(_make_input())
        self.assertTrue(result["detected"])

    def test_valid_input_returns_non_empty_bboxes_when_detected(self) -> None:
        manager = _make_stub_manager(motion_fraction_threshold=0.05)
        result = manager.process(_make_input())
        self.assertGreater(len(result["bboxes"]), 0)

    def test_bbox_has_strictly_positive_dimensions(self) -> None:
        manager = _make_stub_manager()
        result = manager.process(_make_input())
        for bbox in result["bboxes"]:
            self.assertGreater(bbox["width"], 0)
            self.assertGreater(bbox["height"], 0)

    def test_bbox_is_inside_current_image_bounds(self) -> None:
        h, w = 64, 80
        manager = _make_stub_manager()
        result = manager.process(_make_input(height=h, width=w))
        for bbox in result["bboxes"]:
            self.assertGreaterEqual(bbox["x"], 0)
            self.assertGreaterEqual(bbox["y"], 0)
            self.assertLessEqual(bbox["x"] + bbox["width"], w)
            self.assertLessEqual(bbox["y"] + bbox["height"], h)

    def test_bbox_inside_bounds_for_minimal_image(self) -> None:
        """Even a 1×1 image must produce a valid in-bounds bbox."""
        manager = _make_stub_manager()
        result = manager.process(_make_input(height=1, width=1))
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
        result = manager.process(_make_input())
        self.assertFalse(result["detected"])

    def test_bboxes_empty_when_not_detected(self) -> None:
        manager = _make_stub_manager(motion_fraction_threshold=0.5)
        result = manager.process(_make_input())
        self.assertEqual(result["bboxes"], [])

    def test_detected_true_at_exact_threshold(self) -> None:
        # Stub fraction = 0.2; threshold = 0.2 => motion_fraction >= threshold
        manager = _make_stub_manager(motion_fraction_threshold=0.2)
        result = manager.process(_make_input())
        self.assertTrue(result["detected"])

    def test_result_is_deterministic(self) -> None:
        manager = _make_stub_manager()
        inp = _make_input()
        result1 = manager.process(inp)
        result2 = manager.process(inp)
        self.assertEqual(result1, result2)

    def test_result_with_single_channel_3d_image(self) -> None:
        """(H, W, 1) grayscale images must be accepted and processed."""
        manager = _make_stub_manager()
        inp = MotionDetectionInput(
            current_frame=_make_frame(
                image=np.zeros((64, 64, 1), dtype=np.uint8)
            ),
            previous_frame=_make_frame(
                image=np.zeros((64, 64, 1), dtype=np.uint8)
            ),
        )
        result = manager.process(inp)
        self.assertIn("detected", result)
        self.assertIn("bboxes", result)


# ---------------------------------------------------------------------------
# ManagerBehaviorTests — spec §8.1 / §11
# ---------------------------------------------------------------------------


class ManagerBehaviorTests(unittest.TestCase):
    """Manager orchestration and error-handling contract."""

    def test_output_has_exactly_detected_and_bboxes_keys(self) -> None:
        manager = _make_manager()
        result = manager.process(_make_input())
        self.assertEqual(set(result), {"detected", "bboxes"})

    def test_manager_does_not_mutate_input_dict(self) -> None:
        manager = _make_manager()
        inp = _make_input()

        # Snapshot the identity of inner objects
        current_id = id(inp["current_frame"])
        previous_id = id(inp["previous_frame"])
        current_img_id = id(inp["current_frame"]["image"])
        previous_img_id = id(inp["previous_frame"]["image"])

        manager.process(inp)

        self.assertEqual(id(inp["current_frame"]), current_id)
        self.assertEqual(id(inp["previous_frame"]), previous_id)
        self.assertEqual(id(inp["current_frame"]["image"]), current_img_id)
        self.assertEqual(id(inp["previous_frame"]["image"]), previous_img_id)

    def test_algorithm_exception_returns_no_motion(self) -> None:
        class ExplodingAlgorithm:
            def measure(self, previous_image: np.ndarray, current_image: np.ndarray) -> MotionMeasurementResult:
                raise RuntimeError("simulated algorithm failure")

        manager = _make_manager(algorithm=ExplodingAlgorithm())  # type: ignore[arg-type]
        result = manager.process(_make_input())
        self.assertFalse(result["detected"])
        self.assertEqual(result["bboxes"], [])

    def test_detected_is_bool(self) -> None:
        manager = _make_manager()
        result = manager.process(_make_input())
        self.assertIsInstance(result["detected"], bool)

    def test_bboxes_is_list(self) -> None:
        manager = _make_manager()
        result = manager.process(_make_input())
        self.assertIsInstance(result["bboxes"], list)

    def test_bbox_fields_are_correct_types(self) -> None:
        manager = _make_stub_manager()
        result = manager.process(_make_input())
        self.assertTrue(result["detected"])
        bbox = result["bboxes"][0]
        for field_name in ("x", "y", "width", "height"):
            self.assertIsInstance(bbox[field_name], int, msg=f"{field_name} must be int")


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
        self.assertEqual(manager1.process(inp), manager2.process(inp))

    def test_multiple_invocations_produce_identical_output(self) -> None:
        manager = _make_manager()
        inp = _make_input()
        results = [manager.process(inp) for _ in range(5)]
        self.assertTrue(all(r == results[0] for r in results))

    def test_different_image_sizes_produce_in_bounds_bbox(self) -> None:
        """Verify deterministic bbox stays in bounds for various sizes."""
        manager = _make_manager()
        for h, w in [(1, 1), (4, 4), (10, 20), (100, 200), (480, 640)]:
            with self.subTest(h=h, w=w):
                result = manager.process(_make_input(height=h, width=w))
                if result["detected"]:
                    bbox = result["bboxes"][0]
                    self.assertGreaterEqual(bbox["x"], 0)
                    self.assertGreaterEqual(bbox["y"], 0)
                    self.assertLessEqual(bbox["x"] + bbox["width"], w)
                    self.assertLessEqual(bbox["y"] + bbox["height"], h)
                    self.assertGreater(bbox["width"], 0)
                    self.assertGreater(bbox["height"], 0)


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
        return MotionDetectionInput(
            current_frame=FramePacket(
                frame_id=2, camera_id="cam-a", timestamp_ms=2000, image=curr_img
            ),
            previous_frame=FramePacket(
                frame_id=1, camera_id="cam-a", timestamp_ms=1000, image=prev_img
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
        result = manager.process(self._motion_input(img, img))
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
        result = manager.process(self._motion_input(prev_img, curr_img))
        self.assertFalse(result["detected"])

    # --- motion_fraction_threshold -----------------------------------------

    def test_low_motion_fraction_below_threshold_not_detected(self) -> None:
        """A 15×15 block in a 200×200 image: fraction ~0.0056 < 0.05."""
        h, w = 200, 200
        prev_img = np.zeros((h, w), dtype=np.uint8)
        curr_img = np.zeros((h, w), dtype=np.uint8)
        curr_img[90:105, 90:105] = 255  # area 225 >= min_bbox_area=100

        manager = self._real_manager(motion_fraction_threshold=0.05, min_bbox_area=100)
        result = manager.process(self._motion_input(prev_img, curr_img))
        self.assertFalse(result["detected"])

    def test_high_motion_fraction_above_threshold_detected(self) -> None:
        """A 50×50 block in a 100×100 image: fraction 0.25 >= 0.05."""
        h, w = 100, 100
        prev_img = np.zeros((h, w), dtype=np.uint8)
        curr_img = np.zeros((h, w), dtype=np.uint8)
        curr_img[25:75, 25:75] = 255  # area 2500, fraction 0.25

        manager = self._real_manager(motion_fraction_threshold=0.05, min_bbox_area=100)
        result = manager.process(self._motion_input(prev_img, curr_img))
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

    def test_all_cases_process_without_error(self) -> None:
        manager = MotionDetectionManager()  # uses real algorithm by default
        for category, case_name, case_path in self._iter_case_paths():
            prev_path = case_path / "previous.png"
            curr_path = case_path / "current.png"
            if not prev_path.exists() or not curr_path.exists():
                continue

            prev_img = cv2.imread(str(prev_path), cv2.IMREAD_GRAYSCALE)
            curr_img = cv2.imread(str(curr_path), cv2.IMREAD_GRAYSCALE)
            if prev_img is None or curr_img is None:
                self.fail(f"Could not load images from {case_path}")

            inp: MotionDetectionInput = {
                "current_frame": {
                    "frame_id": "frame_0002",
                    "camera_id": case_name,
                    "timestamp_ms": 2000,
                    "image": curr_img,
                },
                "previous_frame": {
                    "frame_id": "frame_0001",
                    "camera_id": case_name,
                    "timestamp_ms": 1000,
                    "image": prev_img,
                },
            }

            with self.subTest(category=category, case=case_name):
                result = manager.process(inp)
                self.assertIn("detected", result)
                self.assertIsInstance(result["detected"], bool)
                self.assertIn("bboxes", result)
                self.assertIsInstance(result["bboxes"], list)
                # Structural guarantees when motion is detected
                if result["detected"]:
                    self.assertGreater(len(result["bboxes"]), 0)
                    h, w = curr_img.shape[0], curr_img.shape[1]
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
