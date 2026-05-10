from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_processing.object_detection import (  # type: ignore[import-not-found]
    BoundingBox,
    Image,
    ObjectDetectionInput,
    ObjectDetectionModule,
    OutputImageType,
    RawDetection,
    ResizePolicy,
)


class StaticInferenceEngine:
    def __init__(self, detections: list[RawDetection]) -> None:
        self._detections = detections

    def infer(self, model_ready_input: np.ndarray, config):
        return list(self._detections)


class ObjectDetectionModuleTests(unittest.TestCase):
    def test_process_returns_documented_public_contract(self) -> None:
        detections = [
            RawDetection(
                label="person",
                confidence=0.91,
                bbox=BoundingBox(x=10, y=12, width=30, height=40),
            ),
            RawDetection(
                label="car",
                confidence=0.99,
                bbox=BoundingBox(x=1, y=1, width=15, height=15),
            ),
        ]
        module = ObjectDetectionModule(inference_engine=StaticInferenceEngine(detections))

        result = module.process(
            np.zeros((100, 120, 3), dtype=np.uint8),
            {
                "camera_id": "camera-a",
                "frame_id": "frame_0005",
                "timestamp_ms": 1000,
                "width": 120,
                "height": 100,
                "roi_bbox_frame": {"x": 0, "y": 0, "width": 120, "height": 100},
            },
        )

        self.assertEqual(set(result), {"frame_id", "person_detected", "persons"})
        self.assertEqual(result["frame_id"], "frame_0005")
        self.assertTrue(result["person_detected"])
        self.assertEqual(result["persons"], [{"x": 10, "y": 12, "width": 30, "height": 40}])

    def test_process_rejects_missing_camera_id(self) -> None:
        module = ObjectDetectionModule(inference_engine=StaticInferenceEngine([]))

        with self.assertRaises(ValueError):
            module.process(
                np.zeros((10, 10, 3), dtype=np.uint8),
                {
                    "camera_id": "",
                    "frame_id": "frame_0009",
                    "timestamp_ms": 1000,
                    "width": 10,
                    "height": 10,
                    "roi_bbox_frame": {"x": 0, "y": 0, "width": 10, "height": 10},
                },
            )

    def test_process_clips_and_drops_invalid_boxes(self) -> None:
        detections = [
            RawDetection(
                label="person",
                confidence=0.95,
                bbox=BoundingBox(x=-5, y=8, width=20, height=30),
            ),
            RawDetection(
                label="person",
                confidence=0.95,
                bbox=BoundingBox(x=205, y=10, width=30, height=20),
            ),
        ]
        module = ObjectDetectionModule(inference_engine=StaticInferenceEngine(detections))

        result = module.process(
            np.zeros((100, 200, 3), dtype=np.uint8),
            {
                "camera_id": "camera-b",
                "frame_id": "frame_0011",
                "timestamp_ms": 1000,
                "width": 200,
                "height": 100,
                "roi_bbox_frame": {"x": 0, "y": 0, "width": 200, "height": 100},
            },
        )

        self.assertEqual(result["persons"], [{"x": 0, "y": 8, "width": 15, "height": 30}])


@unittest.skipUnless(os.environ.get("RUN_REAL_DETECTOR_SMOKE") == "1", "real detector smoke test disabled")
class RealObjectDetectionSmokeTests(unittest.TestCase):
    def test_process_runs_real_inference_on_asset(self) -> None:
        asset_path = Path(__file__).resolve().parent / "assets" / "one_person" / "one_person_01.jpg"
        image = np.array(Image.open(asset_path).convert("RGB"))
        module = ObjectDetectionModule()

        result = module.process(
            image,
            {
                "camera_id": "camera-real",
                "frame_id": "frame_0021",
                "timestamp_ms": 1000,
                "width": image.shape[1],
                "height": image.shape[0],
                "roi_bbox_frame": {"x": 0, "y": 0, "width": image.shape[1], "height": image.shape[0]},
            },
        )

        self.assertEqual(set(result), {"frame_id", "person_detected", "persons"})
        self.assertEqual(result["frame_id"], "frame_0021")
        self.assertIsInstance(result["person_detected"], bool)
        for bbox in result["persons"]:
            self.assertEqual(set(bbox), {"x", "y", "width", "height"})
            self.assertGreaterEqual(bbox["x"], 0)
            self.assertGreaterEqual(bbox["y"], 0)
            self.assertGreater(bbox["width"], 0)
            self.assertGreater(bbox["height"], 0)
            self.assertLessEqual(bbox["x"] + bbox["width"], image.shape[1])
            self.assertLessEqual(bbox["y"] + bbox["height"], image.shape[0])


def _make_roi_image(image: np.ndarray) -> Image:
    return {
        "data": image,
        "width": image.shape[1],
        "height": image.shape[0],
        "color_format": "RGB",
        "layout": "HWC",
        "dtype": "uint8",
        "value_range": "[0,255]",
    }


def _make_roi_bbox() -> BoundingBox:
    return {"x": 0, "y": 0, "width": 640, "height": 640}


def _make_detect_input(
    image: np.ndarray,
    *,
    frame_id: str = "frame_0100",
    camera_id: str = "cam-a",
    timestamp_ms: int = 1000,
    roi_bbox_frame: BoundingBox | None = None,
) -> ObjectDetectionInput:
    if roi_bbox_frame is None:
        roi_bbox_frame = _make_roi_bbox()
    return {
        "frame_id": frame_id,
        "camera_id": camera_id,
        "timestamp_ms": timestamp_ms,
        "roi_image": _make_roi_image(image),
        "roi_bbox_frame": roi_bbox_frame,
        "width": image.shape[1],
        "height": image.shape[0],
    }


class ObjectDetectionDetectMethodTests(unittest.TestCase):
    """Tests for detect() and get_input_contract() — the RPM-facing public API."""

    def _module_with(self, detections: list[RawDetection]) -> ObjectDetectionModule:
        return ObjectDetectionModule(inference_engine=StaticInferenceEngine(detections))

    def _blank_image(self, h: int = 640, w: int = 640) -> np.ndarray:
        return np.zeros((h, w, 3), dtype=np.uint8)

    # --- detect() exists and returns the correct structure ---

    def test_detect_returns_person_detection_result_keys(self) -> None:
        module = self._module_with([])
        result = module.detect(_make_detect_input(self._blank_image()))
        self.assertEqual(set(result), {"frame_id", "person_detected", "persons"})

    def test_detect_preserves_frame_id(self) -> None:
        module = self._module_with([])
        result = module.detect(_make_detect_input(self._blank_image(), frame_id="frame_abc"))
        self.assertEqual(result["frame_id"], "frame_abc")

    def test_detect_delegates_to_internal_pipeline(self) -> None:
        detections = [
            RawDetection(
                label="person",
                confidence=0.9,
                bbox={"x": 5, "y": 10, "width": 20, "height": 30},
            )
        ]
        module = self._module_with(detections)
        result = module.detect(_make_detect_input(self._blank_image()))
        self.assertTrue(result["person_detected"])
        self.assertEqual(result["persons"], [{"x": 5, "y": 10, "width": 20, "height": 30}])

    def test_detect_returns_roi_local_boxes_without_projection(self) -> None:
        # Object Detection must return ROI-local boxes — not projected to full-frame.
        # Verify the output boxes are clipped to roi_image dimensions, not any larger frame.
        detections = [
            RawDetection(
                label="person",
                confidence=0.9,
                bbox={"x": 10, "y": 20, "width": 50, "height": 60},
            )
        ]
        module = self._module_with(detections)
        # roi_image is 100x100; roi_bbox_frame places it at (500, 500) in some larger frame
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        roi_bbox_frame: BoundingBox = {"x": 500, "y": 500, "width": 100, "height": 100}
        result = module.detect(_make_detect_input(image, roi_bbox_frame=roi_bbox_frame))
        # Boxes must be in ROI-local coordinates — x=10, not x=510
        self.assertEqual(result["persons"][0]["x"], 10)
        self.assertEqual(result["persons"][0]["y"], 20)

    def test_detect_confidence_not_in_result(self) -> None:
        detections = [
            RawDetection(
                label="person",
                confidence=0.99,
                bbox={"x": 0, "y": 0, "width": 10, "height": 10},
            )
        ]
        module = self._module_with(detections)
        result = module.detect(_make_detect_input(self._blank_image()))
        # confidence must not appear in PersonDetectionResult or in individual person entries
        self.assertNotIn("confidence", result)
        for person in result["persons"]:
            self.assertNotIn("confidence", person)

    # --- detect() validation ---

    def test_detect_raises_on_missing_timestamp_ms(self) -> None:
        module = self._module_with([])
        image = self._blank_image()
        bad_input: ObjectDetectionInput = {
            "frame_id": "f1",
            "camera_id": "cam",
            "timestamp_ms": 0,  # will be overridden below
            "roi_image": image,
            "roi_bbox_frame": _make_roi_bbox(),
            "width": image.shape[1],
            "height": image.shape[0],
        }
        # Simulate missing timestamp_ms via dict manipulation after construction
        del bad_input["timestamp_ms"]  # type: ignore[misc]
        with self.assertRaises((ValueError, KeyError)):
            module.detect(bad_input)

    def test_detect_raises_on_missing_roi_bbox_frame(self) -> None:
        module = self._module_with([])
        image = self._blank_image()
        bad_input: ObjectDetectionInput = {
            "frame_id": "f1",
            "camera_id": "cam",
            "timestamp_ms": 1000,
            "roi_image": image,
            "roi_bbox_frame": _make_roi_bbox(),
            "width": image.shape[1],
            "height": image.shape[0],
        }
        del bad_input["roi_bbox_frame"]  # type: ignore[misc]
        with self.assertRaises((ValueError, KeyError)):
            module.detect(bad_input)

    def test_detect_raises_on_zero_roi_bbox_width(self) -> None:
        module = self._module_with([])
        image = self._blank_image()
        bad_roi_bbox: BoundingBox = {"x": 0, "y": 0, "width": 0, "height": 100}
        with self.assertRaises(ValueError):
            module.detect(_make_detect_input(image, roi_bbox_frame=bad_roi_bbox))

    def test_detect_raises_on_zero_roi_bbox_height(self) -> None:
        module = self._module_with([])
        image = self._blank_image()
        bad_roi_bbox: BoundingBox = {"x": 0, "y": 0, "width": 100, "height": 0}
        with self.assertRaises(ValueError):
            module.detect(_make_detect_input(image, roi_bbox_frame=bad_roi_bbox))

    # --- get_input_contract() ---

    def test_get_input_contract_returns_expected_image_type(self) -> None:
        module = ObjectDetectionModule()
        contract = module.get_input_contract()
        self.assertEqual(contract["output_image_type"], OutputImageType.RGB_UINT8_HWC)

    def test_get_input_contract_returns_640x640_letterbox(self) -> None:
        module = ObjectDetectionModule()
        contract = module.get_input_contract()
        geometry = contract["geometry_spec"]
        self.assertEqual(geometry["width"], 640)
        self.assertEqual(geometry["height"], 640)
        self.assertEqual(geometry["resize_policy"], ResizePolicy.LETTERBOX)

    def test_get_input_contract_has_required_keys(self) -> None:
        module = ObjectDetectionModule()
        contract = module.get_input_contract()
        self.assertIn("output_image_type", contract)
        self.assertIn("geometry_spec", contract)

    # --- RPM-facing integration: get_input_contract() then detect() ---

    def test_rpm_facing_workflow(self) -> None:
        module = ObjectDetectionModule(inference_engine=StaticInferenceEngine([]))
        contract = module.get_input_contract()
        # Simulate RPM constructing ObjectDetectionInput from FTL ProcessedFrame
        w = contract["geometry_spec"]["width"]
        h = contract["geometry_spec"]["height"]
        image = np.zeros((h, w, 3), dtype=np.uint8)
        result = module.detect(_make_detect_input(image))
        self.assertIn("frame_id", result)
        self.assertIn("person_detected", result)
        self.assertIn("persons", result)


if __name__ == "__main__":
    unittest.main()