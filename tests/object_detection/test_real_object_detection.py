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
    ObjectDetectionModule,
    RawDetection,
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
                "frame_id": 5,
                "width": 120,
                "height": 100,
            },
        )

        self.assertEqual(set(result), {"frame_id", "person_detected", "persons"})
        self.assertEqual(result["frame_id"], 5)
        self.assertTrue(result["person_detected"])
        self.assertEqual(result["persons"], [{"x": 10, "y": 12, "width": 30, "height": 40}])

    def test_process_rejects_missing_camera_id(self) -> None:
        module = ObjectDetectionModule(inference_engine=StaticInferenceEngine([]))

        with self.assertRaises(ValueError):
            module.process(
                np.zeros((10, 10, 3), dtype=np.uint8),
                {
                    "camera_id": "",
                    "frame_id": 9,
                    "width": 10,
                    "height": 10,
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
                "frame_id": 11,
                "width": 200,
                "height": 100,
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
                "frame_id": 21,
                "width": image.shape[1],
                "height": image.shape[0],
            },
        )

        self.assertEqual(set(result), {"frame_id", "person_detected", "persons"})
        self.assertEqual(result["frame_id"], 21)
        self.assertIsInstance(result["person_detected"], bool)
        for bbox in result["persons"]:
            self.assertEqual(set(bbox), {"x", "y", "width", "height"})
            self.assertGreaterEqual(bbox["x"], 0)
            self.assertGreaterEqual(bbox["y"], 0)
            self.assertGreater(bbox["width"], 0)
            self.assertGreater(bbox["height"], 0)
            self.assertLessEqual(bbox["x"] + bbox["width"], image.shape[1])
            self.assertLessEqual(bbox["y"] + bbox["height"], image.shape[0])


if __name__ == "__main__":
    unittest.main()