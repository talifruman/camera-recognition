from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
from PIL import Image

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))


def load_module(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, TESTS_DIR / file_name)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {module_name}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


object_detection_stub = load_module("object_detection_stub", "object_detection_stub.py")
visual_test_app = load_module("visual_test_app", "visual_test_app.py")


class ObjectDetectionStubTests(unittest.TestCase):
    def test_process_returns_documented_public_contract(self) -> None:
        image = np.zeros((120, 200, 3), dtype=np.uint8)
        metadata = {
            "camera_id": "camera-a",
            "frame_id": "frame_0007",
            "width": 200,
            "height": 120,
        }

        result = object_detection_stub.process(image, metadata)

        self.assertEqual(set(result), {"frame_id", "person_detected", "persons"})
        self.assertEqual(result["frame_id"], "frame_0007")
        self.assertTrue(result["person_detected"])
        self.assertEqual(len(result["persons"]), 2)

        for bbox in result["persons"]:
            self.assertEqual(set(bbox), {"x", "y", "width", "height"})
            self.assertGreaterEqual(bbox["x"], 0)
            self.assertGreaterEqual(bbox["y"], 0)
            self.assertGreater(bbox["width"], 0)
            self.assertGreater(bbox["height"], 0)
            self.assertLessEqual(bbox["x"] + bbox["width"], 200)
            self.assertLessEqual(bbox["y"] + bbox["height"], 120)


class VisualTestAppTests(unittest.TestCase):
    def test_process_images_preserves_relative_paths_and_renders_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            input_root = temp_root / "input"
            output_root = temp_root / "output"
            nested_dir = input_root / "room1" / "nested"
            nested_dir.mkdir(parents=True)
            image_path = nested_dir / "sample.png"
            Image.new("RGB", (160, 100), color=(0, 0, 0)).save(image_path)
            (input_root / "ignore.txt").write_text("skip me", encoding="utf-8")

            with mock.patch.object(
                visual_test_app,
                "current_run_folder_name",
                return_value="2026-04-06_14-32-10",
            ):
                processed_count = visual_test_app.process_images(input_root, output_root, "stub")

            self.assertEqual(processed_count, 1)
            rendered_path = (
                output_root
                / "2026-04-06_14-32-10"
                / "room1"
                / "nested"
                / "sample.png"
            )
            self.assertTrue(rendered_path.exists())

            rendered_image = np.array(Image.open(rendered_path).convert("RGB"))
            yellow_pixels = np.count_nonzero(
                (rendered_image[:, :, 0] > 200)
                & (rendered_image[:, :, 1] > 200)
                & (rendered_image[:, :, 2] < 100)
            )
            self.assertGreater(int(yellow_pixels), 0)

    def test_process_images_rejects_invalid_detector_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)

            with self.assertRaisesRegex(ValueError, "detector_mode must be one of: stub, real"):
                visual_test_app.process_images(temp_root / "input", temp_root / "output", "bad-mode")

    def test_main_requires_three_arguments(self) -> None:
        input_root = Path("tests/object_detection/assets")
        output_root = Path("tests/object_detection/rendered_assets")

        with mock.patch.object(visual_test_app, "process_images", return_value=0) as process_images_mock:
            exit_code = visual_test_app.main([str(input_root), str(output_root), "stub"])

        self.assertEqual(exit_code, 0)
        process_images_mock.assert_called_once_with(input_root, output_root, "stub")


if __name__ == "__main__":
    unittest.main()