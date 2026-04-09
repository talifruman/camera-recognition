from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
FD_TESTS_DIR = Path(__file__).resolve().parent
OD_TESTS_DIR = PROJECT_ROOT / "tests" / "object_detection"

for p in (SRC_DIR, FD_TESTS_DIR, OD_TESTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {module_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


two_stage_app = _load_module(
    "two_stage_visual_test_app",
    FD_TESTS_DIR / "two_stage_visual_test_app.py",
)


class TwoStageProcessImagesTests(unittest.TestCase):
    """Integration tests for the two-stage OD + FD pipeline."""

    def test_process_images_produces_output_with_yellow_and_green_pixels(self) -> None:
        """Stub OD detects persons → FD adds green face annotations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            input_root = temp_root / "input"
            output_root = temp_root / "output"
            nested_dir = input_root / "room1"
            nested_dir.mkdir(parents=True)
            Image.new("RGB", (200, 160), color=(0, 0, 0)).save(
                nested_dir / "sample.png"
            )

            with mock.patch.object(
                two_stage_app,
                "build_run_output_root",
                return_value=output_root / "fixed_run",
            ):
                count = two_stage_app.process_images(
                    input_root, output_root, "stub"
                )

            self.assertEqual(count, 1)
            rendered_path = output_root / "fixed_run" / "room1" / "sample.png"
            self.assertTrue(rendered_path.exists())

            rendered = np.array(Image.open(rendered_path).convert("RGB"))

            # Yellow pixels (Object Detection person boxes: R>200, G>200, B<100)
            yellow_pixels = int(np.count_nonzero(
                (rendered[:, :, 0] > 200)
                & (rendered[:, :, 1] > 200)
                & (rendered[:, :, 2] < 100)
            ))
            self.assertGreater(yellow_pixels, 0, "Expected yellow OD boxes")

            # Green pixels (Face Detection boxes + landmarks: R<100, G>200, B<100)
            green_pixels = int(np.count_nonzero(
                (rendered[:, :, 0] < 100)
                & (rendered[:, :, 1] > 200)
                & (rendered[:, :, 2] < 100)
            ))
            self.assertGreater(green_pixels, 0, "Expected green FD annotations")

    def test_process_images_preserves_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            input_root = temp_root / "input"
            output_root = temp_root / "output"
            sub = input_root / "a" / "b"
            sub.mkdir(parents=True)
            Image.new("RGB", (100, 80), color=(128, 128, 128)).save(
                sub / "img.jpg"
            )

            with mock.patch.object(
                two_stage_app,
                "build_run_output_root",
                return_value=output_root / "run",
            ):
                count = two_stage_app.process_images(
                    input_root, output_root, "stub"
                )

            self.assertEqual(count, 1)
            expected = output_root / "run" / "a" / "b" / "img.jpg"
            self.assertTrue(expected.exists())

    def test_non_image_files_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            input_root = temp_root / "input"
            input_root.mkdir()
            (input_root / "notes.txt").write_text("skip me", encoding="utf-8")

            with mock.patch.object(
                two_stage_app,
                "build_run_output_root",
                return_value=temp_root / "output" / "run",
            ):
                count = two_stage_app.process_images(
                    input_root, temp_root / "output", "stub"
                )

            self.assertEqual(count, 0)

    def test_main_requires_three_arguments(self) -> None:
        with self.assertRaises(ValueError):
            two_stage_app.main(["only_one"])

    def test_main_passes_arguments_to_process_images(self) -> None:
        with mock.patch.object(
            two_stage_app, "process_images", return_value=0
        ) as proc_mock:
            exit_code = two_stage_app.main(
                ["input_dir", "output_dir", "stub"]
            )

        self.assertEqual(exit_code, 0)
        proc_mock.assert_called_once_with(
            Path("input_dir"), Path("output_dir"), "stub"
        )

    def test_rejects_invalid_detector_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                ValueError, "detector_mode must be one of"
            ):
                two_stage_app.process_images(
                    Path(temp_dir), Path(temp_dir), "bad_mode"
                )


if __name__ == "__main__":
    unittest.main()
