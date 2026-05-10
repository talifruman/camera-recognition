from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
TESTS_DIR = Path(__file__).resolve().parent
for p in (SRC_DIR, TESTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from image_processing.face_detection import (  # type: ignore[import-not-found]
    BoundingBox,
    DetectedFace,
    FaceDetectionConfig,
    FaceDetectionInput,
    FaceDetectionModule,
    FaceDetectionOutput,
    FaceLandmarks,
    Image,
    Point,
    RawFaceDetection,
)
from face_detection_stub_engine import StubFaceDetectorEngine  # type: ignore[import-not-found]


def _make_roi_image(roi_w: int, roi_h: int) -> Image:
    return {
        "data": np.zeros((roi_h, roi_w, 3), dtype=np.uint8),
        "width": roi_w,
        "height": roi_h,
        "color_format": "RGB",
        "layout": "HWC",
        "dtype": "uint8",
        "value_range": "[0, 255]",
    }


def _make_input(
    roi_w: int = 100,
    roi_h: int = 200,
    frame_id: str = "frame_0001",
    camera_id: str = "cam-a",
    timestamp_ms: int = 1000,
) -> FaceDetectionInput:
    return {
        "frame_id": frame_id,
        "camera_id": camera_id,
        "timestamp_ms": timestamp_ms,
        "roi_image": _make_roi_image(roi_w, roi_h),
    }


def _make_module(
    engine: StubFaceDetectorEngine | None = None,
    confidence_threshold: float = 0.5,
) -> FaceDetectionModule:
    return FaceDetectionModule(
        config=FaceDetectionConfig(confidence_threshold=confidence_threshold),
        detector_engine=engine or StubFaceDetectorEngine(),
    )


class FaceDetectionModuleOutputContractTests(unittest.TestCase):
    """Verify the public output contract matches spec §7.1 exactly."""

    def test_output_has_correct_top_level_keys(self) -> None:
        module = _make_module()
        result = module.detect_faces(_make_input())

        self.assertEqual(
            set(result), {"frame_id", "camera_id", "timestamp_ms", "detections"}
        )

    def test_output_preserves_input_metadata(self) -> None:
        module = _make_module()
        result = module.detect_faces(
            _make_input(frame_id="frame_0042", camera_id="cam-x", timestamp_ms=9999)
        )

        self.assertEqual(result["frame_id"], "frame_0042")
        self.assertEqual(result["camera_id"], "cam-x")
        self.assertEqual(result["timestamp_ms"], 9999)

    def test_detections_contain_face_bbox_and_landmarks(self) -> None:
        module = _make_module()
        result = module.detect_faces(_make_input())

        self.assertGreater(len(result["detections"]), 0)
        face = result["detections"][0]
        self.assertEqual(set(face), {"face_bbox", "landmarks"})

    def test_face_bbox_has_correct_fields(self) -> None:
        module = _make_module()
        result = module.detect_faces(_make_input())

        bbox = result["detections"][0]["face_bbox"]
        self.assertEqual(set(bbox), {"x", "y", "width", "height"})
        self.assertGreater(bbox["width"], 0)
        self.assertGreater(bbox["height"], 0)

    def test_landmarks_have_all_five_canonical_points(self) -> None:
        module = _make_module()
        result = module.detect_faces(_make_input())

        landmarks = result["detections"][0]["landmarks"]
        expected_keys = {"left_eye", "right_eye", "nose", "mouth_left", "mouth_right"}
        self.assertEqual(set(landmarks), expected_keys)

        for key in expected_keys:
            pt = landmarks[key]
            self.assertEqual(set(pt), {"x", "y"})
            self.assertIsInstance(pt["x"], int)
            self.assertIsInstance(pt["y"], int)


class RoiLocalOutputTests(unittest.TestCase):
    """Verify face detection output coordinates are ROI-local (no full-frame projection)."""

    def test_face_bbox_is_roi_local(self) -> None:
        module = _make_module()
        stub = StubFaceDetectorEngine()
        roi_image = np.zeros((200, 150, 3), dtype=np.uint8)
        raw = stub.detect(roi_image)[0]

        result = module.detect_faces(_make_input(roi_w=150, roi_h=200))
        bbox = result["detections"][0]["face_bbox"]

        # Output coordinates must equal the raw ROI-local coordinates — no offset added
        self.assertEqual(bbox["x"], raw.bbox["x"])
        self.assertEqual(bbox["y"], raw.bbox["y"])
        self.assertEqual(bbox["width"], raw.bbox["width"])
        self.assertEqual(bbox["height"], raw.bbox["height"])

    def test_landmarks_are_roi_local(self) -> None:
        module = _make_module()
        stub = StubFaceDetectorEngine()
        roi_image = np.zeros((240, 120, 3), dtype=np.uint8)
        raw = stub.detect(roi_image)[0]

        result = module.detect_faces(_make_input(roi_w=120, roi_h=240))
        landmarks = result["detections"][0]["landmarks"]

        for i, key in enumerate(
            ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")
        ):
            self.assertEqual(landmarks[key]["x"], raw.landmarks[i]["x"])
            self.assertEqual(landmarks[key]["y"], raw.landmarks[i]["y"])

    def test_output_coordinates_within_roi_bounds(self) -> None:
        roi_w, roi_h = 150, 300
        module = _make_module()
        result = module.detect_faces(_make_input(roi_w=roi_w, roi_h=roi_h))

        bbox = result["detections"][0]["face_bbox"]
        self.assertGreaterEqual(bbox["x"], 0)
        self.assertGreaterEqual(bbox["y"], 0)
        self.assertLessEqual(bbox["x"] + bbox["width"], roi_w)
        self.assertLessEqual(bbox["y"] + bbox["height"], roi_h)


class DeterminismTests(unittest.TestCase):
    """Stub must produce identical output every run for the same input."""

    def test_same_input_produces_identical_output(self) -> None:
        module = _make_module()
        inp = _make_input()

        result1 = module.detect_faces(inp)
        result2 = module.detect_faces(inp)

        self.assertEqual(result1, result2)


class ErrorHandlingTests(unittest.TestCase):
    """Spec §10: all errors return valid output with empty detections."""

    def test_missing_camera_id_returns_empty_detections(self) -> None:
        module = _make_module()
        bad_input: FaceDetectionInput = {
            "frame_id": "frame_0001",
            "camera_id": "",
            "timestamp_ms": 100,
            "roi_image": np.zeros((10, 10, 3), dtype=np.uint8),
        }
        result = module.detect_faces(bad_input)

        self.assertEqual(set(result), {"frame_id", "camera_id", "timestamp_ms", "detections"})
        self.assertEqual(result["detections"], [])

    def test_wrong_dtype_roi_returns_empty_detections(self) -> None:
        module = _make_module()
        bad_input: FaceDetectionInput = {
            "frame_id": "frame_0001",
            "camera_id": "cam-a",
            "timestamp_ms": 100,
            "roi_image": np.zeros((10, 10, 3), dtype=np.float32),
        }
        result = module.detect_faces(bad_input)

        self.assertEqual(result["detections"], [])

    def test_none_roi_image_returns_empty_detections(self) -> None:
        module = _make_module()
        bad_input: FaceDetectionInput = {
            "frame_id": "frame_0001",
            "camera_id": "cam-a",
            "timestamp_ms": 100,
            "roi_image": None,
        }
        result = module.detect_faces(bad_input)

        self.assertEqual(result["detections"], [])


class PostprocessorFilteringTests(unittest.TestCase):
    """Postprocessor filters detections below confidence threshold."""

    def test_low_confidence_detections_are_rejected(self) -> None:
        # Set threshold higher than stub confidence (0.95)
        module = _make_module(confidence_threshold=0.99)
        result = module.detect_faces(_make_input())

        self.assertEqual(result["detections"], [])

    def test_detections_above_threshold_are_accepted(self) -> None:
        # Set threshold below stub confidence (0.95)
        module = _make_module(confidence_threshold=0.5)
        result = module.detect_faces(_make_input())

        self.assertGreater(len(result["detections"]), 0)


class StubEngineTests(unittest.TestCase):
    """Verify the stub engine produces valid RawFaceDetection objects."""

    def test_stub_returns_one_detection(self) -> None:
        stub = StubFaceDetectorEngine()
        roi = np.zeros((200, 100, 3), dtype=np.uint8)
        detections = stub.detect(roi)

        self.assertEqual(len(detections), 1)

    def test_stub_detection_has_five_landmarks(self) -> None:
        stub = StubFaceDetectorEngine()
        roi = np.zeros((200, 100, 3), dtype=np.uint8)
        det = stub.detect(roi)[0]

        self.assertEqual(len(det.landmarks), 5)

    def test_stub_coordinates_are_within_roi(self) -> None:
        stub = StubFaceDetectorEngine()
        roi_w, roi_h = 150, 300
        roi = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
        det = stub.detect(roi)[0]

        self.assertGreaterEqual(det.bbox["x"], 0)
        self.assertGreaterEqual(det.bbox["y"], 0)
        self.assertLessEqual(det.bbox["x"] + det.bbox["width"], roi_w)
        self.assertLessEqual(det.bbox["y"] + det.bbox["height"], roi_h)

        for lm in det.landmarks:
            self.assertGreaterEqual(lm["x"], 0)
            self.assertGreaterEqual(lm["y"], 0)
            self.assertLessEqual(lm["x"], roi_w)
            self.assertLessEqual(lm["y"], roi_h)

    def test_stub_is_deterministic(self) -> None:
        stub = StubFaceDetectorEngine()
        roi = np.zeros((200, 100, 3), dtype=np.uint8)

        d1 = stub.detect(roi)[0]
        d2 = stub.detect(roi)[0]

        self.assertEqual(d1.bbox, d2.bbox)
        self.assertEqual(d1.landmarks, d2.landmarks)
        self.assertEqual(d1.confidence, d2.confidence)


class ImageContractValidationTests(unittest.TestCase):
    """Verify FaceDetectionInputValidator enforces shared Image struct contract."""

    def test_accepts_valid_image_struct(self) -> None:
        module = _make_module()
        result = module.detect_faces(_make_input())
        self.assertGreater(len(result["detections"]), 0)

    def test_rejects_missing_data_field(self) -> None:
        module = _make_module()
        bad_roi: FaceDetectionInput = {
            "frame_id": "f1",
            "camera_id": "cam-a",
            "timestamp_ms": 100,
            "roi_image": {"data": None, "width": 100, "height": 200,
                          "color_format": "RGB", "layout": "HWC",
                          "dtype": "uint8", "value_range": "[0, 255]"},
        }
        result = module.detect_faces(bad_roi)
        self.assertEqual(result["detections"], [])

    def test_rejects_wrong_color_format(self) -> None:
        module = _make_module()
        roi = _make_roi_image(100, 200)
        roi["color_format"] = "BGR"
        result = module.detect_faces(
            {"frame_id": "f1", "camera_id": "cam-a", "timestamp_ms": 100, "roi_image": roi}
        )
        self.assertEqual(result["detections"], [])

    def test_rejects_wrong_layout(self) -> None:
        module = _make_module()
        roi = _make_roi_image(100, 200)
        roi["layout"] = "CHW"
        result = module.detect_faces(
            {"frame_id": "f1", "camera_id": "cam-a", "timestamp_ms": 100, "roi_image": roi}
        )
        self.assertEqual(result["detections"], [])

    def test_rejects_wrong_dtype(self) -> None:
        module = _make_module()
        roi = _make_roi_image(100, 200)
        roi["dtype"] = "float32"
        result = module.detect_faces(
            {"frame_id": "f1", "camera_id": "cam-a", "timestamp_ms": 100, "roi_image": roi}
        )
        self.assertEqual(result["detections"], [])

    def test_rejects_wrong_value_range(self) -> None:
        module = _make_module()
        roi = _make_roi_image(100, 200)
        roi["value_range"] = "[0, 1]"
        result = module.detect_faces(
            {"frame_id": "f1", "camera_id": "cam-a", "timestamp_ms": 100, "roi_image": roi}
        )
        self.assertEqual(result["detections"], [])

    def test_rejects_zero_width(self) -> None:
        module = _make_module()
        roi = _make_roi_image(100, 200)
        roi["width"] = 0
        result = module.detect_faces(
            {"frame_id": "f1", "camera_id": "cam-a", "timestamp_ms": 100, "roi_image": roi}
        )
        self.assertEqual(result["detections"], [])

    def test_rejects_zero_height(self) -> None:
        module = _make_module()
        roi = _make_roi_image(100, 200)
        roi["height"] = 0
        result = module.detect_faces(
            {"frame_id": "f1", "camera_id": "cam-a", "timestamp_ms": 100, "roi_image": roi}
        )
        self.assertEqual(result["detections"], [])

    def test_no_roi_bbox_frame_required(self) -> None:
        """FaceDetectionInput must not require roi_bbox_frame — spec §8.5.3."""
        module = _make_module()
        inp = _make_input()
        self.assertNotIn("roi_bbox_frame", inp)
        result = module.detect_faces(inp)
        self.assertGreater(len(result["detections"]), 0)

    def test_engine_receives_ndarray(self) -> None:
        """FaceDetectionModule must extract Image.data before calling the engine."""
        received: list[object] = []

        class CapturingEngine:
            def detect(self, roi_image: np.ndarray) -> list[RawFaceDetection]:  # type: ignore[override]
                received.append(roi_image)
                return []

        module = FaceDetectionModule(
            config=FaceDetectionConfig(),
            detector_engine=CapturingEngine(),  # type: ignore[arg-type]
        )
        module.detect_faces(_make_input())

        self.assertEqual(len(received), 1)
        self.assertIsInstance(received[0], np.ndarray)


if __name__ == "__main__":
    unittest.main()
