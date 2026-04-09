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
    Point,
    RawFaceDetection,
)
from face_detection_stub_engine import StubFaceDetectorEngine  # type: ignore[import-not-found]


def _make_input(
    roi_w: int = 100,
    roi_h: int = 200,
    roi_x: int = 50,
    roi_y: int = 30,
    frame_id: int = 1,
    camera_id: str = "cam-a",
    timestamp_ms: int = 1000,
) -> FaceDetectionInput:
    return {
        "frame_id": frame_id,
        "camera_id": camera_id,
        "timestamp_ms": timestamp_ms,
        "roi_image": np.zeros((roi_h, roi_w, 3), dtype=np.uint8),
        "roi_bbox_frame": {
            "x": roi_x,
            "y": roi_y,
            "width": roi_w,
            "height": roi_h,
        },
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
            _make_input(frame_id=42, camera_id="cam-x", timestamp_ms=9999)
        )

        self.assertEqual(result["frame_id"], 42)
        self.assertEqual(result["camera_id"], "cam-x")
        self.assertEqual(result["timestamp_ms"], 9999)

    def test_detections_contain_face_bbox_frame_and_landmarks(self) -> None:
        module = _make_module()
        result = module.detect_faces(_make_input())

        self.assertGreater(len(result["detections"]), 0)
        face = result["detections"][0]
        self.assertEqual(set(face), {"face_bbox_frame", "landmarks"})

    def test_face_bbox_frame_has_correct_fields(self) -> None:
        module = _make_module()
        result = module.detect_faces(_make_input())

        bbox = result["detections"][0]["face_bbox_frame"]
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


class CoordinateProjectionTests(unittest.TestCase):
    """Verify face coordinates are projected from ROI to frame space."""

    def test_face_bbox_is_offset_by_roi_origin(self) -> None:
        roi_x, roi_y = 100, 200
        module = _make_module()
        result = module.detect_faces(
            _make_input(roi_x=roi_x, roi_y=roi_y, roi_w=150, roi_h=300)
        )

        bbox = result["detections"][0]["face_bbox_frame"]
        # bbox.x must be >= roi_x and bbox.y >= roi_y (projected from ROI)
        self.assertGreaterEqual(bbox["x"], roi_x)
        self.assertGreaterEqual(bbox["y"], roi_y)

    def test_landmarks_are_offset_by_roi_origin(self) -> None:
        roi_x, roi_y = 50, 80
        module = _make_module()
        result = module.detect_faces(
            _make_input(roi_x=roi_x, roi_y=roi_y, roi_w=120, roi_h=240)
        )

        landmarks = result["detections"][0]["landmarks"]
        for key in ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right"):
            pt = landmarks[key]
            self.assertGreaterEqual(pt["x"], roi_x)
            self.assertGreaterEqual(pt["y"], roi_y)

    def test_projection_math_is_correct(self) -> None:
        """Run with known ROI offset; verify exact projection arithmetic."""
        roi_x, roi_y = 60, 40
        roi_w, roi_h = 100, 200
        module = _make_module()

        # Get ROI-local detection from stub directly
        stub = StubFaceDetectorEngine()
        roi_image = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
        raw = stub.detect(roi_image)[0]

        result = module.detect_faces(
            _make_input(roi_x=roi_x, roi_y=roi_y, roi_w=roi_w, roi_h=roi_h)
        )
        face = result["detections"][0]

        # Face bbox: projected = raw + roi offset
        self.assertEqual(face["face_bbox_frame"]["x"], raw.bbox["x"] + roi_x)
        self.assertEqual(face["face_bbox_frame"]["y"], raw.bbox["y"] + roi_y)
        self.assertEqual(face["face_bbox_frame"]["width"], raw.bbox["width"])
        self.assertEqual(face["face_bbox_frame"]["height"], raw.bbox["height"])

        # Landmarks: same offset logic
        for i, key in enumerate(
            ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")
        ):
            expected_x = raw.landmarks[i]["x"] + roi_x
            expected_y = raw.landmarks[i]["y"] + roi_y
            self.assertEqual(face["landmarks"][key]["x"], expected_x)
            self.assertEqual(face["landmarks"][key]["y"], expected_y)


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
            "frame_id": 1,
            "camera_id": "",
            "timestamp_ms": 100,
            "roi_image": np.zeros((10, 10, 3), dtype=np.uint8),
            "roi_bbox_frame": {"x": 0, "y": 0, "width": 10, "height": 10},
        }
        result = module.detect_faces(bad_input)

        self.assertEqual(set(result), {"frame_id", "camera_id", "timestamp_ms", "detections"})
        self.assertEqual(result["detections"], [])

    def test_zero_size_roi_returns_empty_detections(self) -> None:
        module = _make_module()
        bad_input: FaceDetectionInput = {
            "frame_id": 1,
            "camera_id": "cam-a",
            "timestamp_ms": 100,
            "roi_image": np.zeros((10, 10, 3), dtype=np.uint8),
            "roi_bbox_frame": {"x": 0, "y": 0, "width": 0, "height": 10},
        }
        result = module.detect_faces(bad_input)

        self.assertEqual(result["detections"], [])

    def test_wrong_dtype_roi_returns_empty_detections(self) -> None:
        module = _make_module()
        bad_input: FaceDetectionInput = {
            "frame_id": 1,
            "camera_id": "cam-a",
            "timestamp_ms": 100,
            "roi_image": np.zeros((10, 10, 3), dtype=np.float32),
            "roi_bbox_frame": {"x": 0, "y": 0, "width": 10, "height": 10},
        }
        result = module.detect_faces(bad_input)

        self.assertEqual(result["detections"], [])

    def test_none_roi_image_returns_empty_detections(self) -> None:
        module = _make_module()
        bad_input: FaceDetectionInput = {
            "frame_id": 1,
            "camera_id": "cam-a",
            "timestamp_ms": 100,
            "roi_image": None,
            "roi_bbox_frame": {"x": 0, "y": 0, "width": 10, "height": 10},
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


if __name__ == "__main__":
    unittest.main()
