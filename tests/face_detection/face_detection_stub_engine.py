"""
STUB Face Detector Engine
=========================
Temporary stub implementation of FaceDetectorEngine for testing and demo
purposes.  Returns deterministic, image-size-relative face detections so
the output is stable across different image sizes.

This file is the ONLY place that contains stub-specific logic.  It sits
behind the FaceDetectorEngine protocol and can be replaced by a real
detector (e.g. SCRFDFaceDetector) without changing the outer module API,
pipeline flow, or result format.

All coordinates are computed relative to the ROI image dimensions so
results are consistent regardless of input resolution.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_processing.face_detection.module import (  # type: ignore[import-not-found]
    BoundingBox,
    Point,
    RawFaceDetection,
)


class StubFaceDetectorEngine:
    """
    STUB — deterministic face detector for testing / demo only.

    Returns exactly one face detection per invocation.  Coordinates are
    derived from the ROI image dimensions so the output is resolution-
    independent and deterministic.

    Replace this class with a real FaceDetectorEngine implementation
    (e.g. SCRFDFaceDetector) when a trained model is available.
    """

    # STUB: fixed confidence returned for every detection
    _STUB_CONFIDENCE: float = 0.95

    def detect(self, roi_image: np.ndarray) -> list[RawFaceDetection]:
        # STUB: derive face geometry from ROI dimensions
        roi_h, roi_w = roi_image.shape[:2]

        # STUB: face bounding box — centered horizontally, upper third
        face_w = max(1, int(roi_w * 0.30))
        face_h = max(1, int(roi_h * 0.35))
        face_x = max(0, (roi_w - face_w) // 2)
        face_y = max(0, int(roi_h * 0.08))

        face_bbox: BoundingBox = {
            "x": face_x,
            "y": face_y,
            "width": face_w,
            "height": face_h,
        }

        # STUB: five canonical landmarks positioned relative to face bbox
        # Landmark positions approximate a frontal face layout
        cx = face_x + face_w // 2
        landmarks: list[Point] = [
            # left_eye — upper-left quadrant of face
            {"x": face_x + int(face_w * 0.30), "y": face_y + int(face_h * 0.35)},
            # right_eye — upper-right quadrant of face
            {"x": face_x + int(face_w * 0.70), "y": face_y + int(face_h * 0.35)},
            # nose — center of face
            {"x": cx, "y": face_y + int(face_h * 0.55)},
            # mouth_left — lower-left
            {"x": face_x + int(face_w * 0.35), "y": face_y + int(face_h * 0.75)},
            # mouth_right — lower-right
            {"x": face_x + int(face_w * 0.65), "y": face_y + int(face_h * 0.75)},
        ]

        return [
            RawFaceDetection(
                bbox=face_bbox,
                confidence=self._STUB_CONFIDENCE,
                landmarks=landmarks,
            )
        ]
