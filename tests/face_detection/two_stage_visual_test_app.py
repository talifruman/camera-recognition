"""
Two-Stage Visual Test App
=========================
Image pipeline:  Real Object Detection  →  (stub) Face Detection

1. Load image.
2. Run real Object Detection (existing implementation).
3. Draw yellow person bounding boxes exactly as the original pipeline.
4. If at least one person detected, for each person ROI:
   a. Crop the ROI from the annotated image.
   b. Run Face Detection through the architecturally correct module.
   c. Draw green face bounding box and green landmarks on the image.
5. Save the final combined image.

Face Detection uses a STUB detector engine internally (no real AI model).
The module architecture, data contracts, and component boundaries match
the Face Detection markdown specification exactly.
"""

from __future__ import annotations

import sys
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Path wiring — make both src/ and sibling test dirs importable
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _PROJECT_ROOT / "src"
_OD_TESTS_DIR = _PROJECT_ROOT / "tests" / "object_detection"
_FD_TESTS_DIR = Path(__file__).resolve().parent

for _p in (_SRC_DIR, _OD_TESTS_DIR, _FD_TESTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Reuse Object Detection pipeline utilities
from object_detection_stub import (  # type: ignore[import-not-found]
    BoundingBox as ODBoundingBox,
    FrameMetadata,
)
from visual_test_app import (  # type: ignore[import-not-found]
    IMAGE_EXTENSIONS,
    build_run_output_root,
    draw_person_boxes,
    get_detector_process,
    iter_image_files,
)

# Face Detection module (architecturally correct, spec-compliant)
from image_processing.face_detection import (  # type: ignore[import-not-found]
    FaceDetectionConfig,
    FaceDetectionInput,
    FaceDetectionModule,
    FaceDetectionOutput,
)
from face_detection_stub_engine import StubFaceDetectorEngine  # type: ignore[import-not-found]

FRAME_COUNTER = count(1)

GREEN = (0, 255, 0)
LANDMARK_RADIUS = 3


# ---------------------------------------------------------------------------
# Face-detection visualization helpers
# ---------------------------------------------------------------------------


def draw_face_detections(
    image: Image.Image,
    face_output: FaceDetectionOutput,
) -> Image.Image:
    """Draw green face bounding boxes and green landmark dots onto *image*."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for face in face_output["detections"]:
        bbox = face["face_bbox_frame"]
        left = bbox["x"]
        top = bbox["y"]
        right = left + bbox["width"]
        bottom = top + bbox["height"]
        draw.rectangle((left, top, right, bottom), outline=GREEN, width=2)

        landmarks = face["landmarks"]
        for key in ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right"):
            pt = landmarks[key]
            x, y = pt["x"], pt["y"]
            draw.ellipse(
                (x - LANDMARK_RADIUS, y - LANDMARK_RADIUS,
                 x + LANDMARK_RADIUS, y + LANDMARK_RADIUS),
                fill=GREEN,
            )

    return annotated


# ---------------------------------------------------------------------------
# ROI cropping
# ---------------------------------------------------------------------------


def crop_person_roi(image_array: np.ndarray, person_bbox: dict) -> np.ndarray:
    """Crop the person bounding-box region from *image_array*."""
    x = person_bbox["x"]
    y = person_bbox["y"]
    w = person_bbox["width"]
    h = person_bbox["height"]
    return image_array[y : y + h, x : x + w].copy()


# ---------------------------------------------------------------------------
# Two-stage pipeline
# ---------------------------------------------------------------------------


def process_images(
    input_root: Path,
    output_root: Path,
    od_detector_mode: str = "real",
) -> int:
    """Run the two-stage Object Detection → Face Detection pipeline."""
    processed_count = 0
    input_root = Path(input_root)
    output_root = Path(output_root)

    # Stage 1 engine: real or stub Object Detection
    od_process: Callable[[Any, Any], Any] = get_detector_process(od_detector_mode)

    # Stage 2 engine: Face Detection with STUB detector engine
    face_module = FaceDetectionModule(
        config=FaceDetectionConfig(),
        detector_engine=StubFaceDetectorEngine(),
    )

    run_output_root = build_run_output_root(output_root)

    for image_path in iter_image_files(input_root):
        try:
            with Image.open(image_path) as opened_image:
                rgb_image = opened_image.convert("RGB")
                width, height = rgb_image.size
                metadata: FrameMetadata = {
                    "camera_id": "visual-test-camera",
                    "frame_id": next(FRAME_COUNTER),
                    "width": width,
                    "height": height,
                }

            # Stage 1: Object Detection
            image_array = np.array(rgb_image)
            od_result = od_process(image_array, metadata)

            # Draw yellow person boxes (identical to existing pipeline)
            rendered_image = draw_person_boxes(rgb_image, od_result["persons"])

            # Stage 2: Face Detection (only when persons detected)
            if od_result["person_detected"]:
                annotated_array = np.array(rendered_image)

                for person_bbox in od_result["persons"]:
                    roi_image = crop_person_roi(annotated_array, person_bbox)
                    if roi_image.size == 0:
                        continue

                    face_input: FaceDetectionInput = {
                        "frame_id": metadata["frame_id"],
                        "camera_id": metadata["camera_id"],
                        "timestamp_ms": 0,
                        "roi_image": roi_image,
                        "roi_bbox_frame": {
                            "x": person_bbox["x"],
                            "y": person_bbox["y"],
                            "width": person_bbox["width"],
                            "height": person_bbox["height"],
                        },
                    }

                    face_output = face_module.detect_faces(face_input)
                    rendered_image = draw_face_detections(rendered_image, face_output)

        except (OSError, ValueError, UnidentifiedImageError):
            continue

        output_path = run_output_root / image_path.relative_to(input_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rendered_image.save(output_path)
        processed_count += 1

    return processed_count


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3:
        raise ValueError(
            "Expected 3 arguments: input_root output_root od_detector_mode"
        )

    input_root = Path(args[0])
    output_root = Path(args[1])
    od_detector_mode = args[2]
    process_images(input_root, output_root, od_detector_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
