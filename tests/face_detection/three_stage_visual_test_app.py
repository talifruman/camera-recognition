"""
Three-Stage Visual Test App
===========================
Image pipeline:  Real Object Detection  →  Face Detection  →  Real Face Recognition

Extends the two-stage OD → FD pipeline with a Face Recognition stage backed
by real ArcFace embeddings loaded from a pre-built gallery on disk.

Gallery loading
---------------
At startup, FaceGalleryLoaderModule reads real .npy embeddings from
``data/generated_face_gallery_real/`` (built by
``build_real_face_gallery_from_images.py``).  The gallery is loaded once and
held in memory for the entire run.

Recognition pass
----------------
OD → FD → FR on every image.  For each detected face, run Face Recognition
against the loaded gallery using ArcFaceEmbeddingEngine and produce an
annotated output image.

Visual output per face
----------------------
- YELLOW rectangle   — face bounding box
- RED filled circles — 5 canonical facial landmarks
- GREEN text         — "person: <id>"  when person_found = True
- RED text           — "person: UNKNOWN"  when person_found = False

Text output (stdout)
--------------------
Image: <filename>
Faces detected: <N>
  Face #<i>:
    person_found: <True/False>
    person_id: <value or empty>

Gallery is loaded from ``data/generated_face_gallery_real/`` by default.
An alternative gallery root may be passed as the 5th CLI argument.
"""

from __future__ import annotations

import sys
from itertools import count
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Path wiring — mirror two_stage_visual_test_app conventions
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _PROJECT_ROOT / "src"
_OD_TESTS_DIR = _PROJECT_ROOT / "tests" / "object_detection"
_FD_TESTS_DIR = Path(__file__).resolve().parent

for _p in (_SRC_DIR, _OD_TESTS_DIR, _FD_TESTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Object Detection pipeline utilities (reused from existing app)
from object_detection_stub import FrameMetadata  # type: ignore[import-not-found]
from visual_test_app import (  # type: ignore[import-not-found]
    build_run_output_root,
    draw_person_boxes,
    get_detector_process,
    iter_image_files,
)

# Face Detection module
from image_processing.face_detection import (  # type: ignore[import-not-found]
    FaceDetectionConfig,
    FaceDetectionInput,
    FaceDetectionModule,
    FaceDetectionOutput,
    SCRFDFaceDetector,
)
from face_detection_stub_engine import StubFaceDetectorEngine  # type: ignore[import-not-found]

# Face Recognition module
from image_processing.face_recognition import (  # type: ignore[import-not-found]
    ArcFaceEmbeddingEngine,
    FaceRecognitionConfig,
    FaceRecognitionInput,
    FaceRecognitionModule,
    FaceRecognitionOutput,
    GalleryEntry,
)

# Face Gallery Loader module
from image_processing.face_gallery_loader import (  # type: ignore[import-not-found]
    FaceGalleryLoaderConfig,
    FaceGalleryLoaderModule,
    NpyEmbeddingFileReader,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAME_COUNTER = count(1)

# cv2 color tuples applied to RGB arrays — expressed as (R, G, B) to match
# the in-memory channel order.  cv2 writes the tuple values to channels
# 0/1/2 respectively, so pass RGB values when the array is RGB.
_YELLOW = (255, 255, 0)   # face bounding box
_RED    = (255, 0,   0)   # landmarks + UNKNOWN label
_GREEN  = (0,   200, 0)   # recognized-person label

_LANDMARK_RADIUS = 3
_FONT             = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE       = 0.7
_FONT_THICKNESS   = 2

_VALID_FD_MODES = ("stub", "real")
_REAL_GALLERY_ROOT = _PROJECT_ROOT / "data" / "generated_face_gallery_real"


# ---------------------------------------------------------------------------
# Engine / module builder helpers
# ---------------------------------------------------------------------------

def _build_face_engine(mode: str):
    """Return the FaceDetectorEngine for *mode* ('stub' or 'real')."""
    if mode == "stub":
        return StubFaceDetectorEngine()
    if mode == "real":
        return SCRFDFaceDetector(model_dir=_PROJECT_ROOT / "models" / "face_detection")
    raise ValueError(f"fd_detector_mode must be one of: {', '.join(_VALID_FD_MODES)}")


def _build_face_module(mode: str) -> FaceDetectionModule:
    return FaceDetectionModule(
        config=FaceDetectionConfig(),
        detector_engine=_build_face_engine(mode),
    )


# ---------------------------------------------------------------------------
# ROI / landmark helpers
# ---------------------------------------------------------------------------

def crop_person_roi(image_array: np.ndarray, person_bbox: dict) -> np.ndarray:
    """Crop the person bounding-box region from *image_array*."""
    x, y, w, h = (
        person_bbox["x"], person_bbox["y"],
        person_bbox["width"], person_bbox["height"],
    )
    return image_array[y : y + h, x : x + w].copy()


def crop_face_roi(frame_array: np.ndarray, face_bbox: dict) -> np.ndarray:
    """Crop a face bounding-box region from *frame_array* (frame coords)."""
    x, y, w, h = (
        face_bbox["x"], face_bbox["y"],
        face_bbox["width"], face_bbox["height"],
    )
    return frame_array[y : y + h, x : x + w].copy()


def convert_landmarks_to_roi(landmarks: dict, face_bbox: dict) -> dict:
    """
    Subtract the face bbox origin so landmark coords become ROI-relative.

    Required because FaceRecognitionInputValidator checks that every
    landmark coordinate falls within the face_roi_image extent.
    """
    ox, oy = face_bbox["x"], face_bbox["y"]
    return {
        key: {"x": landmarks[key]["x"] - ox, "y": landmarks[key]["y"] - oy}
        for key in ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")
    }


# ---------------------------------------------------------------------------
# Annotation drawing (cv2 on RGB ndarray)
# ---------------------------------------------------------------------------

def draw_recognition_output(
    image_array: np.ndarray,
    detections: list,
    recognition_results: list[FaceRecognitionOutput],
) -> np.ndarray:
    """
    Draw face bboxes, landmarks, and recognition labels onto *image_array*.

    Returns a new annotated copy; does not mutate *image_array*.

    Parameters
    ----------
    image_array:
        RGB HWC uint8 ndarray (origin frame, may already have person boxes).
    detections:
        list[DetectedFace] from FaceDetectionOutput['detections'].
    recognition_results:
        One FaceRecognitionOutput per entry in *detections*, same order.
    """
    annotated = image_array.copy()

    for face, result in zip(detections, recognition_results):
        bbox = face["face_bbox_frame"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]

        # 1. Yellow bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), _YELLOW, 2)

        # 2. Red circles at each of the 5 canonical landmarks (frame coords)
        for key in ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right"):
            pt = face["landmarks"][key]
            cv2.circle(annotated, (pt["x"], pt["y"]), _LANDMARK_RADIUS, _RED, -1)

        # 3. Recognition label above the bounding box
        if result["person_found"]:
            label = f"person: {result['person_id']}"
            color = _GREEN
        else:
            label = "person: UNKNOWN"
            color = _RED

        label_x = x
        label_y = max(y - 10, 14)  # keep label inside frame top edge
        cv2.putText(
            annotated,
            label,
            (label_x, label_y),
            _FONT,
            _FONT_SCALE,
            color,
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated


# ---------------------------------------------------------------------------
# Gallery loading helper
# ---------------------------------------------------------------------------

def _load_real_gallery(gallery_root: Path) -> list[GalleryEntry]:
    """
    Load real .npy embeddings from *gallery_root* using NpyEmbeddingFileReader.

    Returns a list of GalleryEntry items, one per embedding file found.
    Prints a summary of loaded persons to stdout.

    Raises GalleryPathValidationError or GalleryLoadError on failure.
    """
    config = FaceGalleryLoaderConfig()
    reader = NpyEmbeddingFileReader(
        embedding_file_extension=config.embedding_file_extension,
        expected_embedding_dim=config.expected_embedding_dim,
        expected_dtype=config.expected_dtype,
    )
    loader = FaceGalleryLoaderModule(config=config, reader=reader)
    loader.load_gallery(str(gallery_root))
    entries = loader.get_all_embeddings()
    print(f"Gallery loaded from: {gallery_root}")
    print(f"  Persons : {len(loader.get_person_ids())}")
    print(f"  Embeddings: {len(entries)}")
    for person_id in loader.get_person_ids():
        print(f"  - {person_id}")
    return entries


# ---------------------------------------------------------------------------
# Full OD → FD → FR pipeline
# ---------------------------------------------------------------------------

def process_images(
    input_root: Path,
    output_root: Path,
    od_detector_mode: str = "real",
    fd_detector_mode: str | None = None,
    gallery_root: Path | None = None,
) -> int:
    """
    Run the three-stage Object Detection → Face Detection → Face Recognition pipeline.

    Parameters
    ----------
    input_root:
        Root directory containing source images.
    output_root:
        Root directory for timestamped output runs.
    od_detector_mode:
        ``"real"`` or ``"stub"`` — selects the Object Detection engine.
    fd_detector_mode:
        ``"real"`` or ``"stub"`` — selects the Face Detection engine.
        Defaults to *od_detector_mode* when ``None``.
    gallery_root:
        Path to the real gallery directory.  Defaults to
        ``data/generated_face_gallery_real/`` when ``None``.
    """
    resolved_fd_mode = fd_detector_mode if fd_detector_mode is not None else od_detector_mode
    resolved_gallery_root = Path(gallery_root) if gallery_root is not None else _REAL_GALLERY_ROOT
    input_root = Path(input_root)
    output_root = Path(output_root)

    od_process: Callable = get_detector_process(od_detector_mode)
    face_det_module = _build_face_module(resolved_fd_mode)

    # ---- Load real gallery at startup -----------------------------------
    print("=== Loading real face gallery ===")
    gallery_entries = _load_real_gallery(resolved_gallery_root)

    if not gallery_entries:
        print("WARNING: No gallery entries loaded — recognition will return UNKNOWN for all faces.")

    # ---- Initialize recognition module (once, before frame loop) -------
    print("\n=== Initializing Face Recognition (ArcFace) ===")

    face_rec_module = FaceRecognitionModule(
        config=FaceRecognitionConfig(),
        embedding_engine=ArcFaceEmbeddingEngine(),
        gallery_entries=gallery_entries,
    )

    print("=== Starting recognition pipeline ===")

    run_output_root = build_run_output_root(output_root)
    processed_count = 0

    for image_path in iter_image_files(input_root):
        try:
            with Image.open(image_path) as opened_image:
                rgb_image = opened_image.convert("RGB")
                width, height = rgb_image.size
                metadata: FrameMetadata = {
                    "camera_id": "visual-test-camera",
                    "frame_id": f"frame_{next(FRAME_COUNTER):04d}",
                    "width": width,
                    "height": height,
                }

            image_array = np.array(rgb_image)
            od_result = od_process(image_array, metadata)

            # Draw yellow person boxes via existing PIL helper
            rendered_pil = draw_person_boxes(rgb_image, od_result["persons"])
            rendered_array = np.array(rendered_pil)

            all_detections: list = []
            all_results: list[FaceRecognitionOutput] = []

            if od_result["person_detected"]:
                for person_bbox in od_result["persons"]:
                    # Crop person ROI from the unannotated array so face
                    # detection runs on clean pixel data (no drawn boxes).
                    roi_image = crop_person_roi(image_array, person_bbox)
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
                    face_output = face_det_module.detect_faces(face_input)

                    for detected_face in face_output["detections"]:
                        face_roi = crop_face_roi(image_array, detected_face["face_bbox_frame"])
                        if face_roi.size == 0:
                            continue

                        roi_landmarks = convert_landmarks_to_roi(
                            detected_face["landmarks"], detected_face["face_bbox_frame"]
                        )

                        fr_input: FaceRecognitionInput = {
                            "frame_id": metadata["frame_id"],
                            "camera_id": metadata["camera_id"],
                            "timestamp_ms": 0,
                            "face_roi_image": face_roi,
                            "landmarks": roi_landmarks,
                        }
                        fr_output = face_rec_module.recognize_face(fr_input)

                        all_detections.append(detected_face)
                        all_results.append(fr_output)

            # ---- Text log -----------------------------------------------
            print(f"\nImage: {image_path.name}")
            print(f"Faces detected: {len(all_detections)}")
            for i, (_, res) in enumerate(zip(all_detections, all_results), start=1):
                print(f"  Face #{i}:")
                print(f"    person_found: {res['person_found']}")
                print(f"    person_id: {res['person_id'] if res['person_found'] else ''}")

            # ---- Annotate and save --------------------------------------
            final_array = draw_recognition_output(rendered_array, all_detections, all_results)
            final_pil = Image.fromarray(final_array)

        except (OSError, ValueError, UnidentifiedImageError):
            continue

        output_path = run_output_root / image_path.relative_to(input_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_pil.save(output_path)
        processed_count += 1

    print(f"\n=== Done: {processed_count} image(s) saved to {run_output_root} ===")
    return processed_count


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) not in (3, 4, 5):
        raise ValueError(
            "Expected 3-5 arguments: "
            "input_root output_root od_detector_mode [fd_detector_mode] [gallery_root]"
        )

    input_root = Path(args[0])
    output_root = Path(args[1])
    od_detector_mode = args[2]
    fd_detector_mode = args[3] if len(args) >= 4 else None
    gallery_root = Path(args[4]) if len(args) == 5 else None

    return process_images(input_root, output_root, od_detector_mode, fd_detector_mode, gallery_root)


if __name__ == "__main__":
    sys.exit(main())
