"""
build_real_face_gallery_from_images.py

Phase 2 gallery builder — generates real ArcFace embeddings.

Builds a FaceGalleryLoader-compatible embeddings gallery from an image
folder by running the real face pipeline:

    image → face detection → landmarks → FaceAligner → ArcFaceEmbeddingEngine → .npy

This script is for gallery preparation only.
It does NOT perform gallery matching, threshold decisions, or recognition.

Each direct subfolder under the source root represents one enrolled person.
A deterministic UUID5 person_id is derived from the original folder name.
For each image, the real ArcFace embedding is saved as a .npy file inside
the corresponding person_id subfolder.

Output structure:
    <output_root>/
        <person_id>/
            <person_id>__<english_image_name>__<index:03d>.npy
        build_stats.json
        skipped_images.json

Usage:
    python build_real_face_gallery_from_images.py \\
        --source-images-root <path> \\
        --output-gallery-root <path>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

# Ensure project root is on sys.path when running the script directly.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from src.image_processing.face_detection import (  # noqa: E402
    FaceDetectionConfig,
    FaceDetectionInput,
    FaceDetectionModule,
    SCRFDFaceDetector,
)
from src.image_processing.face_recognition import (  # noqa: E402
    ArcFaceEmbeddingEngine,
    FaceAligner,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)

# Stable RFC 4122 namespace — guarantees reproducible UUID5 values across runs
# for the same folder name.  Must match build_face_gallery_from_images.py.
_UUID5_NAMESPACE = uuid.NAMESPACE_OID

# Hebrew word for "image" (common iOS/Android filename prefix: תמונה)
_HEBREW_IMAGE_WORD = "תמונה"

# Default paths
_DEFAULT_SOURCE = r"C:\Users\talif\Desktop\photos"
_DEFAULT_OUTPUT = (
    r"C:\Users\talif\Desktop\Camera-regogintion\data\generated_face_gallery_real"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BuildStats:
    persons_found: int = 0
    images_found: int = 0
    face_detections_succeeded: int = 0
    face_detections_failed: int = 0
    embeddings_written: int = 0
    images_skipped: int = 0


@dataclass
class SkippedImage:
    source_path: str
    reason: str


# ---------------------------------------------------------------------------
# Identity helpers  (same logic as build_face_gallery_from_images.py)
# ---------------------------------------------------------------------------


def generate_person_id(folder_name: str) -> str:
    """Derive a deterministic UUID5 string from a folder name.

    The same folder_name always produces the same UUID5 value across runs.
    Uses uuid.NAMESPACE_OID — must match the Phase 1 script.

    Args:
        folder_name: Original human-readable name of the person's source folder.

    Returns:
        A lowercase UUID5 string, e.g. '550e8400-e29b-41d4-a716-446655440000'.
    """
    return str(uuid.uuid5(_UUID5_NAMESPACE, folder_name))


def sanitize_image_name(stem: str) -> str:
    """Convert an image base name to an English-safe filename part.

    Rules applied in order:
    1. Detect the Hebrew pattern <תמונה><digits> and map to image_<digits>.
    2. NFKD Unicode decomposition so accented letters shed diacritics.
    3. Encode to ASCII with 'ignore' to remove remaining non-ASCII bytes.
    4. Lowercase.
    5. Replace non-alphanumeric characters with underscores.
    6. Collapse consecutive underscores; strip leading/trailing ones.
    7. Return 'img' if the result is empty.

    Args:
        stem: Image filename without extension, e.g. 'תמונה36', 'IMG_1024'.

    Returns:
        An English-safe, lowercase, underscore-separated name string.
    """
    # Step 1: Hebrew image-name pattern
    hebrew_match = re.match(
        r"^" + _HEBREW_IMAGE_WORD + r"[\s_\-]*(\d+)$", stem.strip()
    )
    if hebrew_match:
        return f"image_{hebrew_match.group(1)}"

    # Step 2: Unicode decomposition
    normalized = unicodedata.normalize("NFKD", stem)

    # Step 3: Strip non-ASCII
    ascii_str = normalized.encode("ascii", errors="ignore").decode("ascii")

    # Step 4: Lowercase
    lower = ascii_str.lower()

    # Step 5: Replace non-alphanumeric with underscore
    safe = re.sub(r"[^a-z0-9]+", "_", lower)

    # Step 6: Collapse and strip underscores
    safe = re.sub(r"_+", "_", safe).strip("_")

    # Step 7: Fallback
    return safe if safe else "img"


# ---------------------------------------------------------------------------
# Image loading helper
# ---------------------------------------------------------------------------


def load_image_rgb(image_path: Path) -> np.ndarray | None:
    """Load an image from disk and return it as an RGB uint8 ndarray.

    Uses Path.read_bytes() + cv2.imdecode so that non-ASCII (e.g. Hebrew)
    file paths work correctly on Windows (cv2.imread does not handle them).

    Returns None if the image cannot be read (corrupted, unsupported codec).
    """
    try:
        raw = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Best-face selection helper
# ---------------------------------------------------------------------------


def select_best_detection(detections: list[Any]) -> Any:
    """Return the detection with the largest bounding box area.

    When multiple faces are detected, the dominant (largest) face is chosen
    deterministically by bounding-box area (width × height).

    Args:
        detections: Non-empty list of DetectedFace TypedDicts.

    Returns:
        The DetectedFace with the largest bounding box area.
    """
    return max(
        detections,
        key=lambda d: d["face_bbox_frame"]["width"] * d["face_bbox_frame"]["height"],
    )


# ---------------------------------------------------------------------------
# Core gallery builder
# ---------------------------------------------------------------------------


def build_gallery(
    source_root: Path,
    output_root: Path,
    detection_module: FaceDetectionModule,
    aligner: FaceAligner,
    embedding_engine: ArcFaceEmbeddingEngine,
) -> tuple[BuildStats, list[SkippedImage]]:
    """Scan source_root and write a FaceGalleryLoader-compatible gallery.

    Args:
        source_root:       Root folder; direct subfolders = one person each.
        output_root:       Destination gallery folder.  Must NOT already exist.
        detection_module:  Initialised FaceDetectionModule.
        aligner:           Initialised FaceAligner.
        embedding_engine:  Initialised ArcFaceEmbeddingEngine.

    Returns:
        Tuple of (BuildStats, list[SkippedImage]).

    Raises:
        FileNotFoundError:  source_root does not exist.
        NotADirectoryError: source_root is not a directory.
        FileExistsError:    output_root already exists.
    """
    # --- Validate source -----------------------------------------------------
    if not source_root.exists():
        raise FileNotFoundError(f"Source folder does not exist: {source_root}")
    if not source_root.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_root}")

    # --- Guard: do not silently overwrite -----------------------------------
    if output_root.exists():
        raise FileExistsError(
            f"Output gallery folder already exists: {output_root}\n"
            "Remove or rename it before running this script again."
        )

    # --- Collect person folders (sorted deterministically) -------------------
    person_folders = sorted(
        [p for p in source_root.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )

    if not person_folders:
        print("WARNING: No person subfolders found under source root. Nothing to do.")
        return BuildStats(), []

    stats = BuildStats(persons_found=len(person_folders))
    skipped: list[SkippedImage] = []

    # --- Process each person -------------------------------------------------
    for person_folder in person_folders:
        person_id = generate_person_id(person_folder.name)
        person_output_dir = output_root / person_id
        person_output_dir.mkdir(parents=True, exist_ok=False)

        print(f"\n  Person: {person_folder.name!r}  →  {person_id}")

        # Collect image files (sorted deterministically)
        image_files = sorted(
            [
                f
                for f in person_folder.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            ],
            key=lambda f: f.name,
        )

        stats.images_found += len(image_files)
        person_embedding_index = 0  # counts only successfully written embeddings

        for image_path in image_files:
            # ---- Load image ------------------------------------------------
            rgb_image = load_image_rgb(image_path)
            if rgb_image is None:
                reason = "failed to load image (corrupted or unsupported codec)"
                print(f"    SKIP  {image_path.name} — {reason}")
                skipped.append(SkippedImage(str(image_path), reason))
                stats.images_skipped += 1
                continue

            h, w = rgb_image.shape[:2]

            # ---- Face detection --------------------------------------------
            face_input: FaceDetectionInput = {
                "frame_id": 0,
                "camera_id": "gallery_build",
                "timestamp_ms": 0,
                "roi_image": rgb_image,
                "roi_bbox_frame": {"x": 0, "y": 0, "width": w, "height": h},
            }

            try:
                detection_output = detection_module.detect_faces(face_input)
            except Exception as exc:
                reason = f"face detection error: {exc}"
                print(f"    SKIP  {image_path.name} — {reason}")
                skipped.append(SkippedImage(str(image_path), reason))
                stats.face_detections_failed += 1
                stats.images_skipped += 1
                continue

            detections = detection_output["detections"]

            if not detections:
                reason = "no face detected"
                print(f"    SKIP  {image_path.name} — {reason}")
                skipped.append(SkippedImage(str(image_path), reason))
                stats.face_detections_failed += 1
                stats.images_skipped += 1
                continue

            stats.face_detections_succeeded += 1
            if len(detections) > 1:
                print(
                    f"    INFO  {image_path.name} — {len(detections)} faces found, "
                    "selecting largest"
                )

            best = select_best_detection(detections)
            landmarks = best["landmarks"]

            # ---- Alignment -------------------------------------------------
            try:
                aligned_face = aligner.align(rgb_image, landmarks)
            except Exception as exc:
                reason = f"face alignment error: {exc}"
                print(f"    SKIP  {image_path.name} — {reason}")
                skipped.append(SkippedImage(str(image_path), reason))
                stats.images_skipped += 1
                continue

            # ---- Embedding -------------------------------------------------
            try:
                embedding = embedding_engine.extract_embedding(aligned_face)
            except Exception as exc:
                reason = f"embedding extraction error: {exc}"
                print(f"    SKIP  {image_path.name} — {reason}")
                skipped.append(SkippedImage(str(image_path), reason))
                stats.images_skipped += 1
                continue

            # ---- Save .npy -------------------------------------------------
            person_embedding_index += 1
            safe_name = sanitize_image_name(image_path.stem)
            output_filename = (
                f"{person_id}__{safe_name}__{person_embedding_index:03d}.npy"
            )
            output_path = person_output_dir / output_filename

            try:
                np.save(output_path, embedding)
            except Exception as exc:
                reason = f"failed to write .npy file: {exc}"
                print(f"    SKIP  {image_path.name} — {reason}")
                skipped.append(SkippedImage(str(image_path), reason))
                stats.images_skipped += 1
                person_embedding_index -= 1  # roll back index on write failure
                continue

            stats.embeddings_written += 1
            print(f"    OK    {image_path.name}  →  {output_filename}")

    return stats, skipped


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def write_json(path: Path, data: object) -> None:
    """Write data as UTF-8-safe pretty-printed JSON."""
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2 gallery builder: generates real ArcFace embeddings.\n\n"
            "Pipeline: image → face detection → FaceAligner → "
            "ArcFaceEmbeddingEngine → .npy\n\n"
            "Does NOT perform gallery matching or recognition."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source-images-root",
        type=Path,
        default=Path(_DEFAULT_SOURCE),
        metavar="PATH",
        help=f"Root folder of source person images. Default: {_DEFAULT_SOURCE}",
    )
    parser.add_argument(
        "--output-gallery-root",
        type=Path,
        default=Path(_DEFAULT_OUTPUT),
        metavar="PATH",
        help=(
            f"Destination gallery folder (must not exist). "
            f"Default: {_DEFAULT_OUTPUT}"
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    source_root: Path = args.source_images_root.resolve()
    output_root: Path = args.output_gallery_root.resolve()

    print("=" * 60)
    print("  Phase 2 — Real Face Gallery Builder")
    print("=" * 60)
    print(f"  Source  : {source_root}")
    print(f"  Output  : {output_root}")
    print()

    # --- Initialise pipeline components ------------------------------------
    print("Initialising face detection model …")
    detector = SCRFDFaceDetector()
    detection_module = FaceDetectionModule(
        detector_engine=detector,
        config=FaceDetectionConfig(confidence_threshold=0.5),
    )

    print("Initialising ArcFace embedding model …")
    aligner = FaceAligner()
    embedding_engine = ArcFaceEmbeddingEngine()

    print("Models ready.\n")

    # --- Build gallery -----------------------------------------------------
    stats, skipped = build_gallery(
        source_root=source_root,
        output_root=output_root,
        detection_module=detection_module,
        aligner=aligner,
        embedding_engine=embedding_engine,
    )

    # --- Write metadata ----------------------------------------------------
    write_json(output_root / "build_stats.json", asdict(stats))
    write_json(
        output_root / "skipped_images.json",
        [asdict(s) for s in skipped],
    )

    # --- Summary -----------------------------------------------------------
    print()
    print("=" * 60)
    print("  Gallery build complete")
    print("=" * 60)
    print(f"  Source persons        : {stats.persons_found}")
    print(f"  Source images         : {stats.images_found}")
    print(f"  Face detections OK    : {stats.face_detections_succeeded}")
    print(f"  Face detections failed: {stats.face_detections_failed}")
    print(f"  Embeddings written    : {stats.embeddings_written}")
    print(f"  Images skipped        : {stats.images_skipped}")
    print(f"  Output gallery        : {output_root}")
    print(f"  build_stats.json      : {output_root / 'build_stats.json'}")
    print(f"  skipped_images.json   : {output_root / 'skipped_images.json'}")
    print("=" * 60)

    # --- Sample output paths -----------------------------------------------
    sample_paths = sorted(output_root.glob("*/*.npy"))[:5]
    if sample_paths:
        print("\nSample .npy outputs:")
        for p in sample_paths:
            print(f"  {p}")
    print()


if __name__ == "__main__":
    main()
