"""
build_face_gallery_from_images.py

Builds a FaceGalleryLoader-compatible embeddings gallery from an existing images folder.

Each direct subfolder under the source root represents one person.
A deterministic UUID5 person_id is derived from the original folder name.
For each image, an embedding is computed and saved as a .npy file inside
the corresponding person_id subfolder.

Output structure:
    <output_root>/
        <person_id>/
            <person_id>__<safe_image_name>__<index:03d>.npy
        build_stats.json

Usage:
    python build_face_gallery_from_images.py --source <path> --output <path>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)

# Stable, well-known RFC 4122 namespace — guarantees reproducible UUID5 values
# across all runs for the same folder name.
_UUID5_NAMESPACE = uuid.NAMESPACE_OID

# Default paths
_DEFAULT_SOURCE = r"C:\Users\talif\Desktop\photos"
_DEFAULT_OUTPUT = (
    r"C:\Users\talif\Desktop\Camera-regogintion\data\generated_face_gallery"
)

# Hebrew word for "image" (common iOS/Android filename prefix תמונה)
_HEBREW_IMAGE_WORD = "תמונה"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BuildStats:
    persons_found: int = 0
    images_found: int = 0
    embeddings_written: int = 0
    images_skipped: int = 0


# ---------------------------------------------------------------------------
# Embedding engine interface and stub implementation
# ---------------------------------------------------------------------------


@runtime_checkable
class FaceEmbeddingEngine(Protocol):
    """Interface for face embedding engines.

    Replace StubFaceEmbeddingEngine with a real implementation by satisfying
    this protocol — no other code changes are required.
    """

    def compute_embedding(self, image_path: Path) -> np.ndarray:
        """Return a normalised (512,) float32 embedding for the image at image_path."""
        ...


class StubFaceEmbeddingEngine:
    """Deterministic stub engine for testing and gallery scaffolding.

    Produces a reproducible, normalised (512,) float32 vector derived from
    the SHA-256 hash of the image path string.  The same path always produces
    the same vector.

    Replace this class with a real face embedding model implementation when
    the gallery must contain genuine face embeddings.
    """

    EMBEDDING_DIM: int = 512

    def compute_embedding(self, image_path: Path) -> np.ndarray:
        # Seed a NumPy RNG with a stable 32-bit integer derived from the path.
        path_hash = int(
            hashlib.sha256(str(image_path).encode("utf-8")).hexdigest(), 16
        )
        seed = path_hash % (2**32)
        rng = np.random.default_rng(seed)

        vector = rng.standard_normal(self.EMBEDDING_DIM).astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            # Extremely unlikely with a 512-d vector; handle defensively.
            vector[0] = 1.0
            norm = 1.0
        return vector / norm


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def generate_person_id(folder_name: str) -> str:
    """Derive a deterministic UUID5 string from a folder name.

    The same folder_name always produces the same UUID5 value across runs.

    Args:
        folder_name: Original human-readable name of the person's source folder.

    Returns:
        A lowercase UUID5 string, e.g. '550e8400-e29b-41d4-a716-446655440000'.
    """
    return str(uuid.uuid5(_UUID5_NAMESPACE, folder_name))


def sanitize_image_name(stem: str) -> str:
    """Convert an image base name (without extension) into an English-safe filename part.

    Rules applied in order:
    1. Detect the Hebrew pattern <תמונה><digits> (iOS/Android default photo name)
       and map it to image_<digits>.
    2. Decompose Unicode characters (NFKD) so that accented letters shed their
       diacritics and become plain ASCII letters.
    3. Encode to ASCII with 'ignore' to remove any remaining non-ASCII bytes.
    4. Lowercase the result.
    5. Replace any character that is not a-z, 0-9 with an underscore.
    6. Collapse consecutive underscores into one and strip leading/trailing ones.
    7. Return 'img' if the result is empty after all transformations (caller
       appends a numeric index so the final filename stays unique).

    Args:
        stem: Image filename without extension, e.g. 'תמונה36', 'IMG_1024', 'photo-03'.

    Returns:
        An English-safe, lowercase, underscore-separated name string.
    """
    # --- Step 1: Hebrew image-name pattern -----------------------------------
    # תמונה followed by one or more digits (with optional separators in between)
    hebrew_match = re.match(
        r"^" + _HEBREW_IMAGE_WORD + r"[\s_\-]*(\d+)$", stem.strip()
    )
    if hebrew_match:
        return f"image_{hebrew_match.group(1)}"

    # --- Step 2: Unicode decomposition ---------------------------------------
    normalized = unicodedata.normalize("NFKD", stem)

    # --- Step 3: Strip non-ASCII --------------------------------------------
    ascii_bytes = normalized.encode("ascii", errors="ignore")
    ascii_str = ascii_bytes.decode("ascii")

    # --- Step 4: Lowercase ---------------------------------------------------
    lower = ascii_str.lower()

    # --- Step 5: Replace non-alphanumeric with underscore -------------------
    safe = re.sub(r"[^a-z0-9]+", "_", lower)

    # --- Step 6: Collapse and strip underscores -----------------------------
    safe = re.sub(r"_+", "_", safe).strip("_")

    # --- Step 7: Fallback ----------------------------------------------------
    return safe if safe else "img"


# ---------------------------------------------------------------------------
# Gallery builder
# ---------------------------------------------------------------------------


def build_gallery(
    source_root: Path,
    output_root: Path,
    engine: FaceEmbeddingEngine,
) -> BuildStats:
    """Scan source_root and write a FaceGalleryLoader-compatible gallery to output_root.

    Args:
        source_root:  Root folder whose direct subfolders each represent one person.
        output_root:  Destination folder for the generated gallery.  Must NOT exist.
        engine:       Embedding engine implementing FaceEmbeddingEngine.

    Returns:
        A BuildStats instance with final counts.

    Raises:
        FileNotFoundError: If source_root does not exist.
        NotADirectoryError: If source_root is not a directory.
        FileExistsError:    If output_root already exists.
    """
    # --- Validate source -----------------------------------------------------
    if not source_root.exists():
        raise FileNotFoundError(f"Source folder does not exist: {source_root}")
    if not source_root.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_root}")

    # --- Guard: do not silently overwrite an existing gallery ----------------
    if output_root.exists():
        raise FileExistsError(
            f"Output gallery folder already exists: {output_root}\n"
            "Remove or rename it before running this script again."
        )

    # --- Collect person folders ----------------------------------------------
    person_folders = sorted(
        [p for p in source_root.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )

    if not person_folders:
        print("WARNING: No person subfolders found under source root. Nothing to do.")
        return BuildStats()

    stats = BuildStats(persons_found=len(person_folders))

    # --- Process each person -------------------------------------------------
    for person_folder in person_folders:
        person_id = generate_person_id(person_folder.name)
        person_output_dir = output_root / person_id
        person_output_dir.mkdir(parents=True, exist_ok=False)

        # Collect image files, sorted deterministically
        image_files = sorted(
            [
                f
                for f in person_folder.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            ],
            key=lambda f: f.name,
        )

        stats.images_found += len(image_files)

        for index, image_path in enumerate(image_files, start=1):
            safe_name = sanitize_image_name(image_path.stem)
            output_filename = f"{person_id}__{safe_name}__{index:03d}.npy"
            output_path = person_output_dir / output_filename

            try:
                embedding = engine.compute_embedding(image_path)
                np.save(output_path, embedding)
                stats.embeddings_written += 1
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  SKIP  {image_path.name} [{person_folder.name}] — {exc}"
                )
                stats.images_skipped += 1

    # --- Write build_stats.json ----------------------------------------------
    stats_path = output_root / "build_stats.json"
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(stats), fh, indent=2, ensure_ascii=False)

    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a FaceGalleryLoader-compatible embeddings gallery "
            "from an existing images folder."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(_DEFAULT_SOURCE),
        help=f"Root folder of source person images. Default: {_DEFAULT_SOURCE}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(_DEFAULT_OUTPUT),
        help=f"Destination gallery folder (must not exist). Default: {_DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_root: Path = args.source.resolve()
    output_root: Path = args.output.resolve()

    print(f"Source : {source_root}")
    print(f"Output : {output_root}")
    print()

    engine = StubFaceEmbeddingEngine()
    stats = build_gallery(source_root, output_root, engine)

    print()
    print("=" * 55)
    print("  Gallery build complete")
    print("=" * 55)
    print(f"  Persons found      : {stats.persons_found}")
    print(f"  Images found       : {stats.images_found}")
    print(f"  Embeddings written : {stats.embeddings_written}")
    print(f"  Images skipped     : {stats.images_skipped}")
    print(f"  Stats file         : {output_root / 'build_stats.json'}")
    print("=" * 55)


if __name__ == "__main__":
    main()
