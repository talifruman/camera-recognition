"""
Demo script for manual verification of the FaceGalleryLoaderModule (Phase 1 stub).

Run from the project root:
    python tests/face_gallery_loader/demo_face_gallery_loader.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Allow imports from src/ regardless of working directory.
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.image_processing.face_gallery_loader import (
    FaceGalleryLoaderConfig,
    FaceGalleryLoaderModule,
)

GALLERY_PATH = r"C:\Users\talif\Desktop\Camera-regogintion\data\generated_face_gallery"


def main() -> None:
    print("===== FACE GALLERY LOADER DEMO =====\n")
    print(f"Gallery path:\n{GALLERY_PATH}\n")

    config = FaceGalleryLoaderConfig(
        embedding_file_extension=".npy",
        expected_embedding_dim=512,
        expected_dtype="float32",
    )
    module = FaceGalleryLoaderModule(config=config)
    module.load_gallery(GALLERY_PATH)

    person_ids = module.get_person_ids()
    all_entries = module.get_all_embeddings()

    print(f"Total persons: {len(person_ids)}")
    print(f"Total embeddings: {len(all_entries)}\n")

    print("Person IDs (sorted):")
    for pid in person_ids:
        print(f"  - {pid}")

    print("\nEmbeddings per person:")
    for pid in person_ids:
        entries = module.get_embeddings(pid)
        print(f"  {pid} -> {len(entries)} embeddings")

    print("\nStub validation:")
    reference = all_entries[0]["embedding"]
    all_identical = all(
        np.array_equal(entry["embedding"], reference) for entry in all_entries
    )
    print(f"  All embeddings identical: {all_identical}")

    print("\n===================================")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
