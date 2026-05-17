"""
Validate the REAL face gallery used by Face Recognition.

Usage:
    python scripts/validate_face_gallery.py
    python scripts/validate_face_gallery.py --gallery-dir data/gallery --person-json data/person_directory.json
    python scripts/validate_face_gallery.py --gallery-dir data/generated_face_gallery_real --norm-tolerance 1e-3

Exit codes:
    0  — validation passed (no errors)
    1  — validation failed (one or more errors found)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


EXPECTED_EMBEDDING_SHAPE = (512,)
EXPECTED_EMBEDDING_DTYPE = np.float32


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate the REAL face gallery used by Face Recognition."
    )
    parser.add_argument(
        "--gallery-dir",
        type=Path,
        default=Path("data/gallery"),
        help="Root directory of the face gallery (default: data/gallery)",
    )
    parser.add_argument(
        "--person-json",
        type=Path,
        default=Path("data/person_directory.json"),
        help="Path to person_directory.json (default: data/person_directory.json)",
    )
    parser.add_argument(
        "--norm-tolerance",
        type=float,
        default=1e-3,
        help="Tolerance for L2-norm check: abs(norm - 1.0) <= tolerance (default: 1e-3)",
    )
    return parser.parse_args()


def load_person_directory(person_json: Path):
    """Load person_directory.json and return a set of valid person_id strings."""
    with person_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data.keys())


def validate_gallery(gallery_dir: Path, known_person_ids: set, norm_tolerance: float):
    """
    Walk the gallery directory and validate all person folders and embeddings.

    Returns:
        person_folders   — list of Path for each immediate subdir found
        embeddings_checked — total count of .npy files inspected
        errors           — list of error strings
        warnings         — list of warning strings
    """
    errors = []
    warnings = []
    person_folders = []
    embeddings_checked = 0

    for entry in sorted(gallery_dir.iterdir()):
        if not entry.is_dir():
            continue  # skip loose files (e.g. build_stats.json)
        person_folders.append(entry)

    gallery_person_ids = {folder.name for folder in person_folders}

    # Warn about persons in JSON but absent from gallery
    for person_id in sorted(known_person_ids - gallery_person_ids):
        warnings.append(
            f"Person '{person_id}' is in person_directory.json but has no gallery folder."
        )

    for folder in person_folders:
        person_id = folder.name

        # Error: gallery folder not registered in person_directory.json
        if person_id not in known_person_ids:
            errors.append(
                f"Gallery folder '{person_id}' is not in person_directory.json."
            )

        npy_files = sorted(folder.glob("*.npy"))

        # Error: no embeddings found
        if not npy_files:
            errors.append(
                f"Person '{person_id}': folder contains no .npy embedding files."
            )
            continue

        for npy_path in npy_files:
            embeddings_checked += 1
            rel = npy_path.relative_to(gallery_dir)

            # Error: file cannot be loaded
            try:
                embedding = np.load(str(npy_path))
            except Exception as exc:
                errors.append(f"{rel}: failed to load — {exc}")
                continue

            # Error: wrong shape
            if embedding.shape != EXPECTED_EMBEDDING_SHAPE:
                errors.append(
                    f"{rel}: unexpected shape {embedding.shape} "
                    f"(expected {EXPECTED_EMBEDDING_SHAPE})."
                )
                continue

            # Error: wrong dtype
            if embedding.dtype != EXPECTED_EMBEDDING_DTYPE:
                errors.append(
                    f"{rel}: unexpected dtype {embedding.dtype} "
                    f"(expected {EXPECTED_EMBEDDING_DTYPE})."
                )
                continue

            # Error: not L2-normalized
            norm = float(np.linalg.norm(embedding))
            if abs(norm - 1.0) > norm_tolerance:
                errors.append(
                    f"{rel}: L2-norm {norm:.6f} deviates from 1.0 by "
                    f"{abs(norm - 1.0):.2e} (tolerance {norm_tolerance:.2e})."
                )

    return person_folders, embeddings_checked, errors, warnings


def print_report(
    known_person_ids: set,
    person_folders: list,
    embeddings_checked: int,
    errors: list,
    warnings: list,
):
    print("=" * 60)
    print("Face Gallery Validation Report")
    print("=" * 60)
    print(f"  Persons in person_directory.json : {len(known_person_ids)}")
    print(f"  Persons in gallery               : {len(person_folders)}")
    print(f"  Embedding files checked          : {embeddings_checked}")

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  [WARN]  {w}")
    else:
        print("\nWarnings : none")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  [ERROR] {e}")
    else:
        print("\nErrors   : none")

    print("=" * 60)
    if errors:
        print("Result   : FAILED")
    else:
        print("Result   : PASSED")
    print("=" * 60)


def main():
    args = parse_args()

    errors_early = []

    if not args.person_json.exists():
        errors_early.append(
            f"person_directory.json not found: {args.person_json}"
        )

    if not args.gallery_dir.exists():
        errors_early.append(
            f"Gallery directory not found: {args.gallery_dir}"
        )

    if errors_early:
        print("=" * 60)
        print("Face Gallery Validation Report")
        print("=" * 60)
        print(f"\nErrors ({len(errors_early)}):")
        for e in errors_early:
            print(f"  [ERROR] {e}")
        print("=" * 60)
        print("Result   : FAILED")
        print("=" * 60)
        sys.exit(1)

    known_person_ids = load_person_directory(args.person_json)

    person_folders, embeddings_checked, errors, warnings = validate_gallery(
        args.gallery_dir, known_person_ids, args.norm_tolerance
    )

    print_report(known_person_ids, person_folders, embeddings_checked, errors, warnings)

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
