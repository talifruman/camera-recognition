"""Regression tests for canonical face gallery path usage."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_GALLERY_PATH = "generated_face_gallery"
LEGACY_GALLERY_PATH = f"{CANONICAL_GALLERY_PATH}_real"
FILES_THAT_MUST_USE_CANONICAL_PATH = (
    "tools/build_real_face_gallery_from_images.py",
    "scripts/validate_face_gallery.py",
    "tools/rpm_visual_debug_runner.py",
    "tests/face_detection/three_stage_visual_test_app.py",
    "tests/face_gallery_loader/gallery_consistency_check.py",
    "tests/face_gallery_loader/test_gallery_to_recognition_integration.py",
    "tests/face_gallery_loader/test_npy_embedding_file_reader.py",
)


def test_known_entry_points_do_not_reference_legacy_gallery_path() -> None:
    """Known entry points and tests must not reference the legacy gallery path."""
    files_with_legacy_reference: list[str] = []

    for relative_path in FILES_THAT_MUST_USE_CANONICAL_PATH:
        file_text = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        if LEGACY_GALLERY_PATH in file_text:
            files_with_legacy_reference.append(relative_path)

    assert not files_with_legacy_reference, (
        "Legacy gallery path references remain in: "
        f"{files_with_legacy_reference}"
    )


def test_known_entry_points_reference_canonical_gallery_path() -> None:
    """Known entry points and tests must reference the canonical gallery path."""
    files_missing_canonical_reference: list[str] = []

    for relative_path in FILES_THAT_MUST_USE_CANONICAL_PATH:
        file_text = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        if CANONICAL_GALLERY_PATH not in file_text:
            files_missing_canonical_reference.append(relative_path)

    assert not files_missing_canonical_reference, (
        "Canonical gallery path references are missing from: "
        f"{files_missing_canonical_reference}"
    )