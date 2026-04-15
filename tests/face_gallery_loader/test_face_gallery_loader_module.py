"""
Tests for FaceGalleryLoader Phase 1 (Stub).

Validates behavior defined in the Markdown spec:
  doc/image_processing_service/face_gallery_loader.md

Gallery under test:
  data/generated_face_gallery
  - 4 person directories
  - 57 total .npy files (distribution: 14, 14, 15, 14)

All embeddings are returned by StubEmbeddingFileReader — a constant
normalized float32 vector; all 57 must be element-wise identical.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# ------------------------------------------------------------------ path setup
_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.image_processing.face_gallery_loader import (
    FaceGalleryLoaderConfig,
    FaceGalleryLoaderModule,
    GalleryLoadError,
    GalleryPathValidationError,
)

# ------------------------------------------------------------------ constants

GALLERY_PATH = os.path.join(
    _REPO_ROOT, "data", "generated_face_gallery"
)

# Verified from workspace listing:
#   1f007fe2-6eaf-5148-a240-a449664cb6eb  — 14 files
#   750d3e47-6abb-5694-8b62-ea056a7f29b1  — 14 files
#   872b1095-dff9-50b0-8eb1-22852095fcb6  — 15 files
#   b4de6bd4-d07e-5047-ac26-e4c98cd4baee  — 14 files
_EXPECTED_PERSONS = 4
_EXPECTED_TOTAL_ENTRIES = 57
_PERSON_ID_SECOND = "1f007fe2-6eaf-5148-a240-a449664cb6eb"  # first lexicographically
_EXPECTED_ENTRIES_FOR_FIRST_PERSON = 14


# ------------------------------------------------------------------ helpers

def _make_config() -> FaceGalleryLoaderConfig:
    return FaceGalleryLoaderConfig(
        embedding_file_extension=".npy",
        expected_embedding_dim=512,
        expected_dtype="float32",
    )


def _make_module() -> FaceGalleryLoaderModule:
    """Return a FaceGalleryLoaderModule wired with StubEmbeddingFileReader."""
    return FaceGalleryLoaderModule(config=_make_config())


def _loaded_module() -> FaceGalleryLoaderModule:
    """Return a module that has already loaded the test gallery."""
    module = _make_module()
    module.load_gallery(GALLERY_PATH)
    return module


# ==================================================================
# Test 1 — Successful load from the gallery path
# ==================================================================

class TestSuccessfulLoad:

    def test_successful_load_from_gallery_path(self) -> None:
        """load_gallery must complete without raising on the real gallery."""
        module = _make_module()
        # Must not raise.
        module.load_gallery(GALLERY_PATH)


# ==================================================================
# Test 2 — Correct number of persons detected
# ==================================================================

class TestPersonCount:

    def test_correct_number_of_persons_detected(self) -> None:
        """get_person_ids() must return exactly 4 person identifiers."""
        module = _loaded_module()
        person_ids = module.get_person_ids()
        assert len(person_ids) == _EXPECTED_PERSONS, (
            f"Expected {_EXPECTED_PERSONS} persons, got {len(person_ids)}: "
            f"{person_ids}"
        )


# ==================================================================
# Test 3 — Correct number of entries loaded
# ==================================================================

class TestEntryCount:

    def test_correct_number_of_entries_loaded(self) -> None:
        """get_all_embeddings() must return 57 GalleryEntry values."""
        module = _loaded_module()
        entries = module.get_all_embeddings()
        assert len(entries) == _EXPECTED_TOTAL_ENTRIES, (
            f"Expected {_EXPECTED_TOTAL_ENTRIES} entries, got {len(entries)}"
        )


# ==================================================================
# Test 4 — get_person_ids() returns sorted values
# ==================================================================

class TestSorting:

    def test_get_person_ids_returns_sorted_values(self) -> None:
        """get_person_ids() must return person IDs in lexicographic order."""
        module = _loaded_module()
        person_ids = module.get_person_ids()
        assert person_ids == sorted(person_ids), (
            f"Expected sorted person IDs, got: {person_ids}"
        )


# ==================================================================
# Test 5 — get_all_embeddings() returns all entries
# ==================================================================

class TestAllEmbeddings:

    def test_get_all_embeddings_returns_all_entries(self) -> None:
        """get_all_embeddings() must include one entry per embedding file."""
        module = _loaded_module()
        entries = module.get_all_embeddings()
        assert len(entries) == _EXPECTED_TOTAL_ENTRIES

    def test_all_entries_have_person_id_and_embedding(self) -> None:
        """Every GalleryEntry must have non-empty person_id and an embedding."""
        module = _loaded_module()
        for entry in module.get_all_embeddings():
            assert entry["person_id"], "person_id must be non-empty"
            assert entry["embedding"] is not None, "embedding must not be None"
            assert isinstance(entry["embedding"], np.ndarray), (
                "embedding must be an ndarray"
            )


# ==================================================================
# Test 6 — get_embeddings(person_id) returns correct subset
# ==================================================================

class TestPersonSubset:

    def test_get_embeddings_returns_correct_subset(self) -> None:
        """get_embeddings for the first lexicographic person must return 14 entries."""
        module = _loaded_module()
        entries = module.get_embeddings(_PERSON_ID_SECOND)
        assert len(entries) == _EXPECTED_ENTRIES_FOR_FIRST_PERSON, (
            f"Expected {_EXPECTED_ENTRIES_FOR_FIRST_PERSON} entries for "
            f"{_PERSON_ID_SECOND!r}, got {len(entries)}"
        )

    def test_get_embeddings_all_entries_belong_to_requested_person(self) -> None:
        """Every entry returned by get_embeddings must have the requested person_id."""
        module = _loaded_module()
        entries = module.get_embeddings(_PERSON_ID_SECOND)
        for entry in entries:
            assert entry["person_id"] == _PERSON_ID_SECOND, (
                f"Entry has wrong person_id: {entry['person_id']!r}"
            )


# ==================================================================
# Test 7 — Unknown person_id returns empty list
# ==================================================================

class TestUnknownPerson:

    def test_unknown_person_id_returns_empty_list(self) -> None:
        """get_embeddings with an unknown person_id must return [] without error."""
        module = _loaded_module()
        result = module.get_embeddings("nonexistent-person-id")
        assert result == [], (
            f"Expected empty list for unknown person_id, got {result}"
        )


# ==================================================================
# Test 8 — Invalid path raises GalleryPathValidationError
# ==================================================================

class TestInvalidPath:

    def test_nonexistent_path_raises_gallery_path_validation_error(self) -> None:
        """load_gallery with a non-existent path must raise GalleryPathValidationError."""
        module = _make_module()
        with pytest.raises(GalleryPathValidationError):
            module.load_gallery("/nonexistent/path/that/does/not/exist")

    def test_empty_string_path_raises_gallery_path_validation_error(self) -> None:
        """load_gallery with an empty string must raise GalleryPathValidationError."""
        module = _make_module()
        with pytest.raises(GalleryPathValidationError):
            module.load_gallery("")

    def test_file_path_raises_gallery_path_validation_error(self, tmp_path) -> None:
        """load_gallery with a file (not a directory) must raise GalleryPathValidationError."""
        a_file = tmp_path / "not_a_dir.txt"
        a_file.write_text("content")
        module = _make_module()
        with pytest.raises(GalleryPathValidationError):
            module.load_gallery(str(a_file))


# ==================================================================
# Test 9 — Empty gallery raises GalleryLoadError
# ==================================================================

class TestEmptyGallery:

    def test_gallery_with_only_empty_subdirs_raises_gallery_load_error(
        self, tmp_path
    ) -> None:
        """
        A gallery root that has subdirectories but no matching embedding files
        must raise GalleryLoadError (all files effectively zero → no valid entries).
        """
        # Create a person subdir with no .npy files.
        person_dir = tmp_path / "some-person-id"
        person_dir.mkdir()
        (person_dir / "not_an_embedding.txt").write_text("ignored")

        module = _make_module()
        with pytest.raises(GalleryLoadError):
            module.load_gallery(str(tmp_path))

    def test_gallery_root_with_no_subdirs_raises_gallery_path_validation_error(
        self, tmp_path
    ) -> None:
        """
        A gallery root with no entries at all must fail at GalleryPathValidator
        (no direct children at all — the path check fires before the scan).
        """
        # tmp_path is empty here; GalleryPathValidator requires >= 1 child.
        module = _make_module()
        with pytest.raises(GalleryPathValidationError):
            module.load_gallery(str(tmp_path))


# ==================================================================
# Test 10 — All embeddings are identical (constant stub vector)
# ==================================================================

class TestConstantEmbedding:

    def test_all_embeddings_are_identical(self) -> None:
        """
        StubEmbeddingFileReader returns the same constant vector for every file.
        All 57 embeddings must be element-wise identical.
        """
        module = _loaded_module()
        entries = module.get_all_embeddings()
        assert entries, "No entries loaded"

        reference = entries[0]["embedding"]
        for i, entry in enumerate(entries[1:], start=1):
            assert np.array_equal(entry["embedding"], reference), (
                f"Entry {i} embedding differs from entry 0 — "
                "stub must return a constant vector for all files."
            )

    def test_stub_embedding_is_normalized(self) -> None:
        """The constant stub embedding must be a unit vector (L2 norm == 1)."""
        module = _loaded_module()
        entries = module.get_all_embeddings()
        embedding = entries[0]["embedding"]
        norm = float(np.linalg.norm(embedding))
        assert abs(norm - 1.0) < 1e-5, (
            f"Stub embedding L2 norm expected 1.0, got {norm}"
        )

    def test_stub_embedding_is_float32(self) -> None:
        """The constant stub embedding must have dtype float32."""
        module = _loaded_module()
        entries = module.get_all_embeddings()
        embedding = entries[0]["embedding"]
        assert embedding.dtype == np.float32, (
            f"Stub embedding dtype expected float32, got {embedding.dtype}"
        )

    def test_stub_embedding_has_expected_dimension(self) -> None:
        """The constant stub embedding must have the configured dimension (512)."""
        module = _loaded_module()
        entries = module.get_all_embeddings()
        embedding = entries[0]["embedding"]
        assert embedding.ndim == 1, "Embedding must be a 1D vector"
        assert embedding.shape[0] == 512, (
            f"Embedding dimension expected 512, got {embedding.shape[0]}"
        )
