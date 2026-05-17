"""
Tests for NpyEmbeddingFileReader using the real gallery at
data/generated_face_gallery_real/.

Covers:
  - valid .npy file loads correctly (shape, dtype, normalization)
  - wrong shape rejected
  - wrong dtype rejected
  - non-normalized embedding rejected
  - missing file returns None
  - wrong extension returns None
  - NpyEmbeddingFileReader configuration wiring in FaceGalleryLoaderModule
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

# ------------------------------------------------------------------ path setup
_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from image_processing.face_gallery_loader import (  # type: ignore[import-not-found]
    FaceGalleryLoaderConfig,
    FaceGalleryLoaderModule,
    NpyEmbeddingFileReader,
)

# ------------------------------------------------------------------ constants

REAL_GALLERY_PATH = os.path.join(
    _REPO_ROOT, "data", "generated_face_gallery_real"
)

_FIRST_PERSON_ID = "1f007fe2-6eaf-5148-a240-a449664cb6eb"

# Grab the first .npy file from the first person directory for file-level tests.
_FIRST_PERSON_DIR = os.path.join(REAL_GALLERY_PATH, _FIRST_PERSON_ID)
_VALID_NPY_FILE = os.path.join(
    _FIRST_PERSON_DIR,
    sorted(os.listdir(_FIRST_PERSON_DIR))[0],
)

_EXPECTED_DIM = 512
_EXPECTED_DTYPE = "float32"
_EXPECTED_EXTENSION = ".npy"


def _make_reader() -> NpyEmbeddingFileReader:
    return NpyEmbeddingFileReader(
        embedding_file_extension=_EXPECTED_EXTENSION,
        expected_embedding_dim=_EXPECTED_DIM,
        expected_dtype=_EXPECTED_DTYPE,
    )


# ==================================================================
# 1. Valid .npy file
# ==================================================================

class TestValidLoad:

    def test_valid_file_returns_non_none(self) -> None:
        reader = _make_reader()
        result = reader.read_embedding(_VALID_NPY_FILE)
        assert result is not None, f"Expected a valid embedding from {_VALID_NPY_FILE}"

    def test_valid_file_has_correct_shape(self) -> None:
        reader = _make_reader()
        result = reader.read_embedding(_VALID_NPY_FILE)
        assert result is not None
        assert result.shape == (_EXPECTED_DIM,), (
            f"Expected shape ({_EXPECTED_DIM},), got {result.shape}"
        )

    def test_valid_file_has_correct_dtype(self) -> None:
        reader = _make_reader()
        result = reader.read_embedding(_VALID_NPY_FILE)
        assert result is not None
        assert result.dtype == np.float32, (
            f"Expected float32 dtype, got {result.dtype}"
        )

    def test_valid_file_is_l2_normalized(self) -> None:
        reader = _make_reader()
        result = reader.read_embedding(_VALID_NPY_FILE)
        assert result is not None
        norm = float(np.linalg.norm(result))
        assert abs(norm - 1.0) < 1e-4, (
            f"Expected L2 norm ≈ 1.0, got {norm:.6f}"
        )


# ==================================================================
# 2. All real gallery files pass validation
# ==================================================================

class TestAllRealGalleryFiles:

    def test_all_npy_files_are_accepted(self) -> None:
        """Every .npy file in the real gallery must load without returning None."""
        reader = _make_reader()
        failures: list[str] = []
        for person_dir in sorted(os.listdir(REAL_GALLERY_PATH)):
            person_path = os.path.join(REAL_GALLERY_PATH, person_dir)
            if not os.path.isdir(person_path):
                continue
            for filename in sorted(os.listdir(person_path)):
                if not filename.endswith(".npy"):
                    continue
                file_path = os.path.join(person_path, filename)
                result = reader.read_embedding(file_path)
                if result is None:
                    failures.append(file_path)

        assert not failures, (
            f"{len(failures)} real gallery file(s) failed NpyEmbeddingFileReader validation:\n"
            + "\n".join(f"  {p}" for p in failures)
        )

    def test_all_npy_files_are_l2_normalized(self) -> None:
        """Every embedding in the real gallery must have L2 norm ≈ 1.0."""
        reader = _make_reader()
        bad_norms: list[tuple[str, float]] = []
        for person_dir in sorted(os.listdir(REAL_GALLERY_PATH)):
            person_path = os.path.join(REAL_GALLERY_PATH, person_dir)
            if not os.path.isdir(person_path):
                continue
            for filename in sorted(os.listdir(person_path)):
                if not filename.endswith(".npy"):
                    continue
                file_path = os.path.join(person_path, filename)
                result = reader.read_embedding(file_path)
                if result is not None:
                    norm = float(np.linalg.norm(result))
                    if abs(norm - 1.0) > 1e-4:
                        bad_norms.append((file_path, norm))

        assert not bad_norms, (
            f"{len(bad_norms)} file(s) have non-unit norm:\n"
            + "\n".join(f"  {p}: norm={n:.6f}" for p, n in bad_norms)
        )


# ==================================================================
# 3. Shape validation
# ==================================================================

class TestShapeValidation:

    def test_wrong_shape_returns_none(self, tmp_path) -> None:
        """An array with wrong shape must return None."""
        bad_file = str(tmp_path / "bad_shape.npy")
        np.save(bad_file, np.zeros(256, dtype=np.float32))
        reader = _make_reader()
        result = reader.read_embedding(bad_file)
        assert result is None, "Expected None for wrong-shape embedding"

    def test_2d_array_returns_none(self, tmp_path) -> None:
        """A 2D array (e.g., batch) must return None."""
        bad_file = str(tmp_path / "bad_2d.npy")
        np.save(bad_file, np.zeros((1, 512), dtype=np.float32))
        reader = _make_reader()
        result = reader.read_embedding(bad_file)
        assert result is None


# ==================================================================
# 4. Dtype validation
# ==================================================================

class TestDtypeValidation:

    def test_float64_returns_none(self, tmp_path) -> None:
        """A float64 embedding must return None (wrong dtype)."""
        bad_file = str(tmp_path / "bad_dtype.npy")
        arr = np.zeros(512, dtype=np.float64)
        arr[0] = 1.0  # make it L2-normalized so dtype is the only failing check
        np.save(bad_file, arr)
        reader = _make_reader()
        result = reader.read_embedding(bad_file)
        assert result is None, "Expected None for float64 dtype"

    def test_int32_returns_none(self, tmp_path) -> None:
        """An int32 embedding must return None (wrong dtype)."""
        bad_file = str(tmp_path / "bad_int.npy")
        np.save(bad_file, np.zeros(512, dtype=np.int32))
        reader = _make_reader()
        result = reader.read_embedding(bad_file)
        assert result is None


# ==================================================================
# 5. L2 normalization validation
# ==================================================================

class TestNormalizationValidation:

    def test_non_normalized_returns_none(self, tmp_path) -> None:
        """An embedding whose L2 norm deviates from 1.0 must return None."""
        bad_file = str(tmp_path / "unnormed.npy")
        arr = np.ones(512, dtype=np.float32)  # norm = sqrt(512) ≈ 22.6, not 1.0
        np.save(bad_file, arr)
        reader = _make_reader()
        result = reader.read_embedding(bad_file)
        assert result is None, "Expected None for non-normalized embedding"

    def test_non_normalized_emits_warning(self, tmp_path) -> None:
        """A non-normalized embedding must emit a warnings.warn message."""
        import warnings
        bad_file = str(tmp_path / "unnormed_warn.npy")
        arr = np.ones(512, dtype=np.float32)  # unnormalized
        np.save(bad_file, arr)
        reader = _make_reader()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reader.read_embedding(bad_file)
        assert any("L2" in str(w.message) or "norm" in str(w.message) for w in caught), (
            "Expected a warning about L2 normalization, but none was emitted"
        )

    def test_zero_vector_returns_none(self, tmp_path) -> None:
        """A zero vector (norm=0) must return None."""
        bad_file = str(tmp_path / "zero.npy")
        np.save(bad_file, np.zeros(512, dtype=np.float32))
        reader = _make_reader()
        result = reader.read_embedding(bad_file)
        assert result is None


# ==================================================================
# 6. Missing file / wrong extension
# ==================================================================

class TestMissingAndExtension:

    def test_missing_file_returns_none(self) -> None:
        reader = _make_reader()
        result = reader.read_embedding("/nonexistent/path/file.npy")
        assert result is None

    def test_wrong_extension_returns_none(self, tmp_path) -> None:
        bad_file = str(tmp_path / "embedding.bin")
        arr = np.ones(512, dtype=np.float32)
        arr /= np.linalg.norm(arr)
        np.save(bad_file, arr)
        reader = _make_reader()
        result = reader.read_embedding(bad_file)
        assert result is None


# ==================================================================
# 7. NpyEmbeddingFileReader wiring in FaceGalleryLoaderModule
# ==================================================================

class TestNpyReaderConfigWiring:
    """Verify NpyEmbeddingFileReader is correctly wired via FaceGalleryLoaderModule."""

    def test_module_loads_real_gallery_with_npy_reader(self) -> None:
        """
        FaceGalleryLoaderModule wired with NpyEmbeddingFileReader must load
        the real gallery without raising.
        """
        config = FaceGalleryLoaderConfig(
            embedding_file_extension=".npy",
            expected_embedding_dim=512,
            expected_dtype="float32",
        )
        reader = NpyEmbeddingFileReader(
            embedding_file_extension=config.embedding_file_extension,
            expected_embedding_dim=config.expected_embedding_dim,
            expected_dtype=config.expected_dtype,
        )
        module = FaceGalleryLoaderModule(config=config, reader=reader)
        module.load_gallery(REAL_GALLERY_PATH)  # must not raise

    def test_module_with_npy_reader_returns_correct_person_count(self) -> None:
        config = FaceGalleryLoaderConfig(
            embedding_file_extension=".npy",
            expected_embedding_dim=512,
            expected_dtype="float32",
        )
        reader = NpyEmbeddingFileReader(
            embedding_file_extension=config.embedding_file_extension,
            expected_embedding_dim=config.expected_embedding_dim,
            expected_dtype=config.expected_dtype,
        )
        module = FaceGalleryLoaderModule(config=config, reader=reader)
        module.load_gallery(REAL_GALLERY_PATH)
        person_ids = module.get_person_ids()
        assert len(person_ids) == 4, (
            f"Expected 4 persons, got {len(person_ids)}: {person_ids}"
        )

    def test_module_with_npy_reader_embeddings_are_normalized(self) -> None:
        """All embeddings loaded via NpyEmbeddingFileReader must be L2-normalized."""
        config = FaceGalleryLoaderConfig(
            embedding_file_extension=".npy",
            expected_embedding_dim=512,
            expected_dtype="float32",
        )
        reader = NpyEmbeddingFileReader(
            embedding_file_extension=config.embedding_file_extension,
            expected_embedding_dim=config.expected_embedding_dim,
            expected_dtype=config.expected_dtype,
        )
        module = FaceGalleryLoaderModule(config=config, reader=reader)
        module.load_gallery(REAL_GALLERY_PATH)
        entries = module.get_all_embeddings()
        for entry in entries:
            norm = float(np.linalg.norm(entry["embedding"]))
            assert abs(norm - 1.0) < 1e-4, (
                f"Embedding for person {entry['person_id']!r} has norm {norm:.6f}"
            )
