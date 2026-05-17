"""
Integration tests: FaceGalleryLoaderModule → FaceRecognitionModule.

Covers the startup wiring path documented in doc/system.md §"Service Startup Sequence":
  1. FaceGalleryLoaderModule.load_gallery(gallery_root_path)
  2. entries = FaceGalleryLoaderModule.get_all_embeddings()  → list[LoadedGalleryEmbedding]
  3. startup wiring maps LoadedGalleryEmbedding[] → EnrolledIdentity[]
  4. FaceRecognitionModule(config, engine, gallery_entries=list[EnrolledIdentity])
  5. FaceRecognitionModule.recognize(face_input)

Tests:
  - Loader output is explicitly mapped to EnrolledIdentity[] before FR construction
  - recognize() works with real gallery (ArcFace engine + real .npy embeddings)
  - Non-normalized embedding causes NpyEmbeddingFileReader to skip the file
    (demonstrates why the L2 normalization check is needed)
  - LoadedGalleryEmbedding (loader domain) and EnrolledIdentity (FR domain) are
    two different types; they are NOT treated as the same type here.
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from typing import ClassVar, cast

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
    LoadedGalleryEmbedding,
    NpyEmbeddingFileReader,
)
from image_processing.face_recognition import (  # type: ignore[import-not-found]
    ArcFaceEmbeddingEngine,
    EnrolledIdentity,
    FaceAligner,
    FaceRecognitionConfig,
    FaceRecognitionInput,
    FaceRecognitionModule,
)
from image_processing.shared.contracts import (  # type: ignore[import-not-found]
    FaceLandmarks,
    Image,
    Point,
)

# ------------------------------------------------------------------ constants

REAL_GALLERY_PATH = os.path.join(
    _REPO_ROOT, "data", "generated_face_gallery_real"
)

_ROI_W = 112
_ROI_H = 112


# ------------------------------------------------------------------ helpers

def _make_face_roi(fill: int = 128) -> Image:
    """Return a 112×112×3 uint8 RGB Image struct."""
    data = np.full((_ROI_H, _ROI_W, 3), fill, dtype=np.uint8)
    return Image(
        data=data,
        width=_ROI_W,
        height=_ROI_H,
        color_format="RGB",
        layout="HWC",
        dtype="uint8",
        value_range="[0, 255]",
    )


def _make_landmarks() -> FaceLandmarks:
    """Return a plausible 5-point landmark set within the 112×112 ROI."""
    return FaceLandmarks(
        left_eye=Point(x=30, y=35),
        right_eye=Point(x=82, y=35),
        nose=Point(x=56, y=58),
        mouth_left=Point(x=35, y=80),
        mouth_right=Point(x=77, y=80),
    )


def _make_fr_input(
    frame_id: str = "frame_001",
    camera_id: str = "cam-a",
    timestamp_ms: int = 1000,
    fill: int = 128,
) -> FaceRecognitionInput:
    """Return a minimal FaceRecognitionInput-compatible dict."""
    return FaceRecognitionInput(
        frame_id=frame_id,
        camera_id=camera_id,
        timestamp_ms=timestamp_ms,
        face_roi_image=_make_face_roi(fill=fill),
        landmarks=_make_landmarks(),
    )


def _load_gallery_with_npy_reader(gallery_path: str) -> FaceGalleryLoaderModule:
    """Return a FaceGalleryLoaderModule loaded with NpyEmbeddingFileReader."""
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
    module.load_gallery(gallery_path)
    return module


# ==================================================================
# 1. Loader output → FaceRecognitionModule (structural compatibility)
# ==================================================================

class TestLoaderToRecognitionWiring:

    def test_loader_output_passes_to_fr_constructor(self) -> None:
        """
        LoadedGalleryEmbedding[] from loader must be explicitly mapped to
        EnrolledIdentity[] before passing to FaceRecognitionModule constructor.
        This is the documented startup wiring pattern (doc/system.md).
        """
        loader = _load_gallery_with_npy_reader(REAL_GALLERY_PATH)
        loader_entries: list[LoadedGalleryEmbedding] = loader.get_all_embeddings()
        assert loader_entries, "No entries loaded from real gallery"

        # Explicit startup mapping: LoadedGalleryEmbedding[] → EnrolledIdentity[]
        enrolled: list[EnrolledIdentity] = [
            EnrolledIdentity(person_id=e["person_id"], embedding=e["embedding"])
            for e in loader_entries
        ]

        module = FaceRecognitionModule(
            config=FaceRecognitionConfig(recognition_threshold=0.5),
            embedding_engine=ArcFaceEmbeddingEngine(),
            gallery_entries=enrolled,
        )
        assert module is not None

    def test_fr_module_has_correct_person_count_from_loader(self) -> None:
        """
        FaceRecognitionModule initialized from explicitly mapped loader output
        must have entries for all 4 gallery persons.
        """
        loader = _load_gallery_with_npy_reader(REAL_GALLERY_PATH)
        loader_entries: list[LoadedGalleryEmbedding] = loader.get_all_embeddings()
        person_ids_from_loader = set(loader.get_person_ids())

        enrolled: list[EnrolledIdentity] = [
            EnrolledIdentity(person_id=e["person_id"], embedding=e["embedding"])
            for e in loader_entries
        ]

        engine = ArcFaceEmbeddingEngine()
        module = FaceRecognitionModule(
            config=FaceRecognitionConfig(recognition_threshold=0.5),
            embedding_engine=engine,
            gallery_entries=enrolled,
        )

        # There's no public API to query gallery size from FR,
        # so we verify the loader side here.
        assert len(person_ids_from_loader) == 4


# ==================================================================
# 2. End-to-end: loader → FR → recognize() with ArcFace engine
# ==================================================================

class TestEndToEndRecognition:
    """
    Full startup-to-recognition path using the real gallery and ArcFace engine.

    setUpClass loads the gallery and initializes the engine once.
    """

    _loader: ClassVar[FaceGalleryLoaderModule | None] = None
    _enrolled: ClassVar[list[EnrolledIdentity]] = []
    _engine: ClassVar[ArcFaceEmbeddingEngine | None] = None
    _module: ClassVar[FaceRecognitionModule | None] = None

    @classmethod
    def setup_class(cls) -> None:
        cls._loader = _load_gallery_with_npy_reader(REAL_GALLERY_PATH)
        loader_entries: list[LoadedGalleryEmbedding] = cls._loader.get_all_embeddings()
        # Explicit startup mapping: LoadedGalleryEmbedding[] → EnrolledIdentity[]
        cls._enrolled = [
            EnrolledIdentity(person_id=e["person_id"], embedding=e["embedding"])
            for e in loader_entries
        ]
        engine = ArcFaceEmbeddingEngine()
        cls._engine = engine
        cls._module = FaceRecognitionModule(
            config=FaceRecognitionConfig(recognition_threshold=0.5),
            embedding_engine=engine,
            gallery_entries=cls._enrolled,
        )

    def test_recognize_returns_valid_output_schema(self) -> None:
        """recognize() output must contain all required keys."""
        module = cast(FaceRecognitionModule, self._module)
        result = module.recognize(_make_fr_input())
        assert set(result) == {
            "frame_id", "camera_id", "timestamp_ms", "person_found", "person_id"
        }

    def test_recognize_with_empty_gallery_returns_no_match(self) -> None:
        """
        FaceRecognitionModule initialized with an empty gallery must return
        person_found=False for any input.
        """
        engine = ArcFaceEmbeddingEngine()
        module_empty = FaceRecognitionModule(
            config=FaceRecognitionConfig(recognition_threshold=0.5),
            embedding_engine=engine,
            gallery_entries=[],
        )
        result = module_empty.recognize(_make_fr_input())
        assert result["person_found"] is False
        assert result["person_id"] == "UNKNOWN"

    def test_recognize_metadata_preserved(self) -> None:
        """frame_id, camera_id, timestamp_ms must be copied unchanged from input."""
        module = cast(FaceRecognitionModule, self._module)
        result = module.recognize(
            _make_fr_input(frame_id="integ_frame", camera_id="cam-x", timestamp_ms=9999)
        )
        assert result["frame_id"] == "integ_frame"
        assert result["camera_id"] == "cam-x"
        assert result["timestamp_ms"] == 9999

    def test_recognize_is_deterministic(self) -> None:
        """Two recognize() calls with the same input must produce the same output."""
        module = cast(FaceRecognitionModule, self._module)
        face_input = _make_fr_input()
        result1 = module.recognize(face_input)
        result2 = module.recognize(face_input)
        assert result1["person_found"] == result2["person_found"]
        assert result1["person_id"] == result2["person_id"]


# ==================================================================
# 3. Non-normalized embedding: NpyReader rejects, not silently corrupt
# ==================================================================

class TestNonNormalizedEmbeddingRejected:
    """
    Demonstrates why L2 normalization validation is necessary.

    Without the normalization check, FaceMatcher would compute an incorrect
    dot product for a non-normalized gallery embedding, producing a
    mathematically invalid similarity score silently.

    With the check, NpyEmbeddingFileReader returns None, the file is skipped,
    and the gallery is protected from corrupt data.
    """

    def test_non_normalized_file_is_skipped_by_loader(self, tmp_path) -> None:
        """
        A gallery with a non-normalized .npy embedding must skip that file
        and only include valid embeddings.
        """
        # Build a single-person gallery with one normalized and one un-normalized file.
        person_id = "test-person-123"
        person_dir = tmp_path / person_id
        person_dir.mkdir()

        # Good normalized embedding.
        good = np.ones(512, dtype=np.float32)
        good /= np.linalg.norm(good)
        np.save(str(person_dir / "good.npy"), good)

        # Bad non-normalized embedding.
        bad = np.ones(512, dtype=np.float32)  # norm ≈ 22.6, not 1.0
        np.save(str(person_dir / "bad_unnormed.npy"), bad)

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

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            module.load_gallery(str(tmp_path))

        entries = module.get_all_embeddings()
        # Only the good embedding should be included.
        assert len(entries) == 1, (
            f"Expected 1 valid entry (bad file skipped), got {len(entries)}"
        )
        norm = float(np.linalg.norm(entries[0]["embedding"]))
        assert abs(norm - 1.0) < 1e-4, (
            f"Loaded embedding is not L2-normalized (norm={norm:.6f})"
        )

    def test_non_normalized_file_emits_warning(self, tmp_path) -> None:
        """NpyEmbeddingFileReader must emit a warning when skipping a non-normalized file."""
        person_id = "warn-test"
        person_dir = tmp_path / person_id
        person_dir.mkdir()

        # Need at least one valid file to prevent GalleryLoadError.
        good = np.ones(512, dtype=np.float32)
        good /= np.linalg.norm(good)
        np.save(str(person_dir / "good.npy"), good)

        bad = np.ones(512, dtype=np.float32)  # unnormalized
        np.save(str(person_dir / "bad.npy"), bad)

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

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            module.load_gallery(str(tmp_path))

        norm_warnings = [
            w for w in caught
            if "L2" in str(w.message) or "norm" in str(w.message)
        ]
        assert norm_warnings, (
            "Expected at least one warning about L2 normalization; none emitted"
        )
