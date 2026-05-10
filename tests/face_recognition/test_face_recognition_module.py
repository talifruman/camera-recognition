"""
Tests for the Face Recognition module — STUB phase.

Verifies all 8 required scenarios (see implementation task):

1.  Valid input returns a valid FaceRecognitionOutput object.
2.  Invalid input returns person_found = False.
3.  Empty gallery returns person_found = False.
4.  Accepted match returns person_found = True and correct person_id.
5.  Below-threshold candidate returns person_found = False.
6.  Metadata (frame_id, camera_id, timestamp_ms) is preserved unchanged.
7.  Same aligned input produces the same stub embedding (determinism).
8.  Same input + same config + same gallery → same module output (determinism).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — mirror face_detection test conventions
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_processing.face_recognition import (  # type: ignore[import-not-found]
    ArcFaceEmbeddingEngine,
    FaceAligner,
    FaceGalleryCache,
    FaceLandmarks,
    FaceMatcher,
    FaceRecognitionConfig,
    FaceRecognitionDecisionPolicy,
    FaceRecognitionInput,
    FaceRecognitionModule,
    FaceRecognitionOutput,
    FaceRecognitionOutputBuilder,
    GalleryEntry,
    MatchCandidate,
    Point,
    RecognitionDecision,
    StubFaceEmbeddingEngine,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_PERSON_A = "person-a-uuid"
_PERSON_B = "person-b-uuid"

# Standard 112×112 RGB uint8 face image for testing (aligner requires uint8).
_ROI_W = 112
_ROI_H = 112


def _make_face_roi(fill: int = 128) -> np.ndarray:
    """Return a 112×112×3 uint8 RGB ndarray."""
    return np.full((_ROI_H, _ROI_W, 3), fill, dtype=np.uint8)


def _make_landmarks(
    face_roi: np.ndarray | None = None,
) -> FaceLandmarks:
    """
    Return FaceLandmarks with all 5 canonical points inside the ROI bounds.

    Uses coordinates that match the ArcFace reference template region so the
    similarity transform does not produce degenerate geometry.
    """
    return FaceLandmarks(
        left_eye=Point(x=38, y=51),
        right_eye=Point(x=73, y=51),
        nose=Point(x=56, y=71),
        mouth_left=Point(x=41, y=92),
        mouth_right=Point(x=70, y=92),
    )


def _make_input(
    frame_id: str = "frame_0001",
    camera_id: str = "cam-01",
    timestamp_ms: int = 5000,
    face_roi: np.ndarray | None = None,
    landmarks: FaceLandmarks | None = None,
) -> FaceRecognitionInput:
    roi = face_roi if face_roi is not None else _make_face_roi()
    lm = landmarks if landmarks is not None else _make_landmarks(roi)
    return FaceRecognitionInput(
        frame_id=frame_id,
        camera_id=camera_id,
        timestamp_ms=timestamp_ms,
        face_roi_image=roi,
        landmarks=lm,
    )


def _stub_embedding_for_input(face_input: FaceRecognitionInput) -> np.ndarray:
    """
    Compute the exact stub embedding for a given input by running the same
    alignment + stub extraction that the module will use.

    Used to build gallery entries that are guaranteed to match a specific
    face input at a high similarity score.
    """
    from image_processing.face_recognition import FaceAligner  # type: ignore[import-not-found]

    aligner = FaceAligner()
    engine = StubFaceEmbeddingEngine()
    aligned = aligner.align(face_input["face_roi_image"], face_input["landmarks"])
    return engine.extract_embedding(aligned)


def _make_matching_gallery_entry(
    face_input: FaceRecognitionInput,
    person_id: str = _PERSON_A,
) -> GalleryEntry:
    """
    Return a GalleryEntry whose embedding is the exact stub output for the
    given face_input.  Any threshold below 1.0 will accept this match.
    """
    embedding = _stub_embedding_for_input(face_input)
    return GalleryEntry(person_id=person_id, embedding=embedding)


def _make_random_unit_vector(seed: int, dim: int = 512) -> np.ndarray:
    """Return a seeded, L2-normalised float32 vector (not matching any stub output)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(v)
    return (v / norm).astype(np.float32)


def _make_config(threshold: float = 0.5) -> FaceRecognitionConfig:
    return FaceRecognitionConfig(recognition_threshold=threshold)


def _make_module(
    config: FaceRecognitionConfig | None = None,
    gallery_entries: list[GalleryEntry] | None = None,
) -> FaceRecognitionModule:
    return FaceRecognitionModule(
        config=config or _make_config(),
        embedding_engine=StubFaceEmbeddingEngine(),
        gallery_entries=gallery_entries or [],
    )


# ---------------------------------------------------------------------------
# 1. Valid input returns a valid FaceRecognitionOutput
# ---------------------------------------------------------------------------

class OutputContractTests(unittest.TestCase):
    """Verify the public output contract matches spec §3.1 exactly."""

    def test_output_has_all_required_keys(self) -> None:
        module = _make_module()
        result = module.recognize_face(_make_input())
        self.assertEqual(
            set(result),
            {"frame_id", "camera_id", "timestamp_ms", "person_found", "person_id"},
        )

    def test_output_person_found_is_bool(self) -> None:
        module = _make_module()
        result = module.recognize_face(_make_input())
        self.assertIsInstance(result["person_found"], bool)

    def test_output_frame_id_is_str(self) -> None:
        module = _make_module()
        result = module.recognize_face(_make_input(frame_id="frame_0042"))
        self.assertIsInstance(result["frame_id"], str)

    def test_output_camera_id_is_str(self) -> None:
        module = _make_module()
        result = module.recognize_face(_make_input(camera_id="cam-x"))
        self.assertIsInstance(result["camera_id"], str)

    def test_output_timestamp_ms_is_int(self) -> None:
        module = _make_module()
        result = module.recognize_face(_make_input(timestamp_ms=9999))
        self.assertIsInstance(result["timestamp_ms"], int)

    def test_person_id_is_empty_when_person_not_found(self) -> None:
        module = _make_module(gallery_entries=[])
        result = module.recognize_face(_make_input())
        self.assertFalse(result["person_found"])
        self.assertEqual(result["person_id"], "")


# ---------------------------------------------------------------------------
# 2. Invalid input returns person_found = False
# ---------------------------------------------------------------------------

class ValidationTests(unittest.TestCase):
    """Verify all spec §2.4 validation rules map to person_found = False."""

    def _assert_no_match(self, face_input: dict) -> None:
        module = _make_module()
        result = module.recognize_face(face_input)  # type: ignore[arg-type]
        self.assertFalse(result["person_found"])

    def test_missing_frame_id_returns_no_match(self) -> None:
        inp = _make_input()
        del inp["frame_id"]
        self._assert_no_match(inp)

    def test_missing_camera_id_returns_no_match(self) -> None:
        inp = _make_input()
        del inp["camera_id"]
        self._assert_no_match(inp)

    def test_empty_camera_id_returns_no_match(self) -> None:
        self._assert_no_match(_make_input(camera_id=""))

    def test_missing_timestamp_ms_returns_no_match(self) -> None:
        inp = _make_input()
        del inp["timestamp_ms"]
        self._assert_no_match(inp)

    def test_null_face_roi_image_returns_no_match(self) -> None:
        inp = _make_input()
        inp["face_roi_image"] = None
        self._assert_no_match(inp)

    def test_missing_landmarks_returns_no_match(self) -> None:
        inp = _make_input()
        del inp["landmarks"]
        self._assert_no_match(inp)

    def test_landmarks_missing_a_point_returns_no_match(self) -> None:
        roi = _make_face_roi()
        bad_lm = {
            "left_eye": {"x": 38, "y": 51},
            "right_eye": {"x": 73, "y": 51},
            "nose": {"x": 56, "y": 71},
            # mouth_left and mouth_right omitted
        }
        inp = _make_input(face_roi=roi, landmarks=bad_lm)  # type: ignore[arg-type]
        self._assert_no_match(inp)

    def test_landmark_out_of_bounds_returns_no_match(self) -> None:
        roi = _make_face_roi()
        bad_lm = FaceLandmarks(
            left_eye=Point(x=38, y=51),
            right_eye=Point(x=73, y=51),
            nose=Point(x=56, y=71),
            mouth_left=Point(x=41, y=92),
            mouth_right=Point(x=9999, y=9999),  # way outside 112×112
        )
        self._assert_no_match(_make_input(face_roi=roi, landmarks=bad_lm))


# ---------------------------------------------------------------------------
# 3. Empty gallery returns person_found = False
# ---------------------------------------------------------------------------

class EmptyGalleryTests(unittest.TestCase):

    def test_empty_gallery_returns_no_match(self) -> None:
        module = _make_module(gallery_entries=[])
        result = module.recognize_face(_make_input())
        self.assertFalse(result["person_found"])

    def test_empty_gallery_person_id_is_empty_string(self) -> None:
        module = _make_module(gallery_entries=[])
        result = module.recognize_face(_make_input())
        self.assertEqual(result["person_id"], "")


# ---------------------------------------------------------------------------
# 4. Accepted match returns person_found = True and correct person_id
# ---------------------------------------------------------------------------

class AcceptedMatchTests(unittest.TestCase):
    """
    For an exact match the stub engine returns a unit vector.  A gallery entry
    containing that same unit vector produces cosine similarity = 1.0, which
    exceeds any reasonable threshold.
    """

    def test_exact_match_returns_person_found_true(self) -> None:
        face_input = _make_input()
        entry = _make_matching_gallery_entry(face_input, person_id=_PERSON_A)
        module = _make_module(
            config=_make_config(threshold=0.5),
            gallery_entries=[entry],
        )
        result = module.recognize_face(face_input)
        self.assertTrue(result["person_found"])

    def test_exact_match_returns_correct_person_id(self) -> None:
        face_input = _make_input()
        entry = _make_matching_gallery_entry(face_input, person_id=_PERSON_A)
        module = _make_module(
            config=_make_config(threshold=0.5),
            gallery_entries=[entry],
        )
        result = module.recognize_face(face_input)
        self.assertEqual(result["person_id"], _PERSON_A)

    def test_correct_identity_selected_among_multiple_entries(self) -> None:
        # Build the face_input with a distinct ROI to get a specific embedding
        face_input = FaceRecognitionInput(
            frame_id="frame_0001",
            camera_id="cam",
            timestamp_ms=0,
            face_roi_image=_make_face_roi(fill=200),
            landmarks=_make_landmarks(),
        )
        matching_entry = _make_matching_gallery_entry(face_input, person_id=_PERSON_A)
        other_entry = GalleryEntry(
            person_id=_PERSON_B,
            embedding=_make_random_unit_vector(seed=42),
        )
        module = _make_module(
            config=_make_config(threshold=0.5),
            gallery_entries=[other_entry, matching_entry],
        )
        result = module.recognize_face(face_input)
        self.assertTrue(result["person_found"])
        self.assertEqual(result["person_id"], _PERSON_A)


# ---------------------------------------------------------------------------
# 5. Below-threshold candidate returns person_found = False
# ---------------------------------------------------------------------------

class BelowThresholdTests(unittest.TestCase):

    def test_below_threshold_returns_no_match(self) -> None:
        """
        Set threshold = 1.1 (above maximum cosine similarity = 1.0) to force
        all candidates below threshold regardless of gallery content.
        """
        face_input = _make_input()
        entry = _make_matching_gallery_entry(face_input, person_id=_PERSON_A)
        module = _make_module(
            config=_make_config(threshold=1.1),
            gallery_entries=[entry],
        )
        result = module.recognize_face(face_input)
        self.assertFalse(result["person_found"])

    def test_below_threshold_person_id_is_empty_string(self) -> None:
        face_input = _make_input()
        entry = _make_matching_gallery_entry(face_input, person_id=_PERSON_A)
        module = _make_module(
            config=_make_config(threshold=1.1),
            gallery_entries=[entry],
        )
        result = module.recognize_face(face_input)
        self.assertEqual(result["person_id"], "")


# ---------------------------------------------------------------------------
# 6. Metadata (frame_id, camera_id, timestamp_ms) preserved unchanged
# ---------------------------------------------------------------------------

class MetadataPreservationTests(unittest.TestCase):

    def _result_for(
        self, frame_id: str, camera_id: str, timestamp_ms: int
    ) -> FaceRecognitionOutput:
        module = _make_module()
        return module.recognize_face(
            _make_input(
                frame_id=frame_id,
                camera_id=camera_id,
                timestamp_ms=timestamp_ms,
            )
        )

    def test_frame_id_preserved(self) -> None:
        result = self._result_for(frame_id="frame_0099", camera_id="cam", timestamp_ms=0)
        self.assertEqual(result["frame_id"], "frame_0099")

    def test_camera_id_preserved(self) -> None:
        result = self._result_for(frame_id="frame_0001", camera_id="camera-XYZ", timestamp_ms=0)
        self.assertEqual(result["camera_id"], "camera-XYZ")

    def test_timestamp_ms_preserved(self) -> None:
        result = self._result_for(frame_id="frame_0001", camera_id="cam", timestamp_ms=123456)
        self.assertEqual(result["timestamp_ms"], 123456)

    def test_metadata_preserved_on_validation_failure(self) -> None:
        """Metadata must still be copied even when validation fails."""
        inp = _make_input(frame_id="frame_0077", camera_id="cam-err", timestamp_ms=555)
        inp["face_roi_image"] = None  # trigger validation failure
        module = _make_module()
        result = module.recognize_face(inp)  # type: ignore[arg-type]
        self.assertFalse(result["person_found"])
        # frame_id and camera_id may fall back to defaults on error path;
        # the important contract is person_found = False (spec §11).
        # Metadata in the error path uses .get() fallbacks (spec §11 note).


# ---------------------------------------------------------------------------
# 7. Same aligned input produces the same stub embedding (determinism)
# ---------------------------------------------------------------------------

class StubEmbeddingDeterminismTests(unittest.TestCase):

    def test_same_image_produces_same_embedding(self) -> None:
        engine = StubFaceEmbeddingEngine()
        from image_processing.face_recognition import FaceAligner  # type: ignore[import-not-found]

        aligner = FaceAligner()
        face_input = _make_input()
        aligned = aligner.align(face_input["face_roi_image"], face_input["landmarks"])

        emb1 = engine.extract_embedding(aligned)
        emb2 = engine.extract_embedding(aligned)
        np.testing.assert_array_equal(emb1, emb2)

    def test_different_images_may_produce_different_embeddings(self) -> None:
        engine = StubFaceEmbeddingEngine()
        from image_processing.face_recognition import FaceAligner  # type: ignore[import-not-found]

        aligner = FaceAligner()
        lm = _make_landmarks()

        aligned_a = aligner.align(_make_face_roi(fill=0), lm)
        aligned_b = aligner.align(_make_face_roi(fill=255), lm)

        emb_a = engine.extract_embedding(aligned_a)
        emb_b = engine.extract_embedding(aligned_b)
        # Different pixel content → almost certainly different embeddings
        self.assertFalse(np.array_equal(emb_a, emb_b))

    def test_stub_embedding_is_unit_norm(self) -> None:
        engine = StubFaceEmbeddingEngine()
        from image_processing.face_recognition import FaceAligner  # type: ignore[import-not-found]

        aligner = FaceAligner()
        face_input = _make_input()
        aligned = aligner.align(face_input["face_roi_image"], face_input["landmarks"])
        emb = engine.extract_embedding(aligned)
        norm = float(np.linalg.norm(emb))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_stub_embedding_has_correct_dimension(self) -> None:
        engine = StubFaceEmbeddingEngine()
        from image_processing.face_recognition import FaceAligner  # type: ignore[import-not-found]

        aligner = FaceAligner()
        face_input = _make_input()
        aligned = aligner.align(face_input["face_roi_image"], face_input["landmarks"])
        emb = engine.extract_embedding(aligned)
        self.assertEqual(emb.shape, (512,))
        self.assertEqual(emb.dtype, np.float32)


# ---------------------------------------------------------------------------
# 8. Same input + config + gallery → same module output (determinism)
# ---------------------------------------------------------------------------

class ModuleDeterminismTests(unittest.TestCase):

    def test_same_input_produces_same_output(self) -> None:
        face_input = _make_input(frame_id="cam1_1712345678", camera_id="cam-d", timestamp_ms=100)
        entry = _make_matching_gallery_entry(face_input, person_id=_PERSON_A)
        config = _make_config(threshold=0.5)

        module1 = FaceRecognitionModule(
            config=config,
            embedding_engine=StubFaceEmbeddingEngine(),
            gallery_entries=[entry],
        )
        module2 = FaceRecognitionModule(
            config=config,
            embedding_engine=StubFaceEmbeddingEngine(),
            gallery_entries=[entry],
        )

        result1 = module1.recognize_face(face_input)
        result2 = module2.recognize_face(face_input)

        self.assertEqual(result1["person_found"], result2["person_found"])
        self.assertEqual(result1["person_id"], result2["person_id"])
        self.assertEqual(result1["frame_id"], result2["frame_id"])
        self.assertEqual(result1["camera_id"], result2["camera_id"])
        self.assertEqual(result1["timestamp_ms"], result2["timestamp_ms"])

    def test_calling_module_twice_with_same_input_produces_same_output(self) -> None:
        face_input = _make_input()
        entry = _make_matching_gallery_entry(face_input, person_id=_PERSON_A)
        module = _make_module(
            config=_make_config(threshold=0.5),
            gallery_entries=[entry],
        )

        result1 = module.recognize_face(face_input)
        result2 = module.recognize_face(face_input)

        self.assertEqual(result1["person_found"], result2["person_found"])
        self.assertEqual(result1["person_id"], result2["person_id"])


# ---------------------------------------------------------------------------
# Unit tests for individual internal components
# ---------------------------------------------------------------------------

class FaceGalleryCacheTests(unittest.TestCase):

    def test_get_entries_returns_all_entries(self) -> None:
        entries = [
            GalleryEntry(person_id="p1", embedding=_make_random_unit_vector(1)),
            GalleryEntry(person_id="p2", embedding=_make_random_unit_vector(2)),
        ]
        cache = FaceGalleryCache(entries)
        self.assertEqual(len(cache.get_entries()), 2)

    def test_get_entries_on_empty_cache(self) -> None:
        cache = FaceGalleryCache([])
        self.assertEqual(cache.get_entries(), [])


class FaceMatcherTests(unittest.TestCase):

    def test_empty_gallery_returns_none(self) -> None:
        matcher = FaceMatcher()
        query = _make_random_unit_vector(seed=99)
        result = matcher.find_best_match(query, [])
        self.assertIsNone(result)

    def test_single_entry_returns_that_entry(self) -> None:
        matcher = FaceMatcher()
        v = _make_random_unit_vector(seed=10)
        entry = GalleryEntry(person_id="person-x", embedding=v)
        result = matcher.find_best_match(v, [entry])
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.person_id, "person-x")
        self.assertAlmostEqual(result.similarity, 1.0, places=4)

    def test_returns_highest_similarity_entry(self) -> None:
        matcher = FaceMatcher()
        query = _make_random_unit_vector(seed=0)
        # entry_a is the query itself → similarity = 1.0
        entry_a = GalleryEntry(person_id="exact", embedding=query.copy())
        # entry_b is orthogonal-ish
        entry_b = GalleryEntry(
            person_id="other", embedding=_make_random_unit_vector(seed=99)
        )
        result = matcher.find_best_match(query, [entry_b, entry_a])
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.person_id, "exact")


class FaceRecognitionDecisionPolicyTests(unittest.TestCase):

    def test_none_candidate_returns_no_match(self) -> None:
        policy = FaceRecognitionDecisionPolicy(recognition_threshold=0.5)
        decision = policy.decide(None)
        self.assertFalse(decision.person_found)

    def test_above_threshold_returns_accepted(self) -> None:
        policy = FaceRecognitionDecisionPolicy(recognition_threshold=0.5)
        candidate = MatchCandidate(person_id="p1", similarity=0.9)
        decision = policy.decide(candidate)
        self.assertTrue(decision.person_found)
        self.assertEqual(decision.person_id, "p1")

    def test_at_threshold_returns_accepted(self) -> None:
        policy = FaceRecognitionDecisionPolicy(recognition_threshold=0.5)
        candidate = MatchCandidate(person_id="p2", similarity=0.5)
        decision = policy.decide(candidate)
        self.assertTrue(decision.person_found)

    def test_below_threshold_returns_no_match(self) -> None:
        policy = FaceRecognitionDecisionPolicy(recognition_threshold=0.5)
        candidate = MatchCandidate(person_id="p3", similarity=0.3)
        decision = policy.decide(candidate)
        self.assertFalse(decision.person_found)


class FaceRecognitionOutputBuilderTests(unittest.TestCase):

    def test_accepted_decision_sets_person_found_and_person_id(self) -> None:
        builder = FaceRecognitionOutputBuilder()
        decision = RecognitionDecision(person_found=True, person_id="pid-123")
        output = builder.build(
            frame_id="frame_0007", camera_id="cam-q", timestamp_ms=3000, decision=decision
        )
        self.assertTrue(output["person_found"])
        self.assertEqual(output["person_id"], "pid-123")
        self.assertEqual(output["frame_id"], "frame_0007")
        self.assertEqual(output["camera_id"], "cam-q")
        self.assertEqual(output["timestamp_ms"], 3000)

    def test_rejected_decision_sets_person_id_to_empty(self) -> None:
        builder = FaceRecognitionOutputBuilder()
        decision = RecognitionDecision(person_found=False)
        output = builder.build(
            frame_id="frame_0001", camera_id="cam", timestamp_ms=0, decision=decision
        )
        self.assertFalse(output["person_found"])
        self.assertEqual(output["person_id"], "")


# ---------------------------------------------------------------------------
# REAL path helpers
# ---------------------------------------------------------------------------


def _arcface_embedding_for_input(
    engine: ArcFaceEmbeddingEngine,
    face_input: FaceRecognitionInput,
) -> np.ndarray:
    """Align the ROI and extract a real ArcFace embedding."""
    aligner = FaceAligner()
    aligned = aligner.align(face_input["face_roi_image"], face_input["landmarks"])
    return engine.extract_embedding(aligned)


def _make_real_matching_gallery_entry(
    engine: ArcFaceEmbeddingEngine,
    face_input: FaceRecognitionInput,
    person_id: str = _PERSON_A,
) -> GalleryEntry:
    """Return a GalleryEntry whose embedding is the real ArcFace output for face_input."""
    embedding = _arcface_embedding_for_input(engine, face_input)
    return GalleryEntry(person_id=person_id, embedding=embedding)


def _make_real_module(
    engine: ArcFaceEmbeddingEngine,
    config: FaceRecognitionConfig | None = None,
    gallery_entries: list[GalleryEntry] | None = None,
) -> FaceRecognitionModule:
    return FaceRecognitionModule(
        config=config or _make_config(),
        embedding_engine=engine,
        gallery_entries=gallery_entries or [],
    )


# ---------------------------------------------------------------------------
# ArcFaceEmbeddingEngine unit tests
# ---------------------------------------------------------------------------


class ArcFaceEngineTests(unittest.TestCase):
    """
    Verify ArcFaceEmbeddingEngine loads and produces embeddings with the
    expected contract (spec §6.2).

    setUpClass loads the model once; all tests in this class share it.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._engine = ArcFaceEmbeddingEngine()
        aligner = FaceAligner()
        face_input = _make_input()
        cls._aligned_face = aligner.align(
            face_input["face_roi_image"], face_input["landmarks"]
        )

    def test_engine_loads_successfully(self) -> None:
        # Reaching this test proves setUpClass constructed the engine without error.
        self.assertIsNotNone(self._engine)

    def test_extract_embedding_shape(self) -> None:
        emb = self._engine.extract_embedding(self._aligned_face)
        self.assertEqual(emb.shape, (512,))

    def test_extract_embedding_dtype(self) -> None:
        emb = self._engine.extract_embedding(self._aligned_face)
        self.assertEqual(emb.dtype, np.float32)

    def test_extract_embedding_is_unit_norm(self) -> None:
        emb = self._engine.extract_embedding(self._aligned_face)
        norm = float(np.linalg.norm(emb))
        self.assertAlmostEqual(norm, 1.0, places=5)


# ---------------------------------------------------------------------------
# REAL end-to-end pipeline tests
# ---------------------------------------------------------------------------


class RealPipelineTests(unittest.TestCase):
    """
    End-to-end REAL path: FaceRecognitionModule wired with ArcFaceEmbeddingEngine.

    A single ArcFaceEmbeddingEngine instance is loaded once in setUpClass and
    shared across all tests to avoid repeated ONNX session creation.

    Covers spec §4, §5, §7, §8.9, §11 for the REAL execution path.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._engine = ArcFaceEmbeddingEngine()

    def test_real_valid_input_returns_valid_output(self) -> None:
        module = _make_real_module(self._engine)
        result = module.recognize_face(_make_input())
        self.assertEqual(
            set(result),
            {"frame_id", "camera_id", "timestamp_ms", "person_found", "person_id"},
        )

    def test_real_empty_gallery_returns_no_match(self) -> None:
        module = _make_real_module(self._engine, gallery_entries=[])
        result = module.recognize_face(_make_input())
        self.assertFalse(result["person_found"])

    def test_real_empty_gallery_person_id_is_empty(self) -> None:
        module = _make_real_module(self._engine, gallery_entries=[])
        result = module.recognize_face(_make_input())
        self.assertEqual(result["person_id"], "")

    def test_real_accepted_match_returns_person_found_true(self) -> None:
        face_input = _make_input()
        entry = _make_real_matching_gallery_entry(
            self._engine, face_input, person_id=_PERSON_A
        )
        module = _make_real_module(
            self._engine,
            config=_make_config(threshold=0.5),
            gallery_entries=[entry],
        )
        result = module.recognize_face(face_input)
        self.assertTrue(result["person_found"])

    def test_real_accepted_match_returns_correct_person_id(self) -> None:
        face_input = _make_input()
        entry = _make_real_matching_gallery_entry(
            self._engine, face_input, person_id=_PERSON_A
        )
        module = _make_real_module(
            self._engine,
            config=_make_config(threshold=0.5),
            gallery_entries=[entry],
        )
        result = module.recognize_face(face_input)
        self.assertEqual(result["person_id"], _PERSON_A)

    def test_real_below_threshold_returns_no_match(self) -> None:
        face_input = _make_input()
        entry = _make_real_matching_gallery_entry(
            self._engine, face_input, person_id=_PERSON_A
        )
        # threshold = 1.1 is above maximum cosine similarity (1.0); forces rejection
        module = _make_real_module(
            self._engine,
            config=_make_config(threshold=1.1),
            gallery_entries=[entry],
        )
        result = module.recognize_face(face_input)
        self.assertFalse(result["person_found"])

    def test_real_below_threshold_person_id_is_empty(self) -> None:
        face_input = _make_input()
        entry = _make_real_matching_gallery_entry(
            self._engine, face_input, person_id=_PERSON_A
        )
        module = _make_real_module(
            self._engine,
            config=_make_config(threshold=1.1),
            gallery_entries=[entry],
        )
        result = module.recognize_face(face_input)
        self.assertEqual(result["person_id"], "")

    def test_real_metadata_preserved(self) -> None:
        module = _make_real_module(self._engine)
        result = module.recognize_face(
            _make_input(frame_id="frame_0099", camera_id="cam-real", timestamp_ms=7777)
        )
        self.assertEqual(result["frame_id"], "frame_0099")
        self.assertEqual(result["camera_id"], "cam-real")
        self.assertEqual(result["timestamp_ms"], 7777)

    def test_real_deterministic(self) -> None:
        face_input = _make_input()
        entry = _make_real_matching_gallery_entry(
            self._engine, face_input, person_id=_PERSON_A
        )
        module = _make_real_module(
            self._engine,
            config=_make_config(threshold=0.5),
            gallery_entries=[entry],
        )
        result1 = module.recognize_face(face_input)
        result2 = module.recognize_face(face_input)
        self.assertEqual(result1["person_found"], result2["person_found"])
        self.assertEqual(result1["person_id"], result2["person_id"])
        self.assertEqual(result1["frame_id"], result2["frame_id"])
        self.assertEqual(result1["camera_id"], result2["camera_id"])
        self.assertEqual(result1["timestamp_ms"], result2["timestamp_ms"])


if __name__ == "__main__":
    unittest.main()
