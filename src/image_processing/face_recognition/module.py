"""
Face Recognition Module
=======================
All internal pipeline components for the Face Recognition module, as
specified in Face Recognition Module spec §8.

Pipeline order (spec §8.9, §13.2):
    validation → alignment → embedding → matching → decision → output construction

Public API (spec §4):
    FaceRecognitionModule.recognize_face(input: FaceRecognitionInput) -> FaceRecognitionOutput

This module owns no alignment or embedding logic.  Each responsibility lives
in a dedicated internal component:

    FaceRecognitionInputValidator  — spec §8.2
    FaceAligner                    — spec §8.3  (imported from aligner.py)
    FaceEmbeddingEngine            — spec §8.4  (imported from embedding_engine.py)
    FaceGalleryCache               — spec §8.5
    FaceMatcher                    — spec §8.6
    FaceRecognitionDecisionPolicy  — spec §8.7
    FaceRecognitionOutputBuilder   — spec §8.8
    FaceRecognitionModule          — spec §8.1  (orchestrator only)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, TypedDict

import numpy as np

from .aligner import AlignedFace, FaceAligner
from .embedding_engine import FaceEmbeddingEngine

# ---------------------------------------------------------------------------
# Type alias — matches gallery-loader convention
# ---------------------------------------------------------------------------

# FaceEmbedding is a fixed-dimension float32 unit vector.
FaceEmbedding = np.ndarray


# ---------------------------------------------------------------------------
# Public data structures — spec §2.2, §3.1
# ---------------------------------------------------------------------------


class Point(TypedDict):
    x: int
    y: int


class FaceLandmarks(TypedDict):
    left_eye: Point
    right_eye: Point
    nose: Point
    mouth_left: Point
    mouth_right: Point


class FaceRecognitionInput(TypedDict):
    frame_id: int
    camera_id: str
    timestamp_ms: int
    face_roi_image: Any  # np.ndarray at runtime
    landmarks: FaceLandmarks


class FaceRecognitionOutput(TypedDict):
    frame_id: int
    camera_id: str
    timestamp_ms: int
    person_found: bool
    person_id: str  # populated only when person_found = True (spec §3.2)


class GalleryEntry(TypedDict):
    """
    Enrolled identity record held by FaceGalleryCache (spec §10).

    Structurally identical to face_gallery_loader.GalleryEntry — no
    cross-module import is needed; duck-typing applies.
    """
    person_id: str
    embedding: FaceEmbedding


# ---------------------------------------------------------------------------
# Internal per-call data structures — spec §10 (never escape public API)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MatchCandidate:
    """Best gallery match returned by FaceMatcher (spec §10)."""
    person_id: str
    similarity: float


@dataclass(slots=True)
class RecognitionDecision:
    """Identity decision produced by FaceRecognitionDecisionPolicy (spec §10)."""
    person_found: bool
    person_id: str = field(default="")


# ---------------------------------------------------------------------------
# Configuration — spec §9.1
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FaceRecognitionConfig:
    """
    Immutable module configuration loaded once at initialization (spec §9.2).

    recognition_threshold:
        Cosine-similarity acceptance threshold applied exclusively by
        FaceRecognitionDecisionPolicy.  Never passed per invocation.
    """
    recognition_threshold: float = 0.5


# ---------------------------------------------------------------------------
# FaceRecognitionInputValidator — spec §8.2, §2.4
# ---------------------------------------------------------------------------

_LANDMARK_KEYS = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")


class FaceRecognitionInputValidator:
    """
    Validates all input fields and landmarks before processing begins.

    Does NOT perform any preprocessing, alignment, or matching, and makes
    no acceptance decisions (spec §8.2).
    """

    def validate(self, face_input: FaceRecognitionInput) -> None:
        """
        Verify all spec §2.4 rules.

        Raises
        ------
        ValueError
            On any validation failure.  The caller catches this and returns
            FaceRecognitionOutput with person_found = False.
        """
        if "frame_id" not in face_input:
            raise ValueError("frame_id is required")
        if not face_input.get("camera_id"):
            raise ValueError("camera_id is required and must be non-empty")
        if "timestamp_ms" not in face_input:
            raise ValueError("timestamp_ms is required")

        face_roi_image = face_input.get("face_roi_image")
        if face_roi_image is None:
            raise ValueError("face_roi_image is required and must be non-null")

        landmarks = face_input.get("landmarks")
        if landmarks is None:
            raise ValueError("landmarks are required")

        self._validate_landmarks(landmarks, face_roi_image)

    @staticmethod
    def _validate_landmarks(landmarks: Any, face_roi_image: Any) -> None:
        """Validate all 5 canonical landmark points (spec §2.4)."""
        for key in _LANDMARK_KEYS:
            if key not in landmarks:
                raise ValueError(f"landmarks is missing required point: '{key}'")
            pt = landmarks[key]
            if "x" not in pt or "y" not in pt:
                raise ValueError(
                    f"landmark '{key}' is missing 'x' or 'y' coordinate"
                )
            x = pt["x"]
            y = pt["y"]
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(
                    f"landmark '{key}' has non-finite coordinate: ({x}, {y})"
                )
            # Bounds check — coordinates must be within face_roi_image extent
            if isinstance(face_roi_image, np.ndarray):
                img_h, img_w = face_roi_image.shape[:2]
                if not (0 <= x < img_w and 0 <= y < img_h):
                    raise ValueError(
                        f"landmark '{key}' coordinate ({x}, {y}) is outside "
                        f"face_roi_image bounds ({img_w}×{img_h})"
                    )


# ---------------------------------------------------------------------------
# FaceGalleryCache — spec §8.5
# ---------------------------------------------------------------------------


class FaceGalleryCache:
    """
    Immutable in-memory store of enrolled face embeddings (spec §8.5).

    Built once at module initialization from the enrolled GalleryEntry list.
    Read-only during recognition calls.  No external I/O during recognition.
    """

    def __init__(self, entries: list[GalleryEntry]) -> None:
        # Store an immutable copy — no mutation after construction
        self._entries: list[GalleryEntry] = list(entries)

    def get_entries(self) -> list[GalleryEntry]:
        """Return the enrolled gallery entries (read-only view)."""
        return self._entries


# ---------------------------------------------------------------------------
# FaceMatcher — spec §8.6
# ---------------------------------------------------------------------------


class FaceMatcher:
    """
    Compares the query embedding against all gallery entries and returns the
    best candidate (spec §8.6).

    Does NOT apply recognition_threshold or make accept/reject decisions.
    """

    def find_best_match(
        self,
        embedding: FaceEmbedding,
        gallery: list[GalleryEntry],
    ) -> MatchCandidate | None:
        """
        Return the MatchCandidate with the highest cosine similarity, or
        None if the gallery is empty (spec §8.6).

        Similarity is computed as the dot product of unit-norm vectors,
        which equals cosine similarity for L2-normalised embeddings.

        Parameters
        ----------
        embedding:
            Query (512,) float32 unit vector produced by FaceEmbeddingEngine.
        gallery:
            List of enrolled GalleryEntry records from FaceGalleryCache.

        Returns
        -------
        MatchCandidate | None
            Best candidate with person_id and similarity score, or None.
        """
        if not gallery:
            return None

        best_person_id: str = ""
        best_similarity: float = -2.0  # minimum possible dot product

        for entry in gallery:
            sim = float(np.dot(embedding, entry["embedding"]))
            if sim > best_similarity:
                best_similarity = sim
                best_person_id = entry["person_id"]

        return MatchCandidate(person_id=best_person_id, similarity=best_similarity)


# ---------------------------------------------------------------------------
# FaceRecognitionDecisionPolicy — spec §8.7
# ---------------------------------------------------------------------------


class FaceRecognitionDecisionPolicy:
    """
    Applies recognition_threshold and produces the final identity decision.

    This is the ONLY place inside the module that decides whether a match
    result becomes an accepted identity (spec §8.7).

    recognition_threshold is loaded from configuration once at
    initialization and is never passed per invocation (spec §9.2).
    """

    def __init__(self, recognition_threshold: float) -> None:
        self._threshold = recognition_threshold

    def decide(self, candidate: MatchCandidate | None) -> RecognitionDecision:
        """
        Apply threshold to a MatchCandidate (spec §8.7).

        Parameters
        ----------
        candidate:
            Best match from FaceMatcher, or None (empty gallery).

        Returns
        -------
        RecognitionDecision
            person_found = True and person_id set when similarity >= threshold;
            person_found = False otherwise.
        """
        if candidate is None:
            return RecognitionDecision(person_found=False)
        if candidate.similarity >= self._threshold:
            return RecognitionDecision(
                person_found=True,
                person_id=candidate.person_id,
            )
        return RecognitionDecision(person_found=False)


# ---------------------------------------------------------------------------
# FaceRecognitionOutputBuilder — spec §8.8
# ---------------------------------------------------------------------------


class FaceRecognitionOutputBuilder:
    """
    Constructs the final FaceRecognitionOutput from the identity decision
    and preserved input metadata (spec §8.8).

    Does NOT perform matching, apply threshold logic, or make decisions.
    """

    def build(
        self,
        frame_id: int,
        camera_id: str,
        timestamp_ms: int,
        decision: RecognitionDecision,
    ) -> FaceRecognitionOutput:
        """
        Assemble FaceRecognitionOutput.

        person_id is populated only when person_found = True (spec §3.2).

        Parameters
        ----------
        frame_id, camera_id, timestamp_ms:
            Copied unchanged from FaceRecognitionInput for traceability.
        decision:
            Identity decision from FaceRecognitionDecisionPolicy.
        """
        return FaceRecognitionOutput(
            frame_id=frame_id,
            camera_id=camera_id,
            timestamp_ms=timestamp_ms,
            person_found=decision.person_found,
            person_id=decision.person_id if decision.person_found else "",
        )


# ---------------------------------------------------------------------------
# FaceRecognitionModule — spec §8.1
# ---------------------------------------------------------------------------


class FaceRecognitionModule:
    """
    Orchestration layer for the Face Recognition module (spec §8.1).

    Owns no recognition logic.  Wires all internal subcomponents at
    initialization and invokes them in the prescribed order for each call.

    Pipeline per invocation (spec §8.9, §13.2):
        validation → alignment → embedding → matching → decision → output

    Error handling (spec §11):
        Any exception at any pipeline stage is caught internally.
        The caller always receives a valid FaceRecognitionOutput; on any
        failure person_found = False.

    Usage:
        module = FaceRecognitionModule(
            config=FaceRecognitionConfig(recognition_threshold=0.5),
            embedding_engine=StubFaceEmbeddingEngine(),
            gallery_entries=entries,
        )
        output = module.recognize_face(face_input)
    """

    def __init__(
        self,
        config: FaceRecognitionConfig,
        embedding_engine: FaceEmbeddingEngine,
        gallery_entries: list[GalleryEntry],
    ) -> None:
        """
        Initialize the module and wire all internal subcomponents.

        Parameters
        ----------
        config:
            Immutable module configuration.  Loaded once; never changes.
        embedding_engine:
            FaceEmbeddingEngine implementation injected as an abstract
            dependency.  The module depends on the interface only.
        gallery_entries:
            Enrolled gallery entries supplied at construction time.
            FaceGalleryCache is built from these and held as immutable
            in-memory state for the module's lifetime (spec §13.1).
        """
        self._config = config

        # Wire subcomponents with injected configuration (spec §8.1, §13.1)
        self._validator = FaceRecognitionInputValidator()
        self._aligner = FaceAligner()
        self._embedding_engine: FaceEmbeddingEngine = embedding_engine
        self._gallery_cache = FaceGalleryCache(gallery_entries)
        self._matcher = FaceMatcher()
        self._decision_policy = FaceRecognitionDecisionPolicy(
            recognition_threshold=config.recognition_threshold,
        )
        self._output_builder = FaceRecognitionOutputBuilder()

    # ---- public API (spec §4) --------------------------------------------

    def recognize_face(
        self, face_input: FaceRecognitionInput
    ) -> FaceRecognitionOutput:
        """
        Identify the person in a pre-cropped face ROI image (spec §4).

        Parameters
        ----------
        face_input:
            Validated pre-prepared face input.  One face per invocation.

        Returns
        -------
        FaceRecognitionOutput
            Always returns a valid output.  On any failure returns
            person_found = False without raising exceptions (spec §11).
        """
        try:
            return self._recognize_internal(face_input)
        except Exception:
            # Spec §11: all failure modes return person_found = False
            return self._no_match_output(face_input)

    # ---- internal pipeline (spec §8.9) -----------------------------------

    def _recognize_internal(
        self, face_input: FaceRecognitionInput
    ) -> FaceRecognitionOutput:
        # Step 1: validation (spec §8.2)
        self._validator.validate(face_input)

        # Step 2: alignment (spec §8.3)
        aligned_face: AlignedFace = self._aligner.align(
            face_input["face_roi_image"],
            face_input["landmarks"],
        )

        # Step 3: embedding extraction (spec §8.4)
        embedding: FaceEmbedding = self._embedding_engine.extract_embedding(
            aligned_face
        )

        # Step 4: matching against gallery (spec §8.6)
        candidate = self._matcher.find_best_match(
            embedding,
            self._gallery_cache.get_entries(),
        )

        # Step 5: threshold decision (spec §8.7)
        decision: RecognitionDecision = self._decision_policy.decide(candidate)

        # Step 6: output construction (spec §8.8)
        return self._output_builder.build(
            frame_id=face_input["frame_id"],
            camera_id=face_input["camera_id"],
            timestamp_ms=face_input["timestamp_ms"],
            decision=decision,
        )

    @staticmethod
    def _no_match_output(face_input: FaceRecognitionInput) -> FaceRecognitionOutput:
        """Return a valid no-match output for any error or failure path."""
        return FaceRecognitionOutput(
            frame_id=face_input.get("frame_id", 0),  # type: ignore[call-overload]
            camera_id=face_input.get("camera_id", ""),  # type: ignore[call-overload]
            timestamp_ms=face_input.get("timestamp_ms", 0),  # type: ignore[call-overload]
            person_found=False,
            person_id="",
        )
