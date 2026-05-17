"""
Face Recognition Module
=======================
All internal pipeline components for the Face Recognition module, as
specified in Face Recognition Module spec §8.

Pipeline order (spec §8.9, §13.2):
    validation → alignment → embedding → matching → decision → output construction

Public API (spec §4):
    FaceRecognitionModule.recognize(input: FaceRecognitionInput) -> FaceRecognitionOutput

This module owns no alignment or embedding logic.  Each responsibility lives
in a dedicated internal component:

    FaceRecognitionInputValidator  — spec §8.2
    FaceAligner                    — spec §8.3  (imported from aligner.py)
    FaceEmbeddingEngine            — spec §8.4  (imported from embedding_engine.py)
    EnrolledIdentityCache          — spec §8.5
    FaceMatcher                    — spec §8.6
    FaceRecognitionDecisionPolicy  — spec §8.7
    FaceRecognitionOutputBuilder   — spec §8.8
    FaceRecognitionModule          — spec §8.1  (orchestrator only)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, TypedDict, runtime_checkable

import numpy as np

from image_processing.shared.contracts import (
    FaceLandmarks,
    GeometrySpec,
    Image,
    OutputImageType,
    PipelineStageInputContract,
    Point,
    ResizePolicy,
)

from .aligner import AlignedFace, FaceAligner
from .embedding_engine import FaceEmbeddingEngine

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

# FaceEmbedding is a fixed-dimension float32 unit vector.
FaceEmbedding = np.ndarray


# ---------------------------------------------------------------------------
# Expected public image contract for face recognition
# (matches FaceAligner input requirement; ArcFaceEmbeddingEngine normalizes internally)
# ---------------------------------------------------------------------------

_EXPECTED_COLOR_FORMAT = "RGB"
_EXPECTED_LAYOUT = "HWC"
_EXPECTED_DTYPE = "uint8"
_EXPECTED_VALUE_RANGE = "[0,255]"


# ---------------------------------------------------------------------------
# Public data structures — spec §2.2, §3.1
# (Point, FaceLandmarks imported from image_processing.shared.contracts)
# ---------------------------------------------------------------------------


class FaceRecognitionInput(TypedDict):
    frame_id: str
    camera_id: str
    timestamp_ms: int
    face_roi_image: Image  # shared Image struct (see shared_contracts.md §6)
    landmarks: FaceLandmarks


class FaceRecognitionOutput(TypedDict):
    frame_id: str
    camera_id: str
    timestamp_ms: int
    person_found: bool
    person_id: str  # populated only when person_found = True (spec §3.2)


class EnrolledIdentity(TypedDict):
    """
    Enrolled identity record held by EnrolledIdentityCache (spec §10).

    Internal to Face Recognition. Supplied externally at construction time
    by startup code; never exposed through the public API.
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
    person_id: str = field(default="UNKNOWN")


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
        if face_input["timestamp_ms"] < 0:
            raise ValueError("timestamp_ms must be non-negative (uint64 semantics)")

        face_roi_image = face_input.get("face_roi_image")
        if face_roi_image is None:
            raise ValueError("face_roi_image is required and must be non-null")

        self._validate_image(face_roi_image)

        landmarks = face_input.get("landmarks")
        if landmarks is None:
            raise ValueError("landmarks are required")

        self._validate_landmarks(landmarks, face_roi_image)

    @staticmethod
    def _validate_image(face_roi_image: object) -> None:
        """Validate face_roi_image as a shared Image TypedDict (spec §2.4)."""
        if not isinstance(face_roi_image, dict):
            raise ValueError(
                "face_roi_image must be a shared Image struct (dict), "
                "not a raw np.ndarray or other type"
            )

        data = face_roi_image.get("data")  # type: ignore[union-attr]
        if data is None or not isinstance(data, np.ndarray):
            raise ValueError("face_roi_image.data must be a non-null np.ndarray")

        width = face_roi_image.get("width", 0)  # type: ignore[union-attr]
        height = face_roi_image.get("height", 0)  # type: ignore[union-attr]
        if width <= 0:
            raise ValueError(f"face_roi_image.width must be > 0, got {width}")
        if height <= 0:
            raise ValueError(f"face_roi_image.height must be > 0, got {height}")

        color_format = face_roi_image.get("color_format")  # type: ignore[union-attr]
        if color_format != _EXPECTED_COLOR_FORMAT:
            raise ValueError(
                f"face_roi_image.color_format must be '{_EXPECTED_COLOR_FORMAT}', "
                f"got '{color_format}'"
            )

        layout = face_roi_image.get("layout")  # type: ignore[union-attr]
        if layout != _EXPECTED_LAYOUT:
            raise ValueError(
                f"face_roi_image.layout must be '{_EXPECTED_LAYOUT}', got '{layout}'"
            )

        dtype = face_roi_image.get("dtype")  # type: ignore[union-attr]
        if dtype != _EXPECTED_DTYPE:
            raise ValueError(
                f"face_roi_image.dtype must be '{_EXPECTED_DTYPE}', got '{dtype}'"
            )

        value_range = face_roi_image.get("value_range")  # type: ignore[union-attr]
        if value_range != _EXPECTED_VALUE_RANGE:
            raise ValueError(
                f"face_roi_image.value_range must be '{_EXPECTED_VALUE_RANGE}', "
                f"got '{value_range}'"
            )

        # Shape consistency: RGB HWC → (height, width, 3)
        expected_shape = (height, width, 3)
        if data.shape != expected_shape:
            raise ValueError(
                f"face_roi_image.data.shape {data.shape} is inconsistent with "
                f"width={width}, height={height}, layout=HWC, color_format=RGB "
                f"(expected {expected_shape})"
            )

    @staticmethod
    def _validate_landmarks(landmarks: object, face_roi_image: object) -> None:
        """Validate all 5 canonical landmark points (spec §2.4)."""
        img_w = face_roi_image.get("width", 0)  # type: ignore[union-attr]
        img_h = face_roi_image.get("height", 0)  # type: ignore[union-attr]

        for key in _LANDMARK_KEYS:
            if key not in landmarks:  # type: ignore[operator]
                raise ValueError(f"landmarks is missing required point: '{key}'")
            pt = landmarks[key]  # type: ignore[index]
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
            # Bounds check using Image struct width/height fields
            if img_w > 0 and img_h > 0:
                if not (0 <= x < img_w and 0 <= y < img_h):
                    raise ValueError(
                        f"landmark '{key}' coordinate ({x}, {y}) is outside "
                        f"face_roi_image bounds ({img_w}×{img_h})"
                    )


# ---------------------------------------------------------------------------
# EnrolledIdentityCache — spec §8.5
# ---------------------------------------------------------------------------


class EnrolledIdentityCache:
    """
    Immutable in-memory store of enrolled face embeddings (spec §8.5).

    Built once at module initialization from the EnrolledIdentity list
    supplied by external startup code.
    Read-only during recognition calls.  No external I/O during recognition.
    """

    def __init__(self, entries: list[EnrolledIdentity]) -> None:
        # Store an immutable copy — no mutation after construction
        self._entries: list[EnrolledIdentity] = list(entries)

    def get_entries(self) -> list[EnrolledIdentity]:
        """Return the enrolled identity entries (read-only view)."""
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
        gallery: list[EnrolledIdentity],
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
            List of enrolled EnrolledIdentity records from EnrolledIdentityCache.

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
            return RecognitionDecision(person_found=False, person_id="UNKNOWN")
        if candidate.similarity >= self._threshold:
            return RecognitionDecision(
                person_found=True,
                person_id=candidate.person_id,
            )
        return RecognitionDecision(person_found=False, person_id="UNKNOWN")


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
        frame_id: str,
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
            person_id=decision.person_id if decision.person_found else "UNKNOWN",
        )


# ---------------------------------------------------------------------------
# FaceRecognitionInterface — formal runtime-checkable Protocol (spec §6.1)
# ---------------------------------------------------------------------------


@runtime_checkable
class FaceRecognitionInterface(Protocol):
    """
    Formal public interface for Face Recognition (spec §6.1).

    FaceRecognitionModule satisfies this Protocol structurally.
    Compliance can be verified at runtime via isinstance().
    """

    def recognize(
        self, face_input: FaceRecognitionInput
    ) -> FaceRecognitionOutput: ...

    def get_input_contract(self) -> PipelineStageInputContract: ...


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
        output = module.recognize(face_input)
    """

    def __init__(
        self,
        config: FaceRecognitionConfig,
        embedding_engine: FaceEmbeddingEngine,
        gallery_entries: list[EnrolledIdentity],
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
            Enrolled identity embeddings supplied by external startup code
            at construction time.  EnrolledIdentityCache is built from these
            and held as immutable in-memory state for the module's lifetime
            (spec §13.1).  The source is outside this module.
        """
        self._config = config

        # Wire subcomponents with injected configuration (spec §8.1, §13.1)
        self._validator = FaceRecognitionInputValidator()
        self._aligner = FaceAligner()
        self._embedding_engine: FaceEmbeddingEngine = embedding_engine
        self._gallery_cache = EnrolledIdentityCache(gallery_entries)
        self._matcher = FaceMatcher()
        self._decision_policy = FaceRecognitionDecisionPolicy(
            recognition_threshold=config.recognition_threshold,
        )
        self._output_builder = FaceRecognitionOutputBuilder()

    # ---- public API (spec §4) --------------------------------------------

    def recognize(
        self, face_input: FaceRecognitionInput
    ) -> FaceRecognitionOutput:
        """
        Identify the person in a pre-cropped face ROI image (spec §4).

        Parameters
        ----------
        face_input:
            Validated pre-prepared face input.  One face per invocation.
            face_roi_image must be a shared Image struct with RGB uint8 HWC
            pixel data (see shared_contracts.md §6).

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

    def get_input_contract(self) -> PipelineStageInputContract:
        """
        Return the pipeline stage input contract for Face Recognition (spec §4).

        Called by external orchestration code during initialization to configure
        how face ROI images must be prepared before passing them to recognize().

        The public contract is RGB uint8 HWC — ArcFaceEmbeddingEngine applies
        mean-normalization to float32 internally; the caller must NOT normalize.
        ResizePolicy.NONE means the caller returns the face crop at its natural
        size; FaceAligner performs recognition-specific geometric alignment
        internally.
        """
        return PipelineStageInputContract(
            output_image_type=OutputImageType.RGB_UINT8_HWC,
            geometry_spec=GeometrySpec(
                width=0,
                height=0,
                resize_policy=ResizePolicy.NONE,
            ),
        )

    # ---- internal pipeline (spec §8.9) -----------------------------------

    def _recognize_internal(
        self, face_input: FaceRecognitionInput
    ) -> FaceRecognitionOutput:
        # Step 1: validation (spec §8.2)
        self._validator.validate(face_input)

        # Step 2: alignment (spec §8.3)
        # Extract raw ndarray from the shared Image struct before passing to
        # FaceAligner — the internal aligner interface accepts np.ndarray only.
        aligned_face: AlignedFace = self._aligner.align(
            face_input["face_roi_image"]["data"],
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
            frame_id=face_input.get("frame_id", ""),
            camera_id=face_input.get("camera_id", ""),  # type: ignore[call-overload]
            timestamp_ms=face_input.get("timestamp_ms", 0),  # type: ignore[call-overload]
            person_found=False,
            person_id="UNKNOWN",
        )
