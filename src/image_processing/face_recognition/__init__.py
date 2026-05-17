from __future__ import annotations

from image_processing.shared.contracts import FaceLandmarks, Point

from .aligner import AlignedFace, FaceAligner
from .embedding_engine import ArcFaceEmbeddingEngine, FaceEmbeddingEngine
from .module import (
    FaceEmbedding,
    EnrolledIdentityCache,
    FaceMatcher,
    FaceRecognitionConfig,
    FaceRecognitionDecisionPolicy,
    FaceRecognitionInput,
    FaceRecognitionInputValidator,
    FaceRecognitionInterface,
    FaceRecognitionModule,
    FaceRecognitionOutput,
    FaceRecognitionOutputBuilder,
    EnrolledIdentity,
    MatchCandidate,
    RecognitionDecision,
)
from .stub_engine import StubFaceEmbeddingEngine

__all__ = [
    # aligner
    "AlignedFace",
    "FaceAligner",
    # embedding engines
    "ArcFaceEmbeddingEngine",
    "FaceEmbeddingEngine",
    "StubFaceEmbeddingEngine",
    # shared types (re-exported for convenience)
    "FaceLandmarks",
    "Point",
    # module pipeline components
    "FaceEmbedding",
    "EnrolledIdentityCache",
    "FaceMatcher",
    "FaceRecognitionConfig",
    "FaceRecognitionDecisionPolicy",
    "FaceRecognitionInput",
    "FaceRecognitionInputValidator",
    "FaceRecognitionInterface",
    "FaceRecognitionModule",
    "FaceRecognitionOutput",
    "FaceRecognitionOutputBuilder",
    "EnrolledIdentity",
    "MatchCandidate",
    "RecognitionDecision",
]
