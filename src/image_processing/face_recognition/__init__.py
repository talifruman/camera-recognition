from __future__ import annotations

from .aligner import AlignedFace, FaceAligner
from .embedding_engine import ArcFaceEmbeddingEngine, FaceEmbeddingEngine
from .module import (
    FaceEmbedding,
    FaceGalleryCache,
    FaceLandmarks,
    FaceMatcher,
    FaceRecognitionConfig,
    FaceRecognitionDecisionPolicy,
    FaceRecognitionInput,
    FaceRecognitionInputValidator,
    FaceRecognitionModule,
    FaceRecognitionOutput,
    FaceRecognitionOutputBuilder,
    GalleryEntry,
    MatchCandidate,
    Point,
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
    # module pipeline components
    "FaceEmbedding",
    "FaceGalleryCache",
    "FaceLandmarks",
    "FaceMatcher",
    "FaceRecognitionConfig",
    "FaceRecognitionDecisionPolicy",
    "FaceRecognitionInput",
    "FaceRecognitionInputValidator",
    "FaceRecognitionModule",
    "FaceRecognitionOutput",
    "FaceRecognitionOutputBuilder",
    "GalleryEntry",
    "MatchCandidate",
    "Point",
    "RecognitionDecision",
]
