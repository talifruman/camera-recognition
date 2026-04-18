"""
StubFaceEmbeddingEngine
=======================
Deterministic stub implementation of FaceEmbeddingEngine, used during the
STUB phase of the Face Recognition module.

Spec reference: Face Recognition Module spec §6.1, §6.3, §8.4.

Purpose:
    Replace only this component with ArcFaceEmbeddingEngine for the REAL
    phase.  Every other module component (module.py) remains unchanged.

Stub behaviour:
    - Implements the FaceEmbeddingEngine interface exactly.
    - Accepts AlignedFace (np.ndarray) and returns a FaceEmbedding.
    - Deterministic: the same aligned face image bytes always produce the
      same embedding.  SHA-256 of the raw image bytes seeds a NumPy RNG.
    - Returns a 512-dimensional L2-normalised float32 unit vector.
    - Does NOT run any model inference.
    - Does NOT access the gallery or perform any comparison.
    - Does NOT apply thresholds or make accept/reject decisions.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .aligner import AlignedFace

# Embedding dimension — must match gallery embedding dimension (512).
_STUB_EMBEDDING_DIM: int = 512


class StubFaceEmbeddingEngine:
    """
    Deterministic stub embedding engine (spec §6.1).

    Satisfies the FaceEmbeddingEngine protocol without running any model.
    Determinism is guaranteed by hashing the raw image bytes to derive an
    RNG seed, so the same pixel content always maps to the same embedding.

    Intended for STUB phase only.  Replace with ArcFaceEmbeddingEngine
    for the REAL phase without changing any other module component.
    """

    def extract_embedding(self, aligned_face: AlignedFace) -> np.ndarray:
        """
        Produce a deterministic 512-d unit vector from an aligned face.

        Parameters
        ----------
        aligned_face:
            112 × 112 RGB uint8 ndarray produced by FaceAligner.

        Returns
        -------
        np.ndarray
            Shape (_STUB_EMBEDDING_DIM,), dtype float32, L2-normalised.
        """
        seed = self._seed_from_image(aligned_face)
        raw = self._generate_raw_vector(seed)
        return self._l2_normalize(raw)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _seed_from_image(aligned_face: AlignedFace) -> int:
        """Derive a deterministic integer seed from the raw image bytes."""
        digest = hashlib.sha256(aligned_face.tobytes()).digest()
        # Use the first 8 bytes of the digest as a 64-bit unsigned integer
        return int.from_bytes(digest[:8], byteorder="little")

    @staticmethod
    def _generate_raw_vector(seed: int) -> np.ndarray:
        """Generate a (_STUB_EMBEDDING_DIM,) float32 vector from a seeded RNG."""
        rng = np.random.default_rng(seed)
        return rng.standard_normal(_STUB_EMBEDDING_DIM).astype(np.float32)

    @staticmethod
    def _l2_normalize(vector: np.ndarray) -> np.ndarray:
        """L2-normalise vector to a unit vector."""
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            # Extremely unlikely with a seeded random vector; safe fallback
            result = np.zeros(_STUB_EMBEDDING_DIM, dtype=np.float32)
            result[0] = 1.0
            return result
        return (vector / norm).astype(np.float32)
