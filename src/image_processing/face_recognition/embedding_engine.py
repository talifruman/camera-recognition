"""
FaceEmbeddingEngine Protocol and ArcFaceEmbeddingEngine
=======================================================
Defines the embedding engine abstraction and its default ArcFace
implementation, as specified in Face Recognition Module spec §6.

FaceEmbeddingEngine (spec §6.1 / §8.4):
    Abstract interface — any embedding model producing a fixed-dimension
    vector may implement this without touching any other module component.

ArcFaceEmbeddingEngine (spec §6.2):
    Default implementation backed by w600k_mbf.onnx (ArcFace MobileNet
    FaceNet, 512-d output) from the InsightFace buffalo_sc model package.

    The model is auto-downloaded from the official InsightFace GitHub
    release v0.7 if not already present in the configured model directory
    (mirrors the auto-download pattern of SCRFDFaceDetector).

Internal preprocessing (inside ArcFaceEmbeddingEngine — not visible to
callers, spec §2.3 adaptation clause):
    1. Cast aligned face (RGB HWC uint8) to float32.
    2. Normalize: (pixel − 127.5) / 127.5  →  range [-1, 1].
    3. Transpose HWC → CHW.
    4. Add batch dimension → (1, 3, H, W).
    5. Run ONNX inference.
    6. L2-normalize output → (512,) float32 unit vector.

This class does NOT:
    - Compare embeddings to the gallery.
    - Apply thresholds or make accept/reject decisions.
    - Return any data other than the embedding vector.
"""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .aligner import AlignedFace

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Auto-download constants  (same ZIP as SCRFDFaceDetector uses)
# ---------------------------------------------------------------------------

_BUFFALO_SC_URL = (
    "https://github.com/deepinsight/insightface"
    "/releases/download/v0.7/buffalo_sc.zip"
)
_MODEL_ONNX_NAME = "w600k_mbf.onnx"

_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parents[3]  # …/Camera-regogintion
_DEFAULT_MODEL_DIR = _PROJECT_ROOT / "models" / "face_recognition"

# Expected embedding dimensionality produced by w600k_mbf.onnx
_EMBEDDING_DIM = 512


# ---------------------------------------------------------------------------
# FaceEmbeddingEngine Protocol — spec §6.1 / §8.4
# ---------------------------------------------------------------------------


@runtime_checkable
class FaceEmbeddingEngine(Protocol):
    """
    Abstraction for the embedding inference backend (spec §6.1).

    Any class that implements ``extract_embedding`` satisfies this protocol.
    The module depends on this interface, not on ArcFaceEmbeddingEngine.
    """

    def extract_embedding(self, aligned_face: AlignedFace) -> np.ndarray:
        """
        Run embedding inference on an aligned face.

        Parameters
        ----------
        aligned_face:
            112 × 112 RGB uint8 ndarray produced by FaceAligner.

        Returns
        -------
        np.ndarray
            Shape (512,), dtype float32, L2-normalized unit vector.
        """
        ...


# ---------------------------------------------------------------------------
# ArcFaceEmbeddingEngine — spec §6.2, default implementation
# ---------------------------------------------------------------------------


class ArcFaceEmbeddingEngine:
    """
    Default FaceEmbeddingEngine backed by ArcFace (w600k_mbf.onnx).

    Model: ArcFace MobileNet FaceNet (w600k_mbf) from InsightFace buffalo_sc.
    Input: 112 × 112 × 3 RGB face image (aligned by FaceAligner).
    Output: (512,) float32 L2-normalized embedding.

    The ONNX model is auto-downloaded if absent from ``model_dir``.

    Usage:
        engine = ArcFaceEmbeddingEngine()
        embedding = engine.extract_embedding(aligned_face)
    """

    def __init__(self, model_dir: Path | None = None) -> None:
        """
        Load the ArcFace ONNX model.

        Parameters
        ----------
        model_dir:
            Directory where w600k_mbf.onnx is cached.
            Downloaded automatically if not already present.
            Defaults to ``<project_root>/models/face_recognition/``.

        Raises
        ------
        RuntimeError
            If onnxruntime is not installed, or the model cannot be
            downloaded or loaded.
        """
        self._model_dir = Path(model_dir) if model_dir is not None else _DEFAULT_MODEL_DIR
        self._session = self._load_session()

    # ------------------------------------------------------------------
    # FaceEmbeddingEngine interface
    # ------------------------------------------------------------------

    def extract_embedding(self, aligned_face: AlignedFace) -> np.ndarray:
        """
        Extract a 512-d embedding from an aligned face image.

        Internal preprocessing (spec §2.3 adaptation, hidden from callers):
            pixel normalization, HWC→CHW transpose, batch dim, ONNX run,
            L2 normalization.

        Parameters
        ----------
        aligned_face:
            112 × 112 RGB uint8 ndarray (output of FaceAligner).

        Returns
        -------
        np.ndarray
            Shape (512,), dtype float32, L2-normalized.

        Raises
        ------
        RuntimeError
            If ONNX inference fails.
        ValueError
            If aligned_face does not have the expected shape.
        """
        blob = self._preprocess(aligned_face)
        raw_embedding = self._run_inference(blob)
        return self._l2_normalize(raw_embedding)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(aligned_face: AlignedFace) -> np.ndarray:
        """
        Convert 112×112 RGB uint8 → (1, 3, 112, 112) float32 blob.

        Normalization: (pixel − 127.5) / 127.5  →  range [-1, 1].
        Layout: HWC → CHW → NCHW (batch=1).
        """
        if aligned_face.ndim != 3 or aligned_face.shape[:2] != (112, 112):
            raise ValueError(
                f"aligned_face must have shape (112, 112, C), "
                f"got {aligned_face.shape}"
            )
        img = aligned_face.astype(np.float32)
        img = (img - 127.5) / 127.5          # normalize to [-1, 1]
        img = img.transpose(2, 0, 1)         # HWC → CHW
        img = np.expand_dims(img, axis=0)    # CHW → NCHW (batch=1)
        return img

    def _run_inference(self, blob: np.ndarray) -> np.ndarray:
        """
        Run the ONNX session, return the raw (1, 512) output.
        """
        try:
            outputs = self._session.run(None, {self._input_name: blob})
        except Exception as exc:
            raise RuntimeError(
                f"ArcFace embedding inference failed: {exc}"
            ) from exc
        # outputs[0] shape: (1, 512)
        return outputs[0][0].astype(np.float32)

    @staticmethod
    def _l2_normalize(embedding: np.ndarray) -> np.ndarray:
        """L2-normalize the embedding to a unit vector."""
        norm = np.linalg.norm(embedding)
        if norm == 0.0:
            # Extremely unlikely; return zero-safe fallback
            embedding = np.zeros(_EMBEDDING_DIM, dtype=np.float32)
            embedding[0] = 1.0
            return embedding
        return (embedding / norm).astype(np.float32)

    def _load_session(self):  # type: ignore[return]
        """
        Ensure the ONNX model file is present (downloading if needed) then
        create and return the onnxruntime session.
        """
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for real face embedding. "
                "Install it with: pip install onnxruntime"
            ) from exc

        model_path = self._model_dir / _MODEL_ONNX_NAME
        if not model_path.exists():
            self._download_model(model_path)

        _log.info("Loading ArcFace session from '%s' …", model_path)
        try:
            session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load ArcFace ONNX model from '{model_path}': {exc}"
            ) from exc

        self._input_name: str = session.get_inputs()[0].name
        _log.info("ArcFace model ready (output_dim=%d, CPU).", _EMBEDDING_DIM)
        return session

    def _download_model(self, model_path: Path) -> None:
        """
        Download buffalo_sc.zip from the InsightFace GitHub release and
        extract w600k_mbf.onnx into model_dir.

        Uses the same download URL as SCRFDFaceDetector (det_500m.onnx).
        """
        model_path.parent.mkdir(parents=True, exist_ok=True)
        _log.info(
            "ArcFace model not found at '%s'. Downloading from '%s' …",
            model_path,
            _BUFFALO_SC_URL,
        )
        print(
            f"[ArcFaceEmbeddingEngine] Downloading ArcFace model from:\n  {_BUFFALO_SC_URL}"
        )

        try:
            response = urllib.request.urlopen(_BUFFALO_SC_URL, timeout=180)
            zip_data = response.read()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download ArcFace model from '{_BUFFALO_SC_URL}'. "
                f"Check your network connection. Error: {exc}"
            ) from exc

        try:
            with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                if _MODEL_ONNX_NAME not in zf.namelist():
                    raise RuntimeError(
                        f"'{_MODEL_ONNX_NAME}' not found in downloaded ZIP. "
                        f"Contents: {zf.namelist()}"
                    )
                zf.extract(_MODEL_ONNX_NAME, path=str(model_path.parent))
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Failed to extract '{_MODEL_ONNX_NAME}' from downloaded ZIP: {exc}"
            ) from exc

        if not model_path.exists() or model_path.stat().st_size == 0:
            raise RuntimeError(
                f"Model file '{model_path}' is missing or empty after extraction."
            )
        print(
            f"[ArcFaceEmbeddingEngine] Model saved to: {model_path} "
            f"({model_path.stat().st_size // 1024} KB)"
        )
