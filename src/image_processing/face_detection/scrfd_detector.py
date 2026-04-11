"""
SCRFDFaceDetector
=================
Real implementation of FaceDetectorEngine using SCRFD via pure onnxruntime.

This is the default production implementation of FaceDetectorEngine defined
in the Face Detection markdown spec §5.3 / §9.2.

Responsibilities (spec §5.3):
- Accept the validated ROI image directly from the orchestrator.
- Run SCRFD inference.
- Return raw face detections in ROI coordinates: bounding boxes,
  confidence scores, and 5-point landmarks.

This class does NOT:
- Apply confidence thresholds (FaceDetectionPostprocessor owns that).
- Project coordinates (FaceCoordinateProjector owns that).
- Construct output structures (FaceDetectionOutputBuilder owns that).

The model (det_500m.onnx, SCRFD-500M with 5-keypoint support from the
InsightFace buffalo_sc model pack) is downloaded automatically from the
official InsightFace GitHub release if the ONNX file is not already
present at the configured model_dir.

SCRFD keypoint order: left_eye, right_eye, nose, mouth_left, mouth_right
This matches the canonical FaceLandmarks structure directly — no remapping
is needed.

Any runtime-specific adaptation (e.g. BGR conversion, blob creation)
is performed internally here only and is never visible to callers.
Per spec §2.3 the ROI image arrives as RGB HWC uint8; the SCRFD backend
expects BGR, so the flip is done inside the engine, not in the outer module.

Reference SCRFD inference: InsightFace detection/scrfd/tools/scrfd.py
(Apache-2.0 licence, https://github.com/deepinsight/insightface)
"""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .module import BoundingBox, Point, RawFaceDetection

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Download constants
# ---------------------------------------------------------------------------

# Official InsightFace release that contains det_500m.onnx (SCRFD-500M + KPS)
_BUFFALO_SC_URL = (
    "https://github.com/deepinsight/insightface"
    "/releases/download/v0.7/buffalo_sc.zip"
)
# ONNX file inside the ZIP that is the face detector
_MODEL_ONNX_NAME = "det_500m.onnx"

# Root of this file: .../src/image_processing/face_detection/
_MODULE_DIR = Path(__file__).resolve().parent
# Absolute project root: .../Camera-regogintion/
_PROJECT_ROOT = _MODULE_DIR.parents[2]
# Default model cache directory inside the repository
_DEFAULT_MODEL_DIR = _PROJECT_ROOT / "models" / "face_detection"

# Default inference input size for SCRFD (width, height)
_DEFAULT_INPUT_SIZE: tuple[int, int] = (640, 640)


# ---------------------------------------------------------------------------
# SCRFD geometry helpers  (ported from InsightFace scrfd.py, Apache-2.0)
# ---------------------------------------------------------------------------


def _distance2bbox(anchor_centers: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Decode distance prediction offsets to (x1, y1, x2, y2) bounding boxes."""
    x1 = anchor_centers[:, 0] - distance[:, 0]
    y1 = anchor_centers[:, 1] - distance[:, 1]
    x2 = anchor_centers[:, 0] + distance[:, 2]
    y2 = anchor_centers[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(anchor_centers: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Decode distance offsets to (x, y) keypoints, shape (N, K*2)."""
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = anchor_centers[:, i % 2] + distance[:, i]
        py = anchor_centers[:, i % 2 + 1] + distance[:, i + 1]
        preds.extend([px, py])
    return np.stack(preds, axis=-1)


def _nms(dets: np.ndarray, thresh: float) -> list[int]:
    """Non-maximum suppression on (x1, y1, x2, y2, score) detections."""
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        ovr = (w * h) / (areas[i] + areas[order[1:]] - w * h)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


# ---------------------------------------------------------------------------
# SCRFDSession — thin onnxruntime wrapper around the SCRFD ONNX model
# ---------------------------------------------------------------------------


class _SCRFDSession:
    """
    Loads and holds the SCRFD ONNX session.  Exposes ``forward()`` and
    ``detect()`` methods that mirror InsightFace's SCRFD class exactly,
    without depending on the insightface Python package.
    """

    _NMS_THRESH = 0.4

    def __init__(self, model_path: Path, input_size: tuple[int, int]) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for real face detection. "
                "Install it with: pip install onnxruntime"
            ) from exc

        session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self._session = session
        self._input_name: str = session.get_inputs()[0].name
        self._output_names: list[str] = [o.name for o in session.get_outputs()]
        self._center_cache: dict[tuple[int, int, int], np.ndarray] = {}
        self._input_size = input_size  # (width, height)
        self._init_vars()

    def _init_vars(self) -> None:
        """Detect model variant (num outputs → FPN strides, anchor count, KPS)."""
        n = len(self._output_names)
        self._use_kps = False
        self._num_anchors = 1
        if n == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif n == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self._use_kps = True
        elif n == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
        elif n == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._use_kps = True

    def forward(
        self, img: np.ndarray, thresh: float
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """Run the ONNX session on a padded BGR image, decode anchors."""
        scores_list: list[np.ndarray] = []
        bboxes_list: list[np.ndarray] = []
        kpss_list: list[np.ndarray] = []

        input_h, input_w = img.shape[:2]
        # cv2.dnn.blobFromImage: subtract mean (127.5), scale (1/128), NCHW, RGB swap
        blob = cv2.dnn.blobFromImage(
            img, 1.0 / 128.0, (input_w, input_h), (127.5, 127.5, 127.5), swapRB=True
        )
        net_outs = self._session.run(self._output_names, {self._input_name: blob})

        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + self.fmc] * stride
            if self._use_kps:
                kps_preds = net_outs[idx + self.fmc * 2] * stride

            height = input_h // stride
            width = input_w // stride
            key = (height, width, stride)
            if key in self._center_cache:
                anchor_centers = self._center_cache[key]
            else:
                anchor_centers = (
                    np.stack(np.mgrid[:height, :width][::-1], axis=-1)
                    .astype(np.float32)
                    .reshape(-1, 2)
                    * stride
                )
                if self._num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * self._num_anchors, axis=1
                    ).reshape((-1, 2))
                if len(self._center_cache) < 100:
                    self._center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= thresh)[0]
            bboxes = _distance2bbox(anchor_centers, bbox_preds)
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])
            if self._use_kps:
                kpss = _distance2kps(anchor_centers, kps_preds).reshape(-1, 5, 2)
                kpss_list.append(kpss[pos_inds])

        return scores_list, bboxes_list, kpss_list

    def detect(
        self, img_bgr: np.ndarray, thresh: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Resize/pad img_bgr to self._input_size, run forward, NMS.

        Returns
        -------
        bboxes : ndarray shape (N, 5)  — [x1, y1, x2, y2, score]
        kpss   : ndarray shape (N, 5, 2) or None
        """
        input_w, input_h = self._input_size
        im_ratio = float(img_bgr.shape[0]) / float(img_bgr.shape[1])
        model_ratio = float(input_h) / float(input_w)
        if im_ratio > model_ratio:
            new_h = input_h
            new_w = int(new_h / im_ratio)
        else:
            new_w = input_w
            new_h = int(new_w * im_ratio)
        det_scale = float(new_h) / float(img_bgr.shape[0])

        resized = cv2.resize(img_bgr, (new_w, new_h))
        det_img = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized

        scores_list, bboxes_list, kpss_list = self.forward(det_img, thresh)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32)
        pre_det = pre_det[order]
        keep = _nms(pre_det, self._NMS_THRESH)
        det = pre_det[keep]

        kpss: np.ndarray | None = None
        if self._use_kps and kpss_list:
            kpss_arr = np.vstack(kpss_list) / det_scale
            kpss_arr = kpss_arr[order]
            kpss = kpss_arr[keep]

        return det, kpss


# ---------------------------------------------------------------------------
# SCRFDFaceDetector — public FaceDetectorEngine implementation
# ---------------------------------------------------------------------------


class SCRFDFaceDetector:
    """
    Real FaceDetectorEngine backed by SCRFD (SCRFD-500M with 5 keypoints).

    Implements the FaceDetectorEngine protocol (spec §5.3).

    Model: det_500m.onnx from the InsightFace buffalo_sc model package.
    Download: automatic from the official InsightFace GitHub release v0.7.

    The model is loaded once at construction time and reused for every
    call to detect().  The model is never reloaded per image.
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        input_size: tuple[int, int] = _DEFAULT_INPUT_SIZE,
    ) -> None:
        """
        Initialise and load the SCRFD model.

        Parameters
        ----------
        model_dir:
            Directory where the ONNX file is cached.
            Downloaded automatically if not already present.
            Defaults to ``<project_root>/models/face_detection/``.
        input_size:
            (width, height) used for inference.  Default: ``(640, 640)``.

        Raises
        ------
        RuntimeError
            If onnxruntime is not installed, or the model cannot be
            downloaded or loaded.
        """
        self._model_dir = Path(model_dir) if model_dir is not None else _DEFAULT_MODEL_DIR
        self._input_size = input_size
        self._session: _SCRFDSession = self._load_session()

    # ------------------------------------------------------------------
    # FaceDetectorEngine interface (spec §5.3)
    # ------------------------------------------------------------------

    def detect(self, roi_image: np.ndarray) -> list[RawFaceDetection]:
        """
        Run SCRFD inference on a single ROI image.

        Parameters
        ----------
        roi_image:
            Validated person ROI, RGB HWC uint8 (already validated by
            FaceDetectionInputValidator).

        Returns
        -------
        list[RawFaceDetection]
            Raw face detections in ROI-local coordinates.  Confidence
            thresholds are NOT applied here — FaceDetectionPostprocessor
            owns that responsibility.  On inference error returns [].
        """
        try:
            return self._run_inference(roi_image)
        except Exception as exc:  # noqa: BLE001
            _log.error("SCRFD inference failed: %s", exc, exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_session(self) -> _SCRFDSession:
        """
        Ensure the ONNX model file is present (downloading if needed) then
        create and return the onnxruntime session.

        Raises RuntimeError on any failure so the caller receives a clear
        initialisation error (spec §10: init errors separate from per-image).
        """
        model_path = self._model_dir / _MODEL_ONNX_NAME
        if not model_path.exists():
            self._download_model(model_path)

        _log.info("Loading SCRFD session from '%s' …", model_path)
        try:
            session = _SCRFDSession(model_path, self._input_size)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load SCRFD ONNX model from '{model_path}': {exc}"
            ) from exc

        _log.info("SCRFD model ready (input_size=%s, CPU).", self._input_size)
        return session

    def _download_model(self, model_path: Path) -> None:
        """
        Download buffalo_sc.zip from the InsightFace GitHub release and
        extract det_500m.onnx into model_dir.

        Raises RuntimeError if the download or extraction fails.
        """
        model_path.parent.mkdir(parents=True, exist_ok=True)
        _log.info(
            "SCRFD model not found at '%s'. Downloading from '%s' …",
            model_path,
            _BUFFALO_SC_URL,
        )
        print(
            f"[SCRFDFaceDetector] Downloading SCRFD model from:\n  {_BUFFALO_SC_URL}"
        )

        try:
            response = urllib.request.urlopen(_BUFFALO_SC_URL, timeout=120)
            zip_data = response.read()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download SCRFD model from '{_BUFFALO_SC_URL}'. "
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
            f"[SCRFDFaceDetector] Model saved to: {model_path} "
            f"({model_path.stat().st_size // 1024} KB)"
        )

    def _run_inference(self, roi_image: np.ndarray) -> list[RawFaceDetection]:
        """
        Convert RGB ROI to BGR, run SCRFD detect(), parse bboxes + kpss.

        Internal backend adaptation only — not image preprocessing.
        The validated RGB ROI is converted to BGR as required by the SCRFD
        ONNX backend (spec §2.3: runtime adaptations stay inside the engine).

        Threshold is 0.0 so ALL raw detections are returned; the
        FaceDetectionPostprocessor applies the configured threshold.
        """
        # Internal backend adaptation: RGB → BGR
        # (engine-internal only, not module-level preprocessing per spec §2.3)
        bgr_image = roi_image[:, :, ::-1].copy()

        # thresh=0.0: return all detections; FaceDetectionPostprocessor filters.
        bboxes, kpss = self._session.detect(bgr_image, thresh=0.0)

        if bboxes is None or len(bboxes) == 0:
            return []

        detections: list[RawFaceDetection] = []
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2, score = (
                float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4])
            )
            face_bbox: BoundingBox = {
                "x": int(round(x1)),
                "y": int(round(y1)),
                "width": max(0, int(round(x2 - x1))),
                "height": max(0, int(round(y2 - y1))),
            }

            # SCRFD keypoint order: left_eye, right_eye, nose, mouth_left,
            # mouth_right — matches canonical FaceLandmarks exactly (spec §5.3)
            landmarks: list[Point] = []
            if kpss is not None and i < len(kpss):
                for kp in kpss[i]:
                    landmarks.append(
                        {"x": int(round(float(kp[0]))), "y": int(round(float(kp[1])))}
                    )
            else:
                # No keypoints in model — zero fallback so OutputBuilder can always
                # build the canonical 5-point structure.
                landmarks = [{"x": 0, "y": 0}] * 5

            detections.append(
                RawFaceDetection(
                    bbox=face_bbox,
                    confidence=score,
                    landmarks=landmarks,
                )
            )

        return detections
