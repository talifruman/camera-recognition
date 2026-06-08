"""
FaceAligner
===========
Produces a geometrically aligned (112 × 112, RGB uint8) face image from a
raw ROI image and its canonical 5-point FaceLandmarks.

This is the alignment stage described in Face Recognition Module spec §8.3.

Responsibilities (spec §8.3):
- Accept face_roi_image (RGB HWC uint8 ndarray) and FaceLandmarks.
- Apply a similarity transform that maps the 5 detected landmarks to the
  standard ArcFace reference template positions.
- Return AlignedFace — a 112 × 112 RGB uint8 ndarray ready for embedding.

This class does NOT:
- Run inference of any kind.
- Access the gallery.
- Apply thresholds or make accept/reject decisions.
- Do any colour conversion, normalization beyond the geometric warp.

Reference template: standard ArcFace 5-point template used by InsightFace
(arcface_dst, 112 × 112 target size).
"""

from __future__ import annotations

import numpy as np

# AlignedFace is a 112×112 RGB uint8 numpy array.
AlignedFace = np.ndarray

# Standard ArcFace 5-point reference template.
# Order: left_eye, right_eye, nose, mouth_left, mouth_right
# Source: InsightFace face_align.py arcface_dst constant
_ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

_ALIGNED_SIZE: tuple[int, int] = (112, 112)


class FaceAligner:
    """
    Aligns a face ROI to the standard ArcFace 112 × 112 template.

    Usage:
        aligner = FaceAligner()
        aligned = aligner.align(roi_image, landmarks)
    """

    def align(
        self,
        face_roi_image: np.ndarray,
        landmarks: dict,
    ) -> AlignedFace:
        """
        Apply a similarity transform to align the detected face.

        Parameters
        ----------
        face_roi_image:
            Input image as RGB HWC uint8 ndarray.  Any size; the warp
            maps directly from this image to the 112 × 112 output.
        landmarks:
            FaceLandmarks TypedDict with keys:
                left_eye, right_eye, nose, mouth_left, mouth_right
            Each value is a Point TypedDict with keys ``x`` and ``y``.

        Returns
        -------
        AlignedFace
            112 × 112 RGB uint8 ndarray.

        Raises
        ------
        ValueError
            If landmarks are missing expected keys or the transform
            cannot be estimated (degenerate geometry).
        """
        src_pts = self._extract_src_points(landmarks)
        M = self._estimate_transform(src_pts)
        return self._warp(face_roi_image, M)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_src_points(landmarks: dict) -> np.ndarray:
        """Convert FaceLandmarks TypedDict → (5, 2) float32 array."""
        try:
            pts = np.array(
                [
                    [landmarks["left_eye"]["x"], landmarks["left_eye"]["y"]],
                    [landmarks["right_eye"]["x"], landmarks["right_eye"]["y"]],
                    [landmarks["nose"]["x"], landmarks["nose"]["y"]],
                    [landmarks["mouth_left"]["x"], landmarks["mouth_left"]["y"]],
                    [landmarks["mouth_right"]["x"], landmarks["mouth_right"]["y"]],
                ],
                dtype=np.float32,
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"FaceLandmarks is missing expected keys: {exc}"
            ) from exc
        return pts

    @staticmethod
    def _estimate_transform(src_pts: np.ndarray) -> np.ndarray:
        """
        Estimate the 2×3 similarity (partial affine) transform matrix that
        maps src_pts to the ArcFace reference template.

        Uses cv2.estimateAffinePartial2D (similarity transform: rotation +
        uniform scale + translation — no shear or independent x/y scaling).
        """
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python is required for face alignment. "
                "Install it with: pip install opencv-python"
            ) from exc

        M, inliers = cv2.estimateAffinePartial2D(
            src_pts,
            _ARCFACE_DST,
            method=cv2.LMEDS,
        )
        if M is None:
            raise ValueError(
                "Similarity transform estimation failed (degenerate landmarks). "
                "The detected landmarks may be too close together or colinear."
            )
        return M.astype(np.float32)

    @staticmethod
    def _warp(image: np.ndarray, M: np.ndarray) -> AlignedFace:
        """Apply the 2×3 affine warp and return a 112×112 RGB uint8 image."""
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python is required for face alignment. "
                "Install it with: pip install opencv-python"
            ) from exc

        aligned = cv2.warpAffine(
            image,
            M,
            _ALIGNED_SIZE,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return aligned.astype(np.uint8)
