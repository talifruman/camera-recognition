"""SpatialCoordinator — projects ROI-local bounding boxes to full-frame coordinates.

Implements the mandatory inverse projection formula from
RecognitionPipelineManager spec §8.6 and §8.3.1.

Usage:
    coordinator = SpatialCoordinator()
    full_bbox = coordinator.project_bbox_to_full_frame(
        local_bbox, source_bbox_full_frame, spatial_transform
    )
    if full_bbox is None:
        # invalid inputs — skip this result

Fail-safe contract:
    Returns None (never raises) on any invalid input so that
    PipelineOrchestrator can skip only the affected branch without
    crashing the entire frame pipeline.
"""

from __future__ import annotations

from image_processing.frame_transformation_layer.contracts import SpatialTransform
from image_processing.shared.contracts import BoundingBox


class SpatialCoordinator:
    """Projects ROI-local bounding boxes back to full-frame coordinates.

    Applies the inverse geometry from ``SpatialTransform`` before adding
    the crop-origin offset from ``source_bbox_full_frame``.

    Projection formula (spec §8.3.1):
        1. Remove geometry padding:
               x_unpadded = local_x - spatial_transform.pad_left
               y_unpadded = local_y - spatial_transform.pad_top
        2. Undo geometry scale:
               x_crop = x_unpadded / spatial_transform.scale_x
               y_crop = y_unpadded / spatial_transform.scale_y
               w_crop = local_w / spatial_transform.scale_x
               h_crop = local_h / spatial_transform.scale_y
        3. Translate to full-frame:
               x_full = x_crop + source_bbox_full_frame.x
               y_full = y_crop + source_bbox_full_frame.y

    For ResizePolicy.NONE: scale_x = scale_y = 1.0, pad_left = pad_top = 0,
    so the formula reduces to identity + offset.
    """

    def project_bbox_to_full_frame(
        self,
        local_bbox: BoundingBox,
        source_bbox_full_frame: BoundingBox,
        spatial_transform: SpatialTransform,
    ) -> BoundingBox | None:
        """Project ``local_bbox`` from ROI-local coordinates to full-frame.

        Returns a new ``BoundingBox`` in full-frame coordinates, or ``None``
        when inputs are invalid (zero/negative scale, zero/negative bbox
        dimensions).  The caller must treat ``None`` as "skip this result".

        Args:
            local_bbox: Bounding box in the coordinate space of the
                processed ROI image (output of FTL.get_frame).
            source_bbox_full_frame: The crop region used to request this
                processed frame — expressed in full-frame coordinates.
            spatial_transform: Spatial mapping metadata returned alongside
                the processed frame by the FTL.

        Returns:
            BoundingBox in FULL_FRAME coordinates, or None on invalid input.
        """
        # --- guard: scale must be positive ------------------------------------
        if spatial_transform.scale_x <= 0.0 or spatial_transform.scale_y <= 0.0:
            return None

        # --- guard: local bbox dimensions must be positive --------------------
        local_w = local_bbox["width"]
        local_h = local_bbox["height"]
        if local_w <= 0 or local_h <= 0:
            return None

        # --- guard: source crop dimensions must be positive -------------------
        if source_bbox_full_frame["width"] <= 0 or source_bbox_full_frame["height"] <= 0:
            return None

        local_x = local_bbox["x"]
        local_y = local_bbox["y"]
        pad_left = spatial_transform.pad_left
        pad_top = spatial_transform.pad_top
        scale_x = spatial_transform.scale_x
        scale_y = spatial_transform.scale_y

        # Step 1: remove padding
        x_unpadded = local_x - pad_left
        y_unpadded = local_y - pad_top

        # Step 2: undo scale
        x_crop = x_unpadded / scale_x
        y_crop = y_unpadded / scale_y
        w_crop = local_w / scale_x
        h_crop = local_h / scale_y

        # Step 3: translate to full-frame (integer pixel coordinates)
        x_full = int(round(x_crop + source_bbox_full_frame["x"]))
        y_full = int(round(y_crop + source_bbox_full_frame["y"]))
        w_full = int(round(w_crop))
        h_full = int(round(h_crop))

        if w_full <= 0 or h_full <= 0:
            return None

        return BoundingBox(x=x_full, y=y_full, width=w_full, height=h_full)
