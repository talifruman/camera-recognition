"""Public boundary types for the Frame Transformation Layer.

These types form the stable public contract between the FTL and its callers
(e.g., RecognitionPipelineManager).  They carry no FTL-internal
implementation details and may be imported without pulling in heavy
runtime dependencies such as PIL or NumPy-based processors.

Callers (e.g., RPM) must import from this module, NOT from module.py:

    from image_processing.frame_transformation_layer.contracts import (
        FramePacket,
        FrameTemporalSelector,
        ProcessedFrame,
        SpatialTransform,
        PreviousFrameNotAvailableError,
        FrameNotFoundError,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

try:
    from ..shared.contracts import (
        BoundingBox as SharedBoundingBox,
        FramePacket,
        Image as SharedImage,
    )
except ImportError:  # pragma: no cover — fallback when imported outside package
    from src.image_processing.shared.contracts import (  # type: ignore[no-redef]
        BoundingBox as SharedBoundingBox,
        FramePacket,
        Image as SharedImage,
    )


# ---------------------------------------------------------------------------
# Temporal selector — shared between FTL and RPM
# ---------------------------------------------------------------------------


class FrameTemporalSelector(Enum):
    """Selects which stored frame to retrieve for a given camera."""

    CURRENT = "CURRENT"
    PREVIOUS = "PREVIOUS"


# ---------------------------------------------------------------------------
# Public errors — raised by FTL, caught by RPM
# ---------------------------------------------------------------------------


class PreviousFrameNotAvailableError(Exception):
    """``get_frame(camera_id, PREVIOUS, …)`` called before two successful ingests.

    RPM treats this as a cold-start condition and returns an empty output;
    it is never propagated to the caller as an error.
    """


class FrameNotFoundError(Exception):
    """``get_frame`` called for a camera with no CURRENT frame stored yet."""


# ---------------------------------------------------------------------------
# SpatialTransform — spatial mapping metadata returned with every ProcessedFrame
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpatialTransform:
    """Encodes the geometry applied by the FTL when producing a ``ProcessedFrame``.

    Used by ``SpatialCoordinator`` to project ROI-local coordinates back to
    full-frame coordinates.

    For ``ResizePolicy.NONE``: ``scale_x = scale_y = 1.0``,
    ``pad_left = pad_top = 0`` — inverse projection reduces to identity + offset.

    For ``ResizePolicy.LETTERBOX``: all fields are non-trivial; inverse
    projection must remove padding, undo scale, then add crop origin.
    """

    scale_x: float
    scale_y: float
    pad_left: int
    pad_top: int
    output_width: int
    output_height: int


# ---------------------------------------------------------------------------
# ProcessedFrame — result of ``get_frame``
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessedFrame:
    """Result of ``FrameTransformationLayerInterface.get_frame``.

    ``image.data`` is treated as **read-only** by all callers.  RPM and
    downstream pipeline stages must never mutate it.

    ``source_bbox_full_frame`` is the requested crop region expressed in
    full-frame coordinates.  ``spatial_transform`` encodes the geometry
    applied inside the FTL to produce ``image``.
    """

    frame_id: str
    timestamp_ms: int
    image: SharedImage           # read-only; shared-buffer contract
    source_bbox_full_frame: SharedBoundingBox
    spatial_transform: SpatialTransform
