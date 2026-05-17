from __future__ import annotations

from enum import Enum
from typing import TypedDict

import numpy as np


class ResizePolicy(Enum):
    """Spatial resize behavior applied during image preparation.

    See shared_contracts.md §2 for the full contract.
    """

    NONE = "NONE"
    LETTERBOX = "LETTERBOX"


class GeometrySpec(TypedDict):
    """Spatial transformation applied during frame preparation.

    See shared_contracts.md §3 for the full contract.

    Rules:
    - resize_policy NONE  : width and height are ignored.
    - resize_policy LETTERBOX : width and height must be > 0.
    """

    width: int
    height: int
    resize_policy: ResizePolicy


class OutputImageType(Enum):
    """Pixel representation required by a pipeline stage.

    See shared_contracts.md §4 for the full contract per value.
    """

    GRAYSCALE_UINT8_HWC = "GRAYSCALE_UINT8_HWC"
    RGB_UINT8_HWC = "RGB_UINT8_HWC"


class PipelineStageInputContract(TypedDict):
    """Returned by each pipeline stage via get_input_contract().

    See shared_contracts.md §5 for the full contract.

    RPM queries this at initialization; FTL uses it to prepare the model-ready image.
    """

    output_image_type: OutputImageType
    geometry_spec: GeometrySpec


class BoundingBox(TypedDict):
    """Single canonical bounding-box type shared across the entire pipeline.

    See shared_contracts.md §1 for the full contract.
    All coordinates are integers. Coordinate space must be stated at the usage site.
    """

    x: int
    y: int
    width: int
    height: int


class Image(TypedDict):
    """Single canonical public image type shared across the entire pipeline.

    See shared_contracts.md §6 for the full contract.

    Public APIs must NOT expose raw np.ndarray directly. All image payloads
    crossing a public module boundary must be carried in an Image struct.
    """

    data: np.ndarray
    width: int
    height: int
    color_format: str  # "RGB", "BGR", "GRAY"
    layout: str        # "HWC", "CHW"
    dtype: str         # "uint8", "float32"
    value_range: str   # "[0,255]", "[0,1]", "[-1,1]"


class Point(TypedDict):
    """Single canonical pixel coordinate type shared across the entire pipeline.

    See shared_contracts.md §7 for the full contract.
    Coordinate space must be stated by context at the usage site.
    """

    x: int
    y: int


class FaceLandmarks(TypedDict):
    """Canonical 5-point facial landmark struct shared across the entire pipeline.

    See shared_contracts.md §8 for the full contract.
    All coordinates are Point values; coordinate space must be stated by context.
    """

    left_eye: Point
    right_eye: Point
    nose: Point
    mouth_left: Point
    mouth_right: Point
