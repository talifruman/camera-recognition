from __future__ import annotations

from enum import Enum
from typing import TypedDict

import numpy as np


class ResizePolicy(Enum):
    """Spatial resize behavior applied during image preparation.

    See shared_contracts.md §2 for the full contract.
    """

    NONE = "NONE"
    STRETCH = "STRETCH"
    LETTERBOX = "LETTERBOX"
    PRESERVE_ASPECT_RATIO = "PRESERVE_ASPECT_RATIO"


class GeometrySpec(TypedDict):
    """Spatial transformation applied during frame preparation.

    See shared_contracts.md §3 for the full contract.

    Rules:
    - resize_policy NONE  : width and height are ignored.
    - resize_policy STRETCH / LETTERBOX / PRESERVE_ASPECT_RATIO : width and height must be > 0.
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
    RGB_FLOAT32_HWC_NORMALIZED_0_TO_1 = "RGB_FLOAT32_HWC_NORMALIZED_0_TO_1"
    RGB_FLOAT32_HWC_NORMALIZED_MINUS1_TO_1 = "RGB_FLOAT32_HWC_NORMALIZED_MINUS1_TO_1"


class PipelineStageInputContract(TypedDict):
    """Returned by each pipeline stage via get_input_contract().

    See shared_contracts.md §5 for the full contract.

    RPM queries this at initialization; FTL uses it to prepare the model-ready image.
    """

    output_image_type: OutputImageType
    geometry_spec: GeometrySpec


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
