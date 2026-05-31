"""Frame Transformation Layer (FTL) — real implementation.

Implements the public API, validation, contracts, storage, and full
pixel-level processing defined in
``doc/image_processing_service/frame_transformation_layer.md``.

CropProcessor performs real HWC byte slicing via NumPy.
FrameConverter performs real geometry (NONE / LETTERBOX) and
pixel-format conversion (RGB uint8, grayscale uint8)
via Pillow.  SpatialTransform values follow the MD formulas exactly.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

try:
    from ..shared.contracts import (
        BoundingBox as SharedBoundingBox,
        FramePacket,
        GeometrySpec as SharedGeometrySpec,
        Image as SharedImage,
        OutputImageType,
        ResizePolicy,
    )
    from .contracts import (
        FrameNotFoundError,
        FrameTemporalSelector,
        PreviousFrameNotAvailableError,
        ProcessedFrame,
        SpatialTransform,
    )
except ImportError:  # pragma: no cover - fallback when imported outside package
    from src.image_processing.shared.contracts import (  # type: ignore[no-redef]
        BoundingBox as SharedBoundingBox,
        FramePacket,
        GeometrySpec as SharedGeometrySpec,
        Image as SharedImage,
        OutputImageType,
        ResizePolicy,
    )
    from src.image_processing.frame_transformation_layer.contracts import (  # type: ignore[no-redef]
        FrameNotFoundError,
        FrameTemporalSelector,
        PreviousFrameNotAvailableError,
        ProcessedFrame,
        SpatialTransform,
    )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

# FrameTemporalSelector, FramePacket, SpatialTransform, ProcessedFrame,
# PreviousFrameNotAvailableError, and FrameNotFoundError are defined in
# .contracts and imported above.  They remain part of the public API.


class ValidationError(Exception):
    """Missing or inconsistent required metadata, or invalid ``GeometrySpec``."""


class InvalidFramePacketFormatError(Exception):
    """``FramePacket`` does not satisfy the canonical input format."""


class InvalidCropBboxError(Exception):
    """``region_bbox`` has invalid (zero or negative) dimensions."""


class CropOutOfBoundsError(Exception):
    """``region_bbox`` extends outside the full frame boundaries."""


class UnsupportedOutputImageTypeError(Exception):
    """``output_type`` is not a recognized ``OutputImageType`` enum value."""


class ConversionError(Exception):
    """``FrameConverter`` failed to apply the conversion contract."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned rectangle. Origin top-left; right/bottom edges exclusive."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class RGBColor:
    r: int = 0
    g: int = 0
    b: int = 0


@dataclass(frozen=True)
class GeometrySpec:
    """Spatial transformation spec supplied by the caller of ``get_frame``."""

    resize_policy: ResizePolicy
    width: Optional[int] = None
    height: Optional[int] = None
    padding_color: RGBColor = field(default_factory=RGBColor)


@dataclass(frozen=True)
class ImageConversionContract:
    """Pixel-format-only conversion contract."""

    color_format: str
    layout: str
    dtype: str
    value_range: str


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class StoredFrame:
    """Wraps a full-frame shared ``Image`` with identifying metadata for ``FrameStore``."""

    frame_id: str
    camera_id: str
    timestamp_ms: int
    image: SharedImage


@dataclass
class CameraFrameState:
    """Mutable per-camera CURRENT/PREVIOUS frame state."""

    current: Optional[StoredFrame] = None
    previous: Optional[StoredFrame] = None


# ---------------------------------------------------------------------------
# Hardcoded OutputImageType -> ImageConversionContract mapping (MD source)
# ---------------------------------------------------------------------------


_OUTPUT_IMAGE_TYPE_CONTRACTS: dict[OutputImageType, ImageConversionContract] = {
    OutputImageType.GRAYSCALE_UINT8_HWC: ImageConversionContract(
        color_format="GRAY",
        layout="HWC",
        dtype="uint8",
        value_range="[0,255]",
    ),
    OutputImageType.RGB_UINT8_HWC: ImageConversionContract(
        color_format="RGB",
        layout="HWC",
        dtype="uint8",
        value_range="[0,255]",
    ),
}


# ---------------------------------------------------------------------------
# FramePacketValidator
# ---------------------------------------------------------------------------


class FramePacketValidator:
    """Validates all ``FramePacket`` fields per the MD §Validation table.

    Metadata problems (missing/empty fields, non-positive dimensions) raise
    :class:`ValidationError`. Canonical-format violations (wrong
    ``pixel_format``, ``layout``, channels, bit depth, byte size) raise
    :class:`InvalidFramePacketFormatError`.
    """

    def validate(self, frame_packet: FramePacket) -> ValidationResult:
        # --- required-metadata checks → ValidationError ---
        if not isinstance(frame_packet.frame_id, str) or frame_packet.frame_id == "":
            raise ValidationError("frame_id must be a non-empty string")
        if (
            not isinstance(frame_packet.camera_id, str)
            or frame_packet.camera_id == ""
        ):
            raise ValidationError("camera_id must be a non-empty string")
        if frame_packet.timestamp_ms is None or not isinstance(
            frame_packet.timestamp_ms, int
        ):
            raise ValidationError("timestamp_ms must be present and an integer")
        if not isinstance(frame_packet.width, int) or frame_packet.width <= 0:
            raise ValidationError("width must be > 0")
        if not isinstance(frame_packet.height, int) or frame_packet.height <= 0:
            raise ValidationError("height must be > 0")
        if frame_packet.image_bytes is None or len(frame_packet.image_bytes) == 0:
            raise ValidationError("image_bytes must be present and non-empty")

        # --- canonical-format checks → InvalidFramePacketFormatError ---
        if frame_packet.pixel_format != "RGB":
            raise InvalidFramePacketFormatError(
                f"pixel_format must be 'RGB' (got {frame_packet.pixel_format!r})"
            )
        if frame_packet.layout != "HWC":
            raise InvalidFramePacketFormatError(
                f"layout must be 'HWC' (got {frame_packet.layout!r})"
            )
        if frame_packet.dtype != "uint8":
            raise InvalidFramePacketFormatError(
                f"dtype must be 'uint8' (got {frame_packet.dtype!r})"
            )
        if frame_packet.value_range != "[0,255]":
            raise InvalidFramePacketFormatError(
                f"value_range must be '[0,255]' (got {frame_packet.value_range!r})"
            )
        if frame_packet.num_color_channels != 3:
            raise InvalidFramePacketFormatError(
                f"num_color_channels must be 3 (got {frame_packet.num_color_channels})"
            )
        if frame_packet.bits_per_channel != 8:
            raise InvalidFramePacketFormatError(
                f"bits_per_channel must be 8 (got {frame_packet.bits_per_channel})"
            )
        if frame_packet.packing != "tightly_packed":
            raise InvalidFramePacketFormatError(
                "packing must be 'tightly_packed' (no stride/row padding allowed) "
                f"(got {frame_packet.packing!r})"
            )

        expected = frame_packet.width * frame_packet.height * 3
        if len(frame_packet.image_bytes) != expected:
            raise InvalidFramePacketFormatError(
                f"image_bytes size {len(frame_packet.image_bytes)} does not match "
                f"width*height*3 = {expected} (no stride/row padding allowed)"
            )

        return ValidationResult(ok=True, errors=())


# ---------------------------------------------------------------------------
# BaseImageBuilder
# ---------------------------------------------------------------------------


class BaseImageBuilder:
    """Converts canonical ``FramePacket.image_bytes`` into shared ``Image``.

    Converts bytes → np.ndarray (once, eagerly) and wraps as shared Image TypedDict.
    Re-validates the canonical-format constraints to keep this class safe to
    invoke independently of :class:`FramePacketValidator`.
    """

    def build(self, frame_packet: FramePacket) -> SharedImage:
        if frame_packet.pixel_format != "RGB":
            raise InvalidFramePacketFormatError("pixel_format must be 'RGB'")
        if frame_packet.layout != "HWC":
            raise InvalidFramePacketFormatError("layout must be 'HWC'")
        if frame_packet.dtype != "uint8":
            raise InvalidFramePacketFormatError("dtype must be 'uint8'")
        if frame_packet.value_range != "[0,255]":
            raise InvalidFramePacketFormatError("value_range must be '[0,255]'")
        if frame_packet.num_color_channels != 3:
            raise InvalidFramePacketFormatError("num_color_channels must be 3")
        if frame_packet.bits_per_channel != 8:
            raise InvalidFramePacketFormatError("bits_per_channel must be 8")
        if frame_packet.packing != "tightly_packed":
            raise InvalidFramePacketFormatError("packing must be 'tightly_packed'")
        if frame_packet.width <= 0 or frame_packet.height <= 0:
            raise InvalidFramePacketFormatError("width/height must be > 0")
        if len(frame_packet.image_bytes) != frame_packet.width * frame_packet.height * 3:
            raise InvalidFramePacketFormatError(
                "image_bytes size does not match width*height*3"
            )

        # Convert bytes → np.ndarray (once, eagerly during ingest)
        arr = np.frombuffer(frame_packet.image_bytes, dtype=np.uint8).reshape(
            frame_packet.height, frame_packet.width, 3
        ).copy()

        return {
            "data": arr,
            "width": frame_packet.width,
            "height": frame_packet.height,
            "color_format": "RGB",
            "layout": "HWC",
            "dtype": "uint8",
            "value_range": "[0,255]",
        }


# ---------------------------------------------------------------------------
# FrameStore
# ---------------------------------------------------------------------------


class FrameStore:
    """Per-camera CURRENT/PREVIOUS ``StoredFrame`` store.

    - ``put_latest`` rotates: old CURRENT becomes PREVIOUS, new frame becomes CURRENT.
    - ``get`` retrieves by ``camera_id`` and ``FrameTemporalSelector``.
    - Thread-safe for concurrent ``put_latest``/``get`` calls.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._frames_by_camera: dict[str, CameraFrameState] = {}

    def put_latest(self, stored_frame: StoredFrame) -> None:
        with self._lock:
            state = self._frames_by_camera.get(stored_frame.camera_id)
            if state is None:
                state = CameraFrameState()
                self._frames_by_camera[stored_frame.camera_id] = state
            state.previous = state.current
            state.current = stored_frame

    def get(self, camera_id: str, temporal_selector: FrameTemporalSelector) -> StoredFrame:
        with self._lock:
            state = self._frames_by_camera.get(camera_id)
            if state is None or state.current is None:
                raise FrameNotFoundError(
                    f"No frame stored for camera_id={camera_id!r}"
                )
            if temporal_selector is FrameTemporalSelector.CURRENT:
                return state.current
            if temporal_selector is FrameTemporalSelector.PREVIOUS:
                if state.previous is None:
                    raise PreviousFrameNotAvailableError(
                        f"No previous frame available for camera_id={camera_id!r}"
                    )
                return state.previous
            raise ValidationError(f"Unknown FrameTemporalSelector: {temporal_selector!r}")


# ---------------------------------------------------------------------------
# CropProcessor
# ---------------------------------------------------------------------------


class CropProcessor:
    """Validates ``region_bbox`` and extracts pixel data as ndarray from a
    shared ``Image``. The stored image is never modified.
    """

    def crop(
        self, image: SharedImage, region_bbox: SharedBoundingBox
    ) -> tuple[np.ndarray, SharedBoundingBox]:
        x = self._bbox_attr(region_bbox, "x")
        y = self._bbox_attr(region_bbox, "y")
        width = self._bbox_attr(region_bbox, "width")
        height = self._bbox_attr(region_bbox, "height")

        if width <= 0 or height <= 0:
            raise InvalidCropBboxError(
                f"region_bbox must have positive width/height "
                f"(got width={width}, height={height})"
            )
        if x < 0 or y < 0:
            raise CropOutOfBoundsError(
                f"region_bbox origin must be non-negative "
                f"(got x={x}, y={y})"
            )
        if (
            x + width > image["width"]
            or y + height > image["height"]
        ):
            raise CropOutOfBoundsError(
                f"region_bbox extends outside frame "
                f"({x},{y},{width},{height}) "
                f"vs frame {image['width']}x{image['height']}"
            )

        # Real pixel slice: extract crop region from ndarray.
        data = image["data"]
        crop_arr = data[
            y : y + height,
            x : x + width,
            :,
        ].copy()

        source_bbox_full_frame: SharedBoundingBox = {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        }
        return crop_arr, source_bbox_full_frame

    @staticmethod
    def _bbox_attr(region_bbox: object, key: str) -> int:
        if isinstance(region_bbox, dict):
            value = region_bbox.get(key)
        else:
            value = getattr(region_bbox, key, None)
        if not isinstance(value, int):
            raise ValidationError(f"region_bbox.{key} must be an integer")
        return value


# ---------------------------------------------------------------------------
# OutputImageContractResolver
# ---------------------------------------------------------------------------


class OutputImageContractResolver:
    """Resolves ``OutputImageType`` to its hardcoded ``ImageConversionContract``."""

    def resolve(self, output_type: OutputImageType | Enum) -> ImageConversionContract:
        normalized_output_type: OutputImageType
        if isinstance(output_type, OutputImageType):
            normalized_output_type = output_type
        elif isinstance(output_type, Enum):
            try:
                normalized_output_type = OutputImageType(output_type.value)
            except ValueError as exc:
                raise UnsupportedOutputImageTypeError(
                    f"Unsupported OutputImageType: {output_type!r}"
                ) from exc
        else:
            raise UnsupportedOutputImageTypeError(
                f"Unsupported OutputImageType: {output_type!r}"
            )

        contract = _OUTPUT_IMAGE_TYPE_CONTRACTS.get(normalized_output_type)
        if contract is None:
            raise UnsupportedOutputImageTypeError(
                f"Unsupported OutputImageType: {output_type!r}"
            )
        return contract


# ---------------------------------------------------------------------------
# FrameConverter
# ---------------------------------------------------------------------------


class FrameConverter:
    """Applies spatial transformation (``GeometrySpec``) then pixel-format
    conversion (``ImageConversionContract``) to produce a concrete
    shared ``Image`` TypedDict with real pixel data.

        Geometry policies:
        - ``NONE``: no resize or padding; ``SpatialTransform`` is identity.
    - ``LETTERBOX``: resize preserving aspect ratio then pad to target size
      with ``padding_color``; ``SpatialTransform`` reflects scale and offsets.

    Pixel-format contracts:
    - ``GRAY / uint8 / [0,255]``: single-channel luminance via Pillow ``L`` mode.
    - ``RGB / uint8 / [0,255]``: raw RGB bytes unchanged.
    """

    def convert(
        self,
        image_data: np.ndarray,
        image_width: int,
        image_height: int,
        contract: ImageConversionContract,
        geometry_spec: object,
    ) -> tuple[SharedImage, SpatialTransform]:
        try:
            from PIL import Image as _PILImage  # Pillow required for real conversion

            # Ensure we always have a concrete RGB uint8 HWC ndarray before geometry.
            if image_data.size == 0:
                rgb_arr = np.zeros((image_height, image_width, 3), dtype=np.uint8)
            else:
                rgb_arr = image_data

            if rgb_arr.ndim != 3 or rgb_arr.shape[2] != 3:
                raise ConversionError(
                    f"image_data must be HWC RGB with 3 channels (got shape={rgb_arr.shape})"
                )
            if rgb_arr.shape[0] != image_height or rgb_arr.shape[1] != image_width:
                raise ConversionError(
                    "image_data shape does not match image_width/image_height metadata"
                )
            if rgb_arr.dtype != np.uint8:
                rgb_arr = rgb_arr.astype(np.uint8, copy=False)

            pil_img = _PILImage.fromarray(rgb_arr, mode="RGB")

            resize_policy = FrameTransformationLayer._geometry_attr(
                geometry_spec, "resize_policy"
            )
            tw = FrameTransformationLayer._geometry_attr(geometry_spec, "width")
            th = FrameTransformationLayer._geometry_attr(geometry_spec, "height")

            if isinstance(resize_policy, ResizePolicy):
                normalized_resize_policy = resize_policy
            elif isinstance(resize_policy, Enum):
                try:
                    normalized_resize_policy = ResizePolicy(resize_policy.value)
                except ValueError as exc:
                    raise ConversionError(
                        f"Unknown ResizePolicy: {resize_policy!r}"
                    ) from exc
            else:
                raise ConversionError(
                    f"Unknown ResizePolicy: {resize_policy!r}"
                )

            # ---- geometry -----------------------------------------------
            if normalized_resize_policy is ResizePolicy.NONE:
                spatial = SpatialTransform(
                    scale_x=1.0,
                    scale_y=1.0,
                    pad_left=0,
                    pad_top=0,
                    output_width=image_width,
                    output_height=image_height,
                )
                geom_img = pil_img


            elif normalized_resize_policy is ResizePolicy.LETTERBOX:
                if tw is None or th is None:
                    raise ConversionError(
                        "LETTERBOX requires width and height"
                    )
                cw = image_width
                ch = image_height
                # MD formulas: deterministic, platform-independent.
                scale = min(tw / cw, th / ch)
                resized_w = round(cw * scale)
                resized_h = round(ch * scale)
                pad_left = math.floor((tw - resized_w) / 2)
                pad_top = math.floor((th - resized_h) / 2)
                spatial = SpatialTransform(
                    scale_x=float(scale),
                    scale_y=float(scale),
                    pad_left=int(pad_left),
                    pad_top=int(pad_top),
                    output_width=int(tw),
                    output_height=int(th),
                )
                resampling = getattr(_PILImage, "Resampling", None)
                lanczos = resampling.LANCZOS if resampling is not None else 1
                small = pil_img.resize((resized_w, resized_h), lanczos)
                pr, pg, pb = self._padding_color_rgb(geometry_spec)
                canvas = _PILImage.new("RGB", (tw, th), (pr, pg, pb))
                canvas.paste(small, (pad_left, pad_top))
                geom_img = canvas

            else:  # pragma: no cover - enum exhaustive
                raise ConversionError(
                    f"Unknown ResizePolicy: {resize_policy!r}"
                )

            # ---- pixel-format conversion --------------------------------
            out_w, out_h = geom_img.size

            if contract.color_format == "GRAY":
                gray = geom_img.convert("L")
                data = np.asarray(gray, dtype=np.uint8)
                out_w, out_h = gray.size

            else:  # RGB uint8
                data = np.asarray(geom_img, dtype=np.uint8)

            converted: SharedImage = {
                "data": data,
                "width": out_w,
                "height": out_h,
                "color_format": contract.color_format,
                "layout": contract.layout,
                "dtype": contract.dtype,
                "value_range": contract.value_range,
            }
            return converted, spatial

        except ConversionError:
            raise
        except Exception as exc:
            raise ConversionError(f"FrameConverter.convert failed: {exc}") from exc

    @staticmethod
    def _padding_color_rgb(geometry_spec: object) -> tuple[int, int, int]:
        padding_color = FrameTransformationLayer._geometry_attr(
            geometry_spec, "padding_color"
        )
        if padding_color is None:
            return (0, 0, 0)

        if isinstance(padding_color, dict):
            r = padding_color.get("r")
            g = padding_color.get("g")
            b = padding_color.get("b")
        else:
            r = getattr(padding_color, "r", None)
            g = getattr(padding_color, "g", None)
            b = getattr(padding_color, "b", None)

        if not all(isinstance(ch, int) and 0 <= ch <= 255 for ch in (r, g, b)):
            raise ConversionError("geometry_spec.padding_color must be RGB integers in [0,255]")
        return (r, g, b)


# ---------------------------------------------------------------------------
# FrameTransformationLayer (facade)
# ---------------------------------------------------------------------------


class FrameTransformationLayer:
    """Primary entry point and orchestration facade for the FTL."""

    def __init__(
        self,
        validator: Optional[FramePacketValidator] = None,
        builder: Optional[BaseImageBuilder] = None,
        store: Optional[FrameStore] = None,
        crop_processor: Optional[CropProcessor] = None,
        contract_resolver: Optional[OutputImageContractResolver] = None,
        converter: Optional[FrameConverter] = None,
    ) -> None:
        self._validator = validator or FramePacketValidator()
        self._builder = builder or BaseImageBuilder()
        self._store = store or FrameStore()
        self._crop = crop_processor or CropProcessor()
        self._resolver = contract_resolver or OutputImageContractResolver()
        self._converter = converter or FrameConverter()

    # ---- ingest --------------------------------------------------------
    def ingest_frame(self, frame_packet: FramePacket) -> None:
        # Validate (raises ValidationError or InvalidFramePacketFormatError)
        self._validator.validate(frame_packet)
        # Build shared Image (converts bytes → ndarray, raises InvalidFramePacketFormatError on failure)
        image = self._builder.build(frame_packet)
        # Wrap as StoredFrame and rotate CURRENT/PREVIOUS in FrameStore
        stored_frame = StoredFrame(
            frame_id=frame_packet.frame_id,
            camera_id=frame_packet.camera_id,
            timestamp_ms=frame_packet.timestamp_ms,
            image=image,
        )
        self._store.put_latest(stored_frame)

    # ---- get_frame -----------------------------------------------------
    def get_frame(
        self,
        camera_id: str,
        temporal_selector: FrameTemporalSelector,
        region_bbox: SharedBoundingBox,
        output_type: OutputImageType,
        geometry_spec: SharedGeometrySpec | object,
    ) -> ProcessedFrame:
        self._validate_geometry_spec(geometry_spec)

        stored_frame = self._store.get(camera_id, temporal_selector)
        image = stored_frame.image
        cropped_data, source_bbox_full_frame = self._crop.crop(image, region_bbox)
        contract = self._resolver.resolve(output_type)
        converted_image, spatial_transform = self._converter.convert(
            cropped_data,
            source_bbox_full_frame["width"],
            source_bbox_full_frame["height"],
            contract,
            geometry_spec,
        )

        return ProcessedFrame(
            frame_id=stored_frame.frame_id,
            timestamp_ms=stored_frame.timestamp_ms,
            image=converted_image,
            source_bbox_full_frame=source_bbox_full_frame,
            spatial_transform=spatial_transform,
        )

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _validate_geometry_spec(geometry_spec: object) -> None:
        resize_policy = FrameTransformationLayer._geometry_attr(
            geometry_spec, "resize_policy"
        )
        if isinstance(resize_policy, ResizePolicy):
            normalized_resize_policy = resize_policy
        elif isinstance(resize_policy, Enum):
            try:
                normalized_resize_policy = ResizePolicy(resize_policy.value)
            except ValueError as exc:
                raise ValidationError(
                    "geometry_spec.resize_policy must be a ResizePolicy enum value"
                ) from exc
        else:
            raise ValidationError(
                "geometry_spec.resize_policy must be a ResizePolicy enum value"
            )

        tw = FrameTransformationLayer._geometry_attr(geometry_spec, "width")
        th = FrameTransformationLayer._geometry_attr(geometry_spec, "height")

        if normalized_resize_policy is ResizePolicy.NONE:
            for axis_name, axis_value in (("width", tw), ("height", th)):
                if axis_value is not None and (
                    not isinstance(axis_value, int) or axis_value < 0
                ):
                    raise ValidationError(
                        f"NONE geometry_spec {axis_name} must be an integer >= 0 when provided"
                    )
        elif normalized_resize_policy is ResizePolicy.LETTERBOX:
            if tw is None or th is None:
                raise ValidationError(
                    "LETTERBOX geometry_spec requires width and height"
                )
            if not isinstance(tw, int) or not isinstance(th, int):
                raise ValidationError(
                    "LETTERBOX width/height must be integers"
                )
            if tw <= 0 or th <= 0:
                raise ValidationError(
                    "LETTERBOX width/height must be > 0 "
                    f"(got {tw}, {th})"
                )
        else:
            raise ValidationError(
                f"Unsupported resize_policy: {normalized_resize_policy!r}"
            )

    @staticmethod
    def _geometry_attr(geometry_spec: object, key: str) -> Any:
        if isinstance(geometry_spec, dict):
            return geometry_spec.get(key)
        return getattr(geometry_spec, key, None)
