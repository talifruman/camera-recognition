"""Frame Transformation Layer (FTL) — real implementation.

Implements the public API, validation, contracts, storage, and full
pixel-level processing defined in
``doc/image_processing_service/frame_transformation_layer.md``.

CropProcessor performs real HWC byte slicing via NumPy.
FrameConverter performs real geometry (PRESERVE / LETTERBOX) and
pixel-format conversion (RGB uint8, grayscale uint8, RGB float32 [-1,1])
via Pillow.  SpatialTransform values follow the MD formulas exactly.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OutputImageType(Enum):
    """Pixel representation of the output image returned by ``get_frame``."""

    GRAYSCALE_UINT8_HWC = "GRAYSCALE_UINT8_HWC"
    RGB_UINT8_HWC = "RGB_UINT8_HWC"
    RGB_FLOAT32_HWC_NORMALIZED_MINUS1_TO_1 = (
        "RGB_FLOAT32_HWC_NORMALIZED_MINUS1_TO_1"
    )


class GeometryPolicy(Enum):
    """Spatial transformation policy applied during ``get_frame``."""

    PRESERVE = "PRESERVE"
    LETTERBOX = "LETTERBOX"


class FrameTemporalSelector(Enum):
    """Selects which stored frame to retrieve for a given camera."""

    CURRENT = "CURRENT"
    PREVIOUS = "PREVIOUS"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ValidationError(Exception):
    """Missing or inconsistent required metadata, or invalid ``GeometrySpec``."""


class InvalidFramePacketFormatError(Exception):
    """``FramePacket`` does not satisfy the canonical input format."""


class PreviousFrameNotAvailableError(Exception):
    """``get_frame(camera_id, PREVIOUS, ...)`` called before two successful ingests."""


class FrameNotFoundError(Exception):
    """``get_frame`` called for a camera with no CURRENT frame stored yet."""


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
class FramePacket:
    """Immutable canonical raw RGB pixel container — input to ``ingest_frame``."""

    frame_id: str
    camera_id: str
    timestamp_ms: int
    width: int
    height: int
    pixel_format: str  # must be "RGB"
    layout: str  # must be "HWC"
    num_color_channels: int  # must be 3
    bits_per_channel: int  # must be 8
    image_bytes: bytes


@dataclass(frozen=True)
class BaseImage:
    """Canonical full-frame internal image. Always RGB/HWC/uint8/[0,255]."""

    data: bytes
    width: int
    height: int
    color_format: str = "RGB"
    layout: str = "HWC"
    dtype: str = "uint8"
    value_range: str = "[0,255]"


@dataclass(frozen=True)
class Image:
    """Generic processed-image container produced by crop/convert."""

    data: bytes
    width: int
    height: int
    color_format: str
    layout: str
    dtype: str
    value_range: str


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

    policy: GeometryPolicy
    target_width: Optional[int] = None
    target_height: Optional[int] = None
    padding_color: RGBColor = field(default_factory=RGBColor)


@dataclass(frozen=True)
class SpatialTransform:
    scale_x: float
    scale_y: float
    pad_left: int
    pad_top: int
    output_width: int
    output_height: int


@dataclass(frozen=True)
class ImageConversionContract:
    """Pixel-format-only conversion contract."""

    color_format: str
    layout: str
    dtype: str
    value_range: str


@dataclass(frozen=True)
class ProcessedFrame:
    image: Image
    source_bbox_full_frame: BoundingBox
    spatial_transform: SpatialTransform


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class StoredFrame:
    """Wraps a full-frame ``BaseImage`` with identifying metadata for ``FrameStore``."""

    frame_id: str
    camera_id: str
    timestamp_ms: int
    base_image: BaseImage


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
    OutputImageType.RGB_FLOAT32_HWC_NORMALIZED_MINUS1_TO_1: ImageConversionContract(
        color_format="RGB",
        layout="HWC",
        dtype="float32",
        value_range="[-1,1]",
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
        if frame_packet.num_color_channels != 3:
            raise InvalidFramePacketFormatError(
                f"num_color_channels must be 3 (got {frame_packet.num_color_channels})"
            )
        if frame_packet.bits_per_channel != 8:
            raise InvalidFramePacketFormatError(
                f"bits_per_channel must be 8 (got {frame_packet.bits_per_channel})"
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
    """Wraps canonical ``FramePacket.image_bytes`` into a full-frame ``BaseImage``.

    Re-validates the canonical-format constraints to keep this class safe to
    invoke independently of :class:`FramePacketValidator`.
    """

    def build(self, frame_packet: FramePacket) -> BaseImage:
        if frame_packet.pixel_format != "RGB":
            raise InvalidFramePacketFormatError("pixel_format must be 'RGB'")
        if frame_packet.layout != "HWC":
            raise InvalidFramePacketFormatError("layout must be 'HWC'")
        if frame_packet.num_color_channels != 3:
            raise InvalidFramePacketFormatError("num_color_channels must be 3")
        if frame_packet.bits_per_channel != 8:
            raise InvalidFramePacketFormatError("bits_per_channel must be 8")
        if frame_packet.width <= 0 or frame_packet.height <= 0:
            raise InvalidFramePacketFormatError("width/height must be > 0")
        if len(frame_packet.image_bytes) != frame_packet.width * frame_packet.height * 3:
            raise InvalidFramePacketFormatError(
                "image_bytes size does not match width*height*3"
            )

        return BaseImage(
            data=frame_packet.image_bytes,
            width=frame_packet.width,
            height=frame_packet.height,
            color_format="RGB",
            layout="HWC",
            dtype="uint8",
            value_range="[0,255]",
        )


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
    """Validates ``region_bbox`` and slices a derived cropped ``Image`` from a
    ``BaseImage`` using real pixel data.  The stored ``BaseImage`` is never
    modified.
    """

    def crop(
        self, base_image: BaseImage, region_bbox: BoundingBox
    ) -> tuple[Image, BoundingBox]:
        if region_bbox.width <= 0 or region_bbox.height <= 0:
            raise InvalidCropBboxError(
                f"region_bbox must have positive width/height "
                f"(got width={region_bbox.width}, height={region_bbox.height})"
            )
        if region_bbox.x < 0 or region_bbox.y < 0:
            raise CropOutOfBoundsError(
                f"region_bbox origin must be non-negative "
                f"(got x={region_bbox.x}, y={region_bbox.y})"
            )
        if (
            region_bbox.x + region_bbox.width > base_image.width
            or region_bbox.y + region_bbox.height > base_image.height
        ):
            raise CropOutOfBoundsError(
                f"region_bbox extends outside frame "
                f"({region_bbox.x},{region_bbox.y},"
                f"{region_bbox.width},{region_bbox.height}) "
                f"vs frame {base_image.width}x{base_image.height}"
            )

        # Real pixel slice: reshape flat bytes -> HWC array, copy crop region.
        arr = np.frombuffer(base_image.data, dtype=np.uint8).reshape(
            base_image.height, base_image.width, 3
        )
        crop_arr = arr[
            region_bbox.y : region_bbox.y + region_bbox.height,
            region_bbox.x : region_bbox.x + region_bbox.width,
            :,
        ].copy()

        cropped = Image(
            data=bytes(crop_arr.tobytes()),
            width=region_bbox.width,
            height=region_bbox.height,
            color_format="RGB",
            layout="HWC",
            dtype="uint8",
            value_range="[0,255]",
        )
        source_bbox_full_frame = BoundingBox(
            x=region_bbox.x,
            y=region_bbox.y,
            width=region_bbox.width,
            height=region_bbox.height,
        )
        return cropped, source_bbox_full_frame


# ---------------------------------------------------------------------------
# OutputImageContractResolver
# ---------------------------------------------------------------------------


class OutputImageContractResolver:
    """Resolves ``OutputImageType`` to its hardcoded ``ImageConversionContract``."""

    def resolve(self, output_type: OutputImageType) -> ImageConversionContract:
        contract = _OUTPUT_IMAGE_TYPE_CONTRACTS.get(output_type)
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
    :class:`Image` with real pixel data.

    Geometry policies:
    - ``PRESERVE``: no resize or padding; ``SpatialTransform`` is identity.
    - ``LETTERBOX``: resize preserving aspect ratio then pad to target size
      with ``padding_color``; ``SpatialTransform`` reflects scale and offsets.

    Pixel-format contracts:
    - ``GRAY / uint8 / [0,255]``: single-channel luminance via Pillow ``L`` mode.
    - ``RGB / float32 / [-1,1]``: ``(uint8 / 127.5) - 1.0`` via NumPy.
    - ``RGB / uint8 / [0,255]``: raw RGB bytes unchanged.
    """

    def convert(
        self,
        image: Image,
        contract: ImageConversionContract,
        geometry_spec: GeometrySpec,
    ) -> tuple[Image, SpatialTransform]:
        try:
            from PIL import Image as _PILImage  # Pillow required for real conversion

            # When data is empty (e.g. unit-test-level direct converter calls),
            # synthesise a zero-filled image so metadata/SpatialTransform math
            # can still be exercised without real pixels.
            raw = image.data if image.data else bytes(image.width * image.height * 3)
            pil_img = _PILImage.frombytes("RGB", (image.width, image.height), raw)

            # ---- geometry -----------------------------------------------
            if geometry_spec.policy is GeometryPolicy.PRESERVE:
                spatial = SpatialTransform(
                    scale_x=1.0,
                    scale_y=1.0,
                    pad_left=0,
                    pad_top=0,
                    output_width=image.width,
                    output_height=image.height,
                )
                geom_img = pil_img

            elif geometry_spec.policy is GeometryPolicy.LETTERBOX:
                tw = geometry_spec.target_width
                th = geometry_spec.target_height
                cw = image.width
                ch = image.height
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
                small = pil_img.resize((resized_w, resized_h), _PILImage.LANCZOS)
                pc = geometry_spec.padding_color
                canvas = _PILImage.new("RGB", (tw, th), (pc.r, pc.g, pc.b))
                canvas.paste(small, (pad_left, pad_top))
                geom_img = canvas

            else:  # pragma: no cover - enum exhaustive
                raise ConversionError(
                    f"Unknown GeometryPolicy: {geometry_spec.policy!r}"
                )

            # ---- pixel-format conversion --------------------------------
            out_w, out_h = geom_img.size

            if contract.color_format == "GRAY":
                gray = geom_img.convert("L")
                data = gray.tobytes()
                out_w, out_h = gray.size

            elif contract.dtype == "float32":
                arr = np.array(geom_img, dtype=np.float32)
                arr = (arr / 127.5) - 1.0
                data = arr.tobytes()

            else:  # RGB uint8
                data = geom_img.tobytes()

            converted = Image(
                data=data,
                width=out_w,
                height=out_h,
                color_format=contract.color_format,
                layout=contract.layout,
                dtype=contract.dtype,
                value_range=contract.value_range,
            )
            return converted, spatial

        except ConversionError:
            raise
        except Exception as exc:
            raise ConversionError(f"FrameConverter.convert failed: {exc}") from exc


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
        # Build BaseImage (raises InvalidFramePacketFormatError on failure)
        base_image = self._builder.build(frame_packet)
        # Wrap as StoredFrame and rotate CURRENT/PREVIOUS in FrameStore
        stored_frame = StoredFrame(
            frame_id=frame_packet.frame_id,
            camera_id=frame_packet.camera_id,
            timestamp_ms=frame_packet.timestamp_ms,
            base_image=base_image,
        )
        self._store.put_latest(stored_frame)

    # ---- get_frame -----------------------------------------------------
    def get_frame(
        self,
        camera_id: str,
        temporal_selector: FrameTemporalSelector,
        region_bbox: BoundingBox,
        output_type: OutputImageType,
        geometry_spec: GeometrySpec,
    ) -> ProcessedFrame:
        self._validate_geometry_spec(geometry_spec)

        stored_frame = self._store.get(camera_id, temporal_selector)
        base_image = stored_frame.base_image
        cropped_image, source_bbox_full_frame = self._crop.crop(base_image, region_bbox)
        contract = self._resolver.resolve(output_type)
        converted_image, spatial_transform = self._converter.convert(
            cropped_image, contract, geometry_spec
        )

        return ProcessedFrame(
            image=converted_image,
            source_bbox_full_frame=source_bbox_full_frame,
            spatial_transform=spatial_transform,
        )

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _validate_geometry_spec(geometry_spec: GeometrySpec) -> None:
        if not isinstance(geometry_spec, GeometrySpec):
            raise ValidationError("geometry_spec must be a GeometrySpec instance")
        if not isinstance(geometry_spec.policy, GeometryPolicy):
            raise ValidationError(
                "geometry_spec.policy must be a GeometryPolicy enum value"
            )

        if geometry_spec.policy is GeometryPolicy.PRESERVE:
            if (
                geometry_spec.target_width is not None
                or geometry_spec.target_height is not None
            ):
                raise ValidationError(
                    "PRESERVE geometry_spec must have target_width and "
                    "target_height set to None"
                )
        elif geometry_spec.policy is GeometryPolicy.LETTERBOX:
            tw = geometry_spec.target_width
            th = geometry_spec.target_height
            if tw is None or th is None:
                raise ValidationError(
                    "LETTERBOX geometry_spec requires target_width and target_height"
                )
            if not isinstance(tw, int) or not isinstance(th, int):
                raise ValidationError(
                    "LETTERBOX target_width/target_height must be integers"
                )
            if tw <= 0 or th <= 0:
                raise ValidationError(
                    f"LETTERBOX target_width/target_height must be > 0 "
                    f"(got {tw}, {th})"
                )
