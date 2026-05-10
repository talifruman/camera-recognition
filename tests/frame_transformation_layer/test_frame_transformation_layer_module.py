"""Unit tests for the Frame Transformation Layer STUB."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on sys.path so `src.*` imports resolve.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.image_processing.frame_transformation_layer import (  # noqa: E402
    BaseImage,
    BoundingBox,
    CameraFrameState,
    ConversionError,
    CropOutOfBoundsError,
    FrameConverter,
    FrameNotFoundError,
    FramePacket,
    FrameStore,
    FrameTemporalSelector,
    FrameTransformationLayer,
    GeometryPolicy,
    GeometrySpec,
    Image,
    ImageConversionContract,
    InvalidCropBboxError,
    InvalidFramePacketFormatError,
    OutputImageContractResolver,
    OutputImageType,
    PreviousFrameNotAvailableError,
    ProcessedFrame,
    RGBColor,
    StoredFrame,
    UnsupportedOutputImageTypeError,
    ValidationError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_packet(
    width: int = 4,
    height: int = 3,
    frame_id: str = "f1",
    camera_id: str = "c1",
    *,
    pixel_format: str = "RGB",
    layout: str = "HWC",
    num_color_channels: int = 3,
    bits_per_channel: int = 8,
    image_bytes: bytes | None = None,
    timestamp_ms: int = 1_700_000_000_000,
) -> FramePacket:
    if image_bytes is None:
        size = max(width, 0) * max(height, 0) * 3
        image_bytes = bytes(size) if size > 0 else b"\x00"
    return FramePacket(
        frame_id=frame_id,
        camera_id=camera_id,
        timestamp_ms=timestamp_ms,
        width=width,
        height=height,
        pixel_format=pixel_format,
        layout=layout,
        num_color_channels=num_color_channels,
        bits_per_channel=bits_per_channel,
        image_bytes=image_bytes,
    )


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def test_ingest_stores_full_frame_base_image():
    ftl = FrameTransformationLayer()
    raw = bytes(range(256)) * ((4 * 3 * 3 // 256) + 1)
    raw = raw[: 4 * 3 * 3]
    packet = _make_packet(width=4, height=3, image_bytes=raw)
    ftl.ingest_frame(packet)

    state = ftl._store._frames_by_camera.get("c1")
    assert state is not None
    stored = state.current
    assert isinstance(stored, StoredFrame)
    assert stored.frame_id == "f1"
    assert stored.camera_id == "c1"
    base = stored.base_image
    assert isinstance(base, BaseImage)
    assert base.width == 4
    assert base.height == 3
    assert base.color_format == "RGB"
    assert base.layout == "HWC"
    assert base.dtype == "uint8"
    assert base.value_range == "[0,255]"
    assert base.data == raw


@pytest.mark.parametrize(
    "kwargs,exc",
    [
        ({"pixel_format": "BGR"}, InvalidFramePacketFormatError),
        ({"layout": "CHW"}, InvalidFramePacketFormatError),
        ({"num_color_channels": 4}, InvalidFramePacketFormatError),
        ({"bits_per_channel": 16}, InvalidFramePacketFormatError),
        # byte-size mismatch (4*3*3 = 36, supply 30 bytes)
        ({"image_bytes": bytes(30)}, InvalidFramePacketFormatError),
    ],
)
def test_invalid_frame_packet_format_rejected(kwargs, exc):
    ftl = FrameTransformationLayer()
    packet = _make_packet(width=4, height=3, **kwargs)
    with pytest.raises(exc):
        ftl.ingest_frame(packet)
    # Store unchanged — no CURRENT for camera c1
    with pytest.raises(FrameNotFoundError):
        ftl._store.get("c1", FrameTemporalSelector.CURRENT)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame_id": ""},
        {"camera_id": ""},
        {"width": 0},
        {"height": -1},
        {"image_bytes": b""},
    ],
)
def test_validation_error_on_missing_metadata(kwargs):
    ftl = FrameTransformationLayer()
    packet = _make_packet(**kwargs)
    with pytest.raises(ValidationError):
        ftl.ingest_frame(packet)


# ---------------------------------------------------------------------------
# get_frame — basic flow
# ---------------------------------------------------------------------------


def test_get_frame_returns_processed_frame():
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=10, height=8))

    pf = ftl.get_frame(
        camera_id="c1",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=BoundingBox(x=0, y=0, width=10, height=8),
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=GeometrySpec(policy=GeometryPolicy.PRESERVE),
    )

    assert isinstance(pf, ProcessedFrame)
    assert isinstance(pf.image, Image)
    assert pf.source_bbox_full_frame == BoundingBox(x=0, y=0, width=10, height=8)
    assert pf.image.width == 10
    assert pf.image.height == 8
    assert pf.image.color_format == "RGB"
    assert pf.image.dtype == "uint8"
    assert pf.image.value_range == "[0,255]"


def test_preserve_geometry_keeps_crop_dimensions():
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=200, height=150))

    pf = ftl.get_frame(
        camera_id="c1",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=BoundingBox(x=10, y=20, width=100, height=80),
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=GeometrySpec(policy=GeometryPolicy.PRESERVE),
    )

    assert pf.image.width == 100
    assert pf.image.height == 80
    assert pf.spatial_transform.scale_x == 1.0
    assert pf.spatial_transform.scale_y == 1.0
    assert pf.spatial_transform.pad_left == 0
    assert pf.spatial_transform.pad_top == 0
    assert pf.spatial_transform.output_width == 100
    assert pf.spatial_transform.output_height == 80
    assert pf.source_bbox_full_frame == BoundingBox(x=10, y=20, width=100, height=80)


def test_letterbox_geometry_target_dims_and_transform():
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=400, height=300))

    # Crop 200x100 → target 100x100 → scale=0.5; resized 100x50;
    # pad_left=0, pad_top=floor((100-50)/2)=25
    pf = ftl.get_frame(
        camera_id="c1",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=BoundingBox(x=0, y=0, width=200, height=100),
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=GeometrySpec(
            policy=GeometryPolicy.LETTERBOX,
            target_width=100,
            target_height=100,
        ),
    )

    st = pf.spatial_transform
    assert st.output_width == 100
    assert st.output_height == 100
    assert math.isclose(st.scale_x, 0.5)
    assert math.isclose(st.scale_y, 0.5)
    assert st.pad_left == 0
    assert st.pad_top == 25
    assert pf.image.width == 100
    assert pf.image.height == 100
    assert pf.source_bbox_full_frame == BoundingBox(x=0, y=0, width=200, height=100)


# ---------------------------------------------------------------------------
# OutputImageType -> ImageConversionContract mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "output_type,expected",
    [
        (
            OutputImageType.GRAYSCALE_UINT8_HWC,
            ("GRAY", "HWC", "uint8", "[0,255]"),
        ),
        (
            OutputImageType.RGB_UINT8_HWC,
            ("RGB", "HWC", "uint8", "[0,255]"),
        ),
        (
            OutputImageType.RGB_FLOAT32_HWC_NORMALIZED_MINUS1_TO_1,
            ("RGB", "HWC", "float32", "[-1,1]"),
        ),
    ],
)
def test_output_image_type_changes_image_metadata(output_type, expected):
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=20, height=20))

    pf = ftl.get_frame(
        camera_id="c1",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=BoundingBox(x=0, y=0, width=20, height=20),
        output_type=output_type,
        geometry_spec=GeometrySpec(policy=GeometryPolicy.PRESERVE),
    )

    cf, layout, dtype, vr = expected
    assert pf.image.color_format == cf
    assert pf.image.layout == layout
    assert pf.image.dtype == dtype
    assert pf.image.value_range == vr


def test_unknown_output_type_raises():
    resolver = OutputImageContractResolver()
    with pytest.raises(UnsupportedOutputImageTypeError):
        resolver.resolve("not-an-enum")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# FrameStore — only stores BaseImage
# ---------------------------------------------------------------------------


def test_frame_store_only_stores_base_image():
    store = FrameStore()
    # FrameStore.put_latest only accepts StoredFrame; initial state has no CURRENT
    with pytest.raises(FrameNotFoundError):
        store.get("c1", FrameTemporalSelector.CURRENT)

    # After put_latest, CURRENT is set
    base = BaseImage(data=bytes(4), width=2, height=2)
    sf = StoredFrame(frame_id="f1", camera_id="c1", timestamp_ms=0, base_image=base)
    store.put_latest(sf)
    stored = store.get("c1", FrameTemporalSelector.CURRENT)
    assert isinstance(stored, StoredFrame)
    assert stored.base_image is base

    # PREVIOUS is not available yet after first ingest
    with pytest.raises(PreviousFrameNotAvailableError):
        store.get("c1", FrameTemporalSelector.PREVIOUS)


def test_full_get_frame_does_not_pollute_store_with_derived_outputs():
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=50, height=40))
    before = ftl._store._frames_by_camera["c1"].current.base_image

    ftl.get_frame(
        camera_id="c1",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=BoundingBox(x=0, y=0, width=10, height=10),
        output_type=OutputImageType.RGB_FLOAT32_HWC_NORMALIZED_MINUS1_TO_1,
        geometry_spec=GeometrySpec(
            policy=GeometryPolicy.LETTERBOX,
            target_width=32,
            target_height=32,
        ),
    )
    after = ftl._store._frames_by_camera["c1"].current.base_image

    # Same canonical full-frame BaseImage object/state
    assert isinstance(after, BaseImage)
    assert after.width == 50 and after.height == 40
    assert after.color_format == "RGB"
    assert after.dtype == "uint8"
    assert after.data == before.data


# ---------------------------------------------------------------------------
# Multiple frames
# ---------------------------------------------------------------------------


def test_two_frame_references_independent():
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=10, height=10, frame_id="a", camera_id="cam1"))
    ftl.ingest_frame(_make_packet(width=20, height=15, frame_id="b", camera_id="cam2"))

    pf_a = ftl.get_frame(
        camera_id="cam1",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=BoundingBox(x=0, y=0, width=10, height=10),
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=GeometrySpec(policy=GeometryPolicy.PRESERVE),
    )
    pf_b = ftl.get_frame(
        camera_id="cam2",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=BoundingBox(x=0, y=0, width=20, height=15),
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=GeometrySpec(policy=GeometryPolicy.PRESERVE),
    )

    assert (pf_a.image.width, pf_a.image.height) == (10, 10)
    assert (pf_b.image.width, pf_b.image.height) == (20, 15)

    # cam2 has no PREVIOUS (only one ingest)
    with pytest.raises(PreviousFrameNotAvailableError):
        ftl.get_frame(
            camera_id="cam2",
            temporal_selector=FrameTemporalSelector.PREVIOUS,
            region_bbox=BoundingBox(x=0, y=0, width=20, height=15),
            output_type=OutputImageType.RGB_UINT8_HWC,
            geometry_spec=GeometrySpec(policy=GeometryPolicy.PRESERVE),
        )


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_frame_not_found_raises():
    ftl = FrameTransformationLayer()
    with pytest.raises(FrameNotFoundError):
        ftl.get_frame(
            camera_id="missing_camera",
            temporal_selector=FrameTemporalSelector.CURRENT,
            region_bbox=BoundingBox(x=0, y=0, width=1, height=1),
            output_type=OutputImageType.RGB_UINT8_HWC,
            geometry_spec=GeometrySpec(policy=GeometryPolicy.PRESERVE),
        )


def test_invalid_crop_bbox_raises():
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=10, height=10))
    with pytest.raises(InvalidCropBboxError):
        ftl.get_frame(
            camera_id="c1",
            temporal_selector=FrameTemporalSelector.CURRENT,
            region_bbox=BoundingBox(x=0, y=0, width=0, height=5),
            output_type=OutputImageType.RGB_UINT8_HWC,
            geometry_spec=GeometrySpec(policy=GeometryPolicy.PRESERVE),
        )


def test_crop_out_of_bounds_raises():
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=10, height=10))
    with pytest.raises(CropOutOfBoundsError):
        ftl.get_frame(
            camera_id="c1",
            temporal_selector=FrameTemporalSelector.CURRENT,
            region_bbox=BoundingBox(x=5, y=5, width=10, height=10),
            output_type=OutputImageType.RGB_UINT8_HWC,
            geometry_spec=GeometrySpec(policy=GeometryPolicy.PRESERVE),
        )


@pytest.mark.parametrize(
    "spec",
    [
        # LETTERBOX with missing target
        GeometrySpec(policy=GeometryPolicy.LETTERBOX, target_width=None, target_height=100),
        GeometrySpec(policy=GeometryPolicy.LETTERBOX, target_width=100, target_height=None),
        # LETTERBOX with non-positive target
        GeometrySpec(policy=GeometryPolicy.LETTERBOX, target_width=0, target_height=100),
        GeometrySpec(policy=GeometryPolicy.LETTERBOX, target_width=100, target_height=-1),
        # PRESERVE with target dims set (forbidden by spec)
        GeometrySpec(policy=GeometryPolicy.PRESERVE, target_width=10, target_height=10),
    ],
)
def test_geometry_spec_validation(spec):
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=10, height=10))
    with pytest.raises(ValidationError):
        ftl.get_frame(
            camera_id="c1",
            temporal_selector=FrameTemporalSelector.CURRENT,
            region_bbox=BoundingBox(x=0, y=0, width=10, height=10),
            output_type=OutputImageType.RGB_UINT8_HWC,
            geometry_spec=spec,
        )


# ---------------------------------------------------------------------------
# Direct component sanity checks (defaults / mappings)
# ---------------------------------------------------------------------------


def test_geometry_spec_default_padding_is_black():
    spec = GeometrySpec(policy=GeometryPolicy.PRESERVE)
    assert spec.padding_color == RGBColor(r=0, g=0, b=0)


def test_resolver_returns_hardcoded_contract_instance():
    resolver = OutputImageContractResolver()
    contract = resolver.resolve(OutputImageType.RGB_UINT8_HWC)
    assert isinstance(contract, ImageConversionContract)
    assert contract.color_format == "RGB"
    assert contract.layout == "HWC"
    assert contract.dtype == "uint8"
    assert contract.value_range == "[0,255]"


def test_converter_letterbox_pad_left_when_taller_target():
    # crop 100x200, target 100x100 → scale=min(1.0, 0.5)=0.5
    # resized 50x100, pad_left=floor((100-50)/2)=25, pad_top=0
    converter = FrameConverter()
    src = Image(
        data=b"",
        width=100,
        height=200,
        color_format="RGB",
        layout="HWC",
        dtype="uint8",
        value_range="[0,255]",
    )
    contract = OutputImageContractResolver().resolve(OutputImageType.RGB_UINT8_HWC)
    spec = GeometrySpec(
        policy=GeometryPolicy.LETTERBOX, target_width=100, target_height=100
    )
    img, st = converter.convert(src, contract, spec)
    assert (img.width, img.height) == (100, 100)
    assert math.isclose(st.scale_x, 0.5)
    assert math.isclose(st.scale_y, 0.5)
    assert st.pad_left == 25
    assert st.pad_top == 0
    assert (st.output_width, st.output_height) == (100, 100)


def test_conversion_error_is_exposed():
    # ConversionError must be importable and a subclass of Exception
    assert issubclass(ConversionError, Exception)


# ---------------------------------------------------------------------------
# Real-image STUB integration tests
# ---------------------------------------------------------------------------

try:
    from PIL import Image as PILImage  # noqa: E402
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

ASSET_PATH = Path(__file__).parent / "assets" / "sample.jpg"

print("Looking for sample image at:", ASSET_PATH.resolve())


def _load_real_image() -> tuple[int, int, bytes]:
    """Load sample.jpg → (width, height, raw_RGB_bytes) in HWC layout."""
    if not _PIL_AVAILABLE:
        pytest.skip("Pillow not installed — skipping real-image tests")
    if not ASSET_PATH.exists():
        pytest.skip(f"sample.jpg not found at expected path: {ASSET_PATH.resolve()}")
    img = PILImage.open(ASSET_PATH).convert("RGB")
    width, height = img.size
    image_bytes = img.tobytes()
    return width, height, image_bytes


def _make_real_packet() -> tuple[int, int, "FramePacket"]:
    """Return (orig_width, orig_height, FramePacket) built from sample.jpg."""
    orig_w, orig_h, raw = _load_real_image()
    packet = FramePacket(
        frame_id="frame_001",
        camera_id="camera_001",
        timestamp_ms=123456789,
        width=orig_w,
        height=orig_h,
        pixel_format="RGB",
        layout="HWC",
        num_color_channels=3,
        bits_per_channel=8,
        image_bytes=raw,
    )
    return orig_w, orig_h, packet


def _print_processed_frame(pf: "ProcessedFrame") -> None:
    st = pf.spatial_transform
    print(
        f"\n  image.width={pf.image.width}  image.height={pf.image.height}"
        f"  color_format={pf.image.color_format!r}"
        f"  dtype={pf.image.dtype!r}  value_range={pf.image.value_range!r}"
        f"\n  source_bbox_full_frame={pf.source_bbox_full_frame}"
        f"\n  spatial_transform: scale_x={st.scale_x}  scale_y={st.scale_y}"
        f"  pad_left={st.pad_left}  pad_top={st.pad_top}"
        f"  output_width={st.output_width}  output_height={st.output_height}"
    )


def _real_bbox(orig_w: int, orig_h: int) -> tuple[int, int, int, int]:
    """Return (bx, by, bw, bh) as the centre-quarter of the image."""
    return orig_w // 4, orig_h // 4, orig_w // 2, orig_h // 2


def _skip_if_no_real_image() -> None:
    """Skip the calling test if Pillow is unavailable or sample.jpg is missing."""
    if not _PIL_AVAILABLE:
        pytest.skip("Pillow not installed")
    if not ASSET_PATH.exists():
        pytest.skip(f"sample.jpg not found: {ASSET_PATH.resolve()}")


def test_ingest_real_image_stub():
    """Ingest a real JPEG as a canonical FramePacket; verify BaseImage metadata."""
    orig_w, orig_h, packet = _make_real_packet()

    ftl = FrameTransformationLayer()
    ftl.ingest_frame(packet)  # must not raise

    stored = ftl._store._frames_by_camera.get("camera_001").current
    assert isinstance(stored, StoredFrame)
    assert stored.base_image.width == orig_w
    assert stored.base_image.height == orig_h
    assert stored.base_image.color_format == "RGB"
    assert stored.base_image.layout == "HWC"
    assert stored.base_image.dtype == "uint8"
    assert stored.base_image.value_range == "[0,255]"

    print(
        f"\n  [ingest] stored BaseImage: width={stored.base_image.width}  height={stored.base_image.height}"
        f"  color_format={stored.base_image.color_format!r}  dtype={stored.base_image.dtype!r}"
        f"  value_range={stored.base_image.value_range!r}  data_len={len(stored.base_image.data)}"
    )


def test_get_full_frame_preserve_real_image_stub():
    """get_frame with full-frame BoundingBox + PRESERVE returns correct metadata."""
    orig_w, orig_h, packet = _make_real_packet()
    full_bbox = BoundingBox(x=0, y=0, width=orig_w, height=orig_h)

    ftl = FrameTransformationLayer()
    ftl.ingest_frame(packet)

    pf = ftl.get_frame(
        camera_id="camera_001",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=full_bbox,
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=GeometrySpec(policy=GeometryPolicy.PRESERVE),
    )

    assert pf.image.width == orig_w
    assert pf.image.height == orig_h
    assert pf.image.color_format == "RGB"
    assert pf.source_bbox_full_frame == full_bbox
    st = pf.spatial_transform
    assert math.isclose(st.scale_x, 1.0)
    assert math.isclose(st.scale_y, 1.0)
    assert st.pad_left == 0
    assert st.pad_top == 0
    assert st.output_width == orig_w
    assert st.output_height == orig_h

    _print_processed_frame(pf)


def test_get_crop_preserve_real_image_stub():
    """get_frame with a sub-region BoundingBox + PRESERVE yields bbox dimensions."""
    orig_w, orig_h, packet = _make_real_packet()
    crop_w = min(100, orig_w - 10)
    crop_h = min(100, orig_h - 10)
    crop_bbox = BoundingBox(x=10, y=10, width=crop_w, height=crop_h)

    ftl = FrameTransformationLayer()
    ftl.ingest_frame(packet)

    pf = ftl.get_frame(
        camera_id="camera_001",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=crop_bbox,
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=GeometrySpec(policy=GeometryPolicy.PRESERVE),
    )

    assert pf.image.width == crop_w
    assert pf.image.height == crop_h
    assert pf.source_bbox_full_frame == crop_bbox
    st = pf.spatial_transform
    assert st.output_width == crop_w
    assert st.output_height == crop_h

    _print_processed_frame(pf)


def test_get_grayscale_metadata_real_image_stub():
    """get_frame with GRAYSCALE_UINT8_HWC output type returns correct metadata."""
    orig_w, orig_h, packet = _make_real_packet()

    ftl = FrameTransformationLayer()
    ftl.ingest_frame(packet)

    pf = ftl.get_frame(
        camera_id="camera_001",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=BoundingBox(x=0, y=0, width=orig_w, height=orig_h),
        output_type=OutputImageType.GRAYSCALE_UINT8_HWC,
        geometry_spec=GeometrySpec(policy=GeometryPolicy.PRESERVE),
    )

    assert pf.image.color_format == "GRAY"
    assert pf.image.layout == "HWC"
    assert pf.image.dtype == "uint8"
    assert pf.image.value_range == "[0,255]"

    _print_processed_frame(pf)


def test_get_float_normalized_metadata_real_image_stub():
    """get_frame with RGB_FLOAT32_HWC_NORMALIZED_MINUS1_TO_1 returns correct metadata."""
    orig_w, orig_h, packet = _make_real_packet()

    ftl = FrameTransformationLayer()
    ftl.ingest_frame(packet)

    pf = ftl.get_frame(
        camera_id="camera_001",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=BoundingBox(x=0, y=0, width=orig_w, height=orig_h),
        output_type=OutputImageType.RGB_FLOAT32_HWC_NORMALIZED_MINUS1_TO_1,
        geometry_spec=GeometrySpec(policy=GeometryPolicy.PRESERVE),
    )

    assert pf.image.color_format == "RGB"
    assert pf.image.dtype == "float32"
    assert pf.image.value_range == "[-1,1]"

    _print_processed_frame(pf)


def test_get_letterbox_metadata_real_image_stub():
    """get_frame with LETTERBOX 640×640 returns correct scale/padding transform."""
    orig_w, orig_h, packet = _make_real_packet()
    full_bbox = BoundingBox(x=0, y=0, width=orig_w, height=orig_h)

    ftl = FrameTransformationLayer()
    ftl.ingest_frame(packet)

    pf = ftl.get_frame(
        camera_id="camera_001",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=full_bbox,
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=GeometrySpec(
            policy=GeometryPolicy.LETTERBOX,
            target_width=640,
            target_height=640,
            padding_color=RGBColor(r=0, g=0, b=0),
        ),
    )

    expected_scale = min(640 / orig_w, 640 / orig_h)
    resized_w = round(orig_w * expected_scale)
    resized_h = round(orig_h * expected_scale)
    expected_pad_left = math.floor((640 - resized_w) / 2)
    expected_pad_top = math.floor((640 - resized_h) / 2)

    assert pf.image.width == 640
    assert pf.image.height == 640
    assert math.isclose(pf.spatial_transform.scale_x, expected_scale)
    assert math.isclose(pf.spatial_transform.scale_y, expected_scale)
    assert pf.spatial_transform.pad_left == expected_pad_left
    assert pf.spatial_transform.pad_top == expected_pad_top
    assert pf.spatial_transform.output_width == 640
    assert pf.spatial_transform.output_height == 640

    _print_processed_frame(pf)


# ---------------------------------------------------------------------------
# Visual debug test
# ---------------------------------------------------------------------------

_OUTPUTS_DIR = Path(__file__).parent / "outputs"


def test_visual_debug_real_image_stub():
    """Draw bbox + STUB metadata onto the original image and save as a debug JPEG."""
    if not _PIL_AVAILABLE:
        pytest.skip("Pillow not installed — skipping visual debug test")
    if not ASSET_PATH.exists():
        pytest.skip(f"sample.jpg not found at expected path: {ASSET_PATH.resolve()}")

    from PIL import ImageDraw, ImageFont  # noqa: E402

    orig_w, orig_h, packet = _make_real_packet()

    ftl = FrameTransformationLayer()
    ftl.ingest_frame(packet)

    # Define a centre-region bbox
    bx = orig_w // 4
    by = orig_h // 4
    bw = orig_w // 2
    bh = orig_h // 2
    bbox = BoundingBox(x=bx, y=by, width=bw, height=bh)

    # PRESERVE
    pf_preserve = ftl.get_frame(
        camera_id="camera_001",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=bbox,
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=GeometrySpec(policy=GeometryPolicy.PRESERVE),
    )

    # LETTERBOX 640×640
    pf_lb = ftl.get_frame(
        camera_id="camera_001",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=bbox,
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=GeometrySpec(
            policy=GeometryPolicy.LETTERBOX,
            target_width=640,
            target_height=640,
            padding_color=RGBColor(r=0, g=0, b=0),
        ),
    )

    # Assertions
    assert pf_preserve.image.width == bw
    assert pf_preserve.image.height == bh
    assert pf_lb.image.width == 640
    assert pf_lb.image.height == 640
    expected_scale = min(640 / bw, 640 / bh)
    expected_pad_left = math.floor((640 - round(bw * expected_scale)) / 2)
    expected_pad_top = math.floor((640 - round(bh * expected_scale)) / 2)
    assert math.isclose(pf_lb.spatial_transform.scale_x, expected_scale)
    assert math.isclose(pf_lb.spatial_transform.scale_y, expected_scale)
    assert pf_lb.spatial_transform.pad_left == expected_pad_left
    assert pf_lb.spatial_transform.pad_top == expected_pad_top

    # Build debug image
    debug_img = PILImage.open(ASSET_PATH).convert("RGB")
    draw = ImageDraw.Draw(debug_img)

    # Draw bbox rectangle (red, 3 px)
    draw.rectangle(
        [bx, by, bx + bw - 1, by + bh - 1],
        outline=(255, 0, 0),
        width=3,
    )

    # Compose metadata text
    st_lb = pf_lb.spatial_transform
    lines = [
        f"frame_id: frame_001",
        f"bbox: x={bx} y={by} w={bw} h={bh}",
        f"[PRESERVE] out: {pf_preserve.image.width}x{pf_preserve.image.height}",
        f"[LETTERBOX] out: {pf_lb.image.width}x{pf_lb.image.height}",
        f"scale_x={st_lb.scale_x:.4f}  scale_y={st_lb.scale_y:.4f}",
        f"pad_left={st_lb.pad_left}  pad_top={st_lb.pad_top}",
        f"[REAL] pixel data is live — real crop/letterbox applied",
    ]

    # Draw text with a fallback font
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except (IOError, OSError):
        font = ImageFont.load_default()

    text_x, text_y = 10, 10
    line_gap = 22
    for line in lines:
        # Shadow for readability
        draw.text((text_x + 1, text_y + 1), line, fill=(0, 0, 0), font=font)
        draw.text((text_x, text_y), line, fill=(255, 255, 0), font=font)
        text_y += line_gap

    # Save
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUTPUTS_DIR / "ftl_real_visual_debug.jpg"
    debug_img.save(str(out_path), quality=95)

    assert out_path.exists()
    print(f"\n  [visual debug] saved to: {out_path.resolve()}")


# ---------------------------------------------------------------------------
# Real pixel + visual tests  (bbox = w//4, h//4, w//2, h//2 throughout)
# ---------------------------------------------------------------------------


def test_real_crop_visual():
    """Save real_crop_output.jpg and original_with_bbox.jpg; assert pixel bytes match."""
    _skip_if_no_real_image()
    from PIL import ImageDraw  # noqa: E402

    orig_w, orig_h, packet = _make_real_packet()
    bx, by, bw, bh = _real_bbox(orig_w, orig_h)
    bbox = BoundingBox(x=bx, y=by, width=bw, height=bh)

    ftl = FrameTransformationLayer()
    ftl.ingest_frame(packet)
    pf = ftl.get_frame(
        camera_id="camera_001",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=bbox,
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=GeometrySpec(policy=GeometryPolicy.PRESERVE),
    )

    # --- pixel assertion ---------------------------------------------------
    orig_arr = np.frombuffer(packet.image_bytes, dtype=np.uint8).reshape(orig_h, orig_w, 3)
    expected = orig_arr[by : by + bh, bx : bx + bw, :].copy().tobytes()

    assert pf.image.data == expected
    assert pf.image.width == bw
    assert pf.image.height == bh

    # --- save outputs ------------------------------------------------------
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    crop_img = PILImage.frombytes("RGB", (pf.image.width, pf.image.height), pf.image.data)
    crop_path = _OUTPUTS_DIR / "real_crop_output.jpg"
    crop_img.save(str(crop_path), quality=95)

    orig_img = PILImage.open(ASSET_PATH).convert("RGB")
    draw = ImageDraw.Draw(orig_img)
    draw.rectangle([bx, by, bx + bw - 1, by + bh - 1], outline=(255, 0, 0), width=3)
    bbox_path = _OUTPUTS_DIR / "original_with_bbox.jpg"
    orig_img.save(str(bbox_path), quality=95)

    assert crop_path.exists()
    assert bbox_path.exists()
    print(f"\n  [real crop visual]  crop={crop_path.resolve()}")
    print(f"  [real crop visual]  orig_bbox={bbox_path.resolve()}")


def test_real_grayscale_visual():
    """Save real_grayscale_output.jpg using the centre bbox; assert metadata + pixels."""
    _skip_if_no_real_image()

    orig_w, orig_h, packet = _make_real_packet()
    bx, by, bw, bh = _real_bbox(orig_w, orig_h)
    bbox = BoundingBox(x=bx, y=by, width=bw, height=bh)

    ftl = FrameTransformationLayer()
    ftl.ingest_frame(packet)
    pf = ftl.get_frame(
        camera_id="camera_001",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=bbox,
        output_type=OutputImageType.GRAYSCALE_UINT8_HWC,
        geometry_spec=GeometrySpec(policy=GeometryPolicy.PRESERVE),
    )

    # --- metadata assertions -----------------------------------------------
    assert pf.image.color_format == "GRAY"
    assert pf.image.dtype == "uint8"
    assert pf.image.value_range == "[0,255]"
    assert len(pf.image.data) == bw * bh

    # --- pixel assertions (sampled) ----------------------------------------
    ref_gray = np.array(
        PILImage.open(ASSET_PATH).convert("RGB").crop((bx, by, bx + bw, by + bh)).convert("L"),
        dtype=np.uint8,
    )
    actual = np.frombuffer(pf.image.data, dtype=np.uint8).reshape(bh, bw)
    assert actual[0, 0] == ref_gray[0, 0]
    assert actual[bh // 2, bw // 2] == ref_gray[bh // 2, bw // 2]

    # --- save output -------------------------------------------------------
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    gray_img = PILImage.fromarray(actual, mode="L")
    gray_path = _OUTPUTS_DIR / "real_grayscale_output.jpg"
    gray_img.save(str(gray_path), quality=95)
    assert gray_path.exists()
    print(f"\n  [real grayscale visual]  pixel[{bh//2},{bw//2}]={actual[bh//2, bw//2]}")
    print(f"  [real grayscale visual]  path={gray_path.resolve()}")


def test_real_letterbox_visual():
    """Save real_letterbox_output.jpg; assert 640x640, transform values, padding, content."""
    _skip_if_no_real_image()

    orig_w, orig_h, packet = _make_real_packet()
    bx, by, bw, bh = _real_bbox(orig_w, orig_h)
    bbox = BoundingBox(x=bx, y=by, width=bw, height=bh)

    ftl = FrameTransformationLayer()
    ftl.ingest_frame(packet)
    pf = ftl.get_frame(
        camera_id="camera_001",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=bbox,
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=GeometrySpec(
            policy=GeometryPolicy.LETTERBOX,
            target_width=640,
            target_height=640,
            padding_color=RGBColor(r=0, g=0, b=0),
        ),
    )

    # --- dimension + transform assertions ----------------------------------
    assert pf.image.width == 640
    assert pf.image.height == 640
    assert len(pf.image.data) == 640 * 640 * 3

    expected_scale = min(640 / bw, 640 / bh)
    resized_w = round(bw * expected_scale)
    resized_h = round(bh * expected_scale)
    expected_pad_left = math.floor((640 - resized_w) / 2)
    expected_pad_top = math.floor((640 - resized_h) / 2)

    assert math.isclose(pf.spatial_transform.scale_x, expected_scale)
    assert math.isclose(pf.spatial_transform.scale_y, expected_scale)
    assert pf.spatial_transform.pad_left == expected_pad_left
    assert pf.spatial_transform.pad_top == expected_pad_top

    # --- pixel assertions: padding is black, content is non-zero ----------
    arr = np.frombuffer(pf.image.data, dtype=np.uint8).reshape(640, 640, 3)
    if expected_pad_left > 0:
        assert arr[:, :expected_pad_left, :].sum() == 0, "left padding should be black"
    if expected_pad_top > 0:
        assert arr[:expected_pad_top, :, :].sum() == 0, "top padding should be black"
    content = arr[
        expected_pad_top : expected_pad_top + resized_h,
        expected_pad_left : expected_pad_left + resized_w,
        :,
    ]
    assert content.sum() > 0, "content region should contain non-zero pixels"

    # --- save output -------------------------------------------------------
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    lb_img = PILImage.frombytes("RGB", (640, 640), pf.image.data)
    lb_path = _OUTPUTS_DIR / "real_letterbox_output.jpg"
    lb_img.save(str(lb_path), quality=95)
    assert lb_path.exists()
    print(
        f"\n  [real letterbox visual]  scale={expected_scale:.4f}"
        f"  pad_left={expected_pad_left}  pad_top={expected_pad_top}"
    )
    print(f"  [real letterbox visual]  path={lb_path.resolve()}")


def test_real_float_normalized():
    """Float32 output using centre bbox must be in [-1,1] and match (uint8/127.5)-1.0."""
    _skip_if_no_real_image()

    orig_w, orig_h, packet = _make_real_packet()
    bx, by, bw, bh = _real_bbox(orig_w, orig_h)
    bbox = BoundingBox(x=bx, y=by, width=bw, height=bh)

    ftl = FrameTransformationLayer()
    ftl.ingest_frame(packet)
    pf = ftl.get_frame(
        camera_id="camera_001",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=bbox,
        output_type=OutputImageType.RGB_FLOAT32_HWC_NORMALIZED_MINUS1_TO_1,
        geometry_spec=GeometrySpec(policy=GeometryPolicy.PRESERVE),
    )

    assert pf.image.color_format == "RGB"
    assert pf.image.dtype == "float32"
    assert pf.image.value_range == "[-1,1]"
    assert len(pf.image.data) == bw * bh * 3 * 4  # 4 bytes per float32

    float_arr = np.frombuffer(pf.image.data, dtype=np.float32).reshape(bh, bw, 3)
    assert float(float_arr.min()) >= -1.0
    assert float(float_arr.max()) <= 1.0

    # Verify formula on sampled pixels
    ref_crop = np.array(
        PILImage.open(ASSET_PATH).convert("RGB").crop((bx, by, bx + bw, by + bh)),
        dtype=np.float32,
    )
    expected = (ref_crop / 127.5) - 1.0
    np.testing.assert_allclose(float_arr[0, 0], expected[0, 0], atol=1e-5)
    np.testing.assert_allclose(float_arr[bh // 2, bw // 2], expected[bh // 2, bw // 2], atol=1e-5)
    print(
        f"\n  [real float32]  min={float_arr.min():.4f}  max={float_arr.max():.4f}"
        f"  pixel[0,0]={float_arr[0, 0]}"
    )


def test_real_coordinate_mapping():
    """A point in the letterbox output non-padding area maps back inside the source bbox."""
    _skip_if_no_real_image()

    orig_w, orig_h, packet = _make_real_packet()
    bx, by, bw, bh = _real_bbox(orig_w, orig_h)
    bbox = BoundingBox(x=bx, y=by, width=bw, height=bh)

    ftl = FrameTransformationLayer()
    ftl.ingest_frame(packet)
    pf = ftl.get_frame(
        camera_id="camera_001",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=bbox,
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=GeometrySpec(
            policy=GeometryPolicy.LETTERBOX,
            target_width=640,
            target_height=640,
            padding_color=RGBColor(r=0, g=0, b=0),
        ),
    )

    st = pf.spatial_transform
    # Pick a point in the centre of the resized content region
    resized_w = round(bw * st.scale_x)
    resized_h = round(bh * st.scale_y)
    ox = st.pad_left + resized_w // 2
    oy = st.pad_top + resized_h // 2

    # Reverse-map output point → full-frame coordinates
    cx = ox - st.pad_left
    cy = oy - st.pad_top
    rx = cx / st.scale_x
    ry = cy / st.scale_y
    fx = rx + pf.source_bbox_full_frame.x
    fy = ry + pf.source_bbox_full_frame.y

    # The mapped point must lie within the original source bbox
    assert bx <= fx < bx + bw, f"fx={fx} outside bbox x=[{bx},{bx + bw})"
    assert by <= fy < by + bh, f"fy={fy} outside bbox y=[{by},{by + bh})"
    print(
        f"\n  [coordinate mapping]  output=({ox},{oy}) → full-frame=({fx:.1f},{fy:.1f})"
        f"  bbox=[{bx},{by},{bw},{bh}]"
    )


def test_real_visual_debug():
    """Create real_visual_debug.jpg: original with bbox + metadata overlay."""
    _skip_if_no_real_image()
    from PIL import ImageDraw, ImageFont  # noqa: E402

    orig_w, orig_h, packet = _make_real_packet()
    bx, by, bw, bh = _real_bbox(orig_w, orig_h)
    bbox = BoundingBox(x=bx, y=by, width=bw, height=bh)

    ftl = FrameTransformationLayer()
    ftl.ingest_frame(packet)

    pf_crop = ftl.get_frame(
        camera_id="camera_001",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=bbox,
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=GeometrySpec(policy=GeometryPolicy.PRESERVE),
    )
    pf_lb = ftl.get_frame(
        camera_id="camera_001",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=bbox,
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=GeometrySpec(
            policy=GeometryPolicy.LETTERBOX,
            target_width=640,
            target_height=640,
            padding_color=RGBColor(r=0, g=0, b=0),
        ),
    )

    # Build debug image from original with red bbox drawn
    debug_img = PILImage.open(ASSET_PATH).convert("RGB")
    draw = ImageDraw.Draw(debug_img)
    draw.rectangle([bx, by, bx + bw - 1, by + bh - 1], outline=(255, 0, 0), width=3)

    st = pf_lb.spatial_transform
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    crop_path = _OUTPUTS_DIR / "real_crop_output.jpg"
    gray_path = _OUTPUTS_DIR / "real_grayscale_output.jpg"
    lb_path = _OUTPUTS_DIR / "real_letterbox_output.jpg"

    lines = [
        f"frame_id: frame_001  |  bbox: x={bx} y={by} w={bw} h={bh}",
        f"[PRESERVE]  out: {pf_crop.image.width}x{pf_crop.image.height}",
        f"[LETTERBOX] out: {pf_lb.image.width}x{pf_lb.image.height}",
        f"scale_x={st.scale_x:.4f}  scale_y={st.scale_y:.4f}",
        f"pad_left={st.pad_left}  pad_top={st.pad_top}",
        f"crop  -> {crop_path.name}",
        f"gray  -> {gray_path.name}",
        f"lb    -> {lb_path.name}",
    ]

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except (IOError, OSError):
        font = ImageFont.load_default()

    tx, ty, gap = 10, 10, 22
    for line in lines:
        draw.text((tx + 1, ty + 1), line, fill=(0, 0, 0), font=font)
        draw.text((tx, ty), line, fill=(255, 255, 0), font=font)
        ty += gap

    debug_path = _OUTPUTS_DIR / "real_visual_debug.jpg"
    debug_img.save(str(debug_path), quality=95)
    assert debug_path.exists()

    print(f"\n  [real visual debug]  path={debug_path.resolve()}")
    print(f"  crop output:      {crop_path.resolve()}")
    print(f"  grayscale output: {gray_path.resolve()}")
    print(f"  letterbox output: {lb_path.resolve()}")


# ---------------------------------------------------------------------------
# Temporal model tests
# ---------------------------------------------------------------------------


def test_first_ingest_current_set_previous_raises():
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=4, height=3, frame_id="f1", camera_id="c1"))
    # CURRENT works
    pf = ftl.get_frame(
        camera_id="c1",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=BoundingBox(x=0, y=0, width=4, height=3),
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=GeometrySpec(policy=GeometryPolicy.PRESERVE),
    )
    assert pf.image.width == 4
    # PREVIOUS raises before second ingest
    with pytest.raises(PreviousFrameNotAvailableError):
        ftl.get_frame(
            camera_id="c1",
            temporal_selector=FrameTemporalSelector.PREVIOUS,
            region_bbox=BoundingBox(x=0, y=0, width=4, height=3),
            output_type=OutputImageType.RGB_UINT8_HWC,
            geometry_spec=GeometrySpec(policy=GeometryPolicy.PRESERVE),
        )


def test_second_ingest_current_and_previous():
    ftl = FrameTransformationLayer()
    raw1 = bytes([1] * 4 * 3 * 3)
    raw2 = bytes([2] * 4 * 3 * 3)
    ftl.ingest_frame(_make_packet(width=4, height=3, frame_id="f1", image_bytes=raw1))
    ftl.ingest_frame(_make_packet(width=4, height=3, frame_id="f2", image_bytes=raw2))

    state = ftl._store._frames_by_camera["c1"]
    assert state.current.frame_id == "f2"
    assert state.previous.frame_id == "f1"
    assert state.current.base_image.data == raw2
    assert state.previous.base_image.data == raw1


def test_third_ingest_drops_oldest():
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(frame_id="f1", camera_id="c1"))
    ftl.ingest_frame(_make_packet(frame_id="f2", camera_id="c1"))
    ftl.ingest_frame(_make_packet(frame_id="f3", camera_id="c1"))

    state = ftl._store._frames_by_camera["c1"]
    assert state.current.frame_id == "f3"
    assert state.previous.frame_id == "f2"
    # f1 is gone — only the last two are kept


def test_failed_ingest_does_not_rotate():
    ftl = FrameTransformationLayer()
    raw = bytes(4 * 3 * 3)
    ftl.ingest_frame(_make_packet(frame_id="f1", camera_id="c1", image_bytes=raw))
    ftl.ingest_frame(_make_packet(frame_id="f2", camera_id="c1", image_bytes=raw))

    # Try to ingest an invalid packet
    with pytest.raises((ValidationError, InvalidFramePacketFormatError)):
        ftl.ingest_frame(_make_packet(pixel_format="BGR"))

    # State unchanged — still f2/f1
    state = ftl._store._frames_by_camera["c1"]
    assert state.current.frame_id == "f2"
    assert state.previous.frame_id == "f1"


def test_get_current_before_ingest_raises_frame_not_found():
    ftl = FrameTransformationLayer()
    with pytest.raises(FrameNotFoundError):
        ftl.get_frame(
            camera_id="c1",
            temporal_selector=FrameTemporalSelector.CURRENT,
            region_bbox=BoundingBox(x=0, y=0, width=1, height=1),
            output_type=OutputImageType.RGB_UINT8_HWC,
            geometry_spec=GeometrySpec(policy=GeometryPolicy.PRESERVE),
        )


def test_get_previous_before_second_ingest_raises():
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=4, height=3))
    with pytest.raises(PreviousFrameNotAvailableError):
        ftl.get_frame(
            camera_id="c1",
            temporal_selector=FrameTemporalSelector.PREVIOUS,
            region_bbox=BoundingBox(x=0, y=0, width=4, height=3),
            output_type=OutputImageType.RGB_UINT8_HWC,
            geometry_spec=GeometrySpec(policy=GeometryPolicy.PRESERVE),
        )
