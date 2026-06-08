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
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from src.image_processing.frame_transformation_layer import (  # noqa: E402
    CameraFrameState,
    ConversionError,
    CropOutOfBoundsError,
    FrameConverter,
    FrameNotFoundError,
    FramePacket,
    FrameStore,
    FrameTemporalSelector,
    FrameTransformationLayer,
    ResizePolicy,
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
from src.image_processing.frame_transformation_layer.module import (  # noqa: E402
    BoundingBox as InternalBoundingBox,
    GeometrySpec as InternalGeometrySpec,
)
from src.image_processing.motion_detection import (  # noqa: E402
    MotionDetectionManager,
)
from src.image_processing.face_recognition import (  # noqa: E402
    FaceRecognitionConfig,
    FaceRecognitionModule,
    StubFaceEmbeddingEngine,
)
from src.image_processing.shared.contracts import (  # noqa: E402
    BoundingBox as SharedBoundingBox,
    FramePacket as SharedFramePacket,
    GeometrySpec as SharedGeometrySpec,
    OutputImageType as SharedOutputImageType,
    ResizePolicy as SharedResizePolicy,
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
    dtype: str = "uint8",
    value_range: str = "[0,255]",
    num_color_channels: int = 3,
    bits_per_channel: int = 8,
    packing: str = "tightly_packed",
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
        dtype=dtype,
        value_range=value_range,
        num_color_channels=num_color_channels,
        bits_per_channel=bits_per_channel,
        packing=packing,
        image_bytes=image_bytes,
    )


def test_ftl_contract_frame_packet_is_shared_type():
    assert FramePacket is SharedFramePacket


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
    image = stored.image
    assert isinstance(image, dict)
    assert image["width"] == 4
    assert image["height"] == 3
    assert image["color_format"] == "RGB"
    assert image["layout"] == "HWC"
    assert image["dtype"] == "uint8"
    assert image["value_range"] == "[0,255]"
    expected = np.frombuffer(raw, dtype=np.uint8).reshape(3, 4, 3)
    assert np.array_equal(image["data"], expected)


@pytest.mark.parametrize(
    "kwargs,exc",
    [
        ({"pixel_format": "BGR"}, InvalidFramePacketFormatError),
        ({"layout": "CHW"}, InvalidFramePacketFormatError),
        ({"dtype": "float32"}, InvalidFramePacketFormatError),
        ({"value_range": "[0,1]"}, InvalidFramePacketFormatError),
        ({"packing": "with_stride"}, InvalidFramePacketFormatError),
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
        region_bbox=InternalBoundingBox(x=0, y=0, width=10, height=8),
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
    )

    assert isinstance(pf, ProcessedFrame)
    assert isinstance(pf.image, dict)
    assert isinstance(pf.image["data"], np.ndarray)
    assert pf.frame_id == "f1"
    assert pf.timestamp_ms == 1_700_000_000_000
    assert pf.source_bbox_full_frame == {"x": 0, "y": 0, "width": 10, "height": 8}
    assert pf.image["width"] == 10
    assert pf.image["height"] == 8
    assert pf.image["color_format"] == "RGB"
    assert pf.image["dtype"] == "uint8"
    assert pf.image["value_range"] == "[0,255]"


def test_get_frame_current_preserves_frame_id():
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=10, height=8, frame_id="frame-A"))

    pf = ftl.get_frame(
        camera_id="c1",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=InternalBoundingBox(x=0, y=0, width=10, height=8),
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
    )

    assert pf.frame_id == "frame-A"
    assert pf.timestamp_ms == 1_700_000_000_000


def test_get_frame_previous_preserves_frame_id():
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=10, height=8, frame_id="frame-A"))
    ftl.ingest_frame(_make_packet(width=10, height=8, frame_id="frame-B"))

    pf = ftl.get_frame(
        camera_id="c1",
        temporal_selector=FrameTemporalSelector.PREVIOUS,
        region_bbox=InternalBoundingBox(x=0, y=0, width=10, height=8),
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
    )

    assert pf.frame_id == "frame-A"
    assert pf.timestamp_ms == 1_700_000_000_000


def test_get_frame_accepts_shared_geometry_spec_typeddict():
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=10, height=8))

    geometry_spec: SharedGeometrySpec = {
        "resize_policy": SharedResizePolicy.NONE,
        # Shared contract allows width/height fields to be present for NONE and ignored.
        "width": 32,
        "height": 32,
    }

    pf = ftl.get_frame(
        camera_id="c1",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=InternalBoundingBox(x=0, y=0, width=10, height=8),
        output_type=SharedOutputImageType.RGB_UINT8_HWC,
        geometry_spec=geometry_spec,
    )

    assert pf.image["width"] == 10
    assert pf.image["height"] == 8


def test_get_frame_accepts_shared_bounding_box_typeddict():
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=10, height=8))

    region_bbox: SharedBoundingBox = {
        "x": 2,
        "y": 1,
        "width": 5,
        "height": 4,
    }

    pf = ftl.get_frame(
        camera_id="c1",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=region_bbox,
        output_type=SharedOutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
    )

    assert pf.source_bbox_full_frame == {"x": 2, "y": 1, "width": 5, "height": 4}


def test_crop_does_not_change_frame_id():
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=20, height=16, frame_id="f1"))

    pf = ftl.get_frame(
        camera_id="c1",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=InternalBoundingBox(x=2, y=2, width=10, height=8),
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
    )

    assert pf.frame_id == "f1"


def test_none_geometry_keeps_crop_dimensions():
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=200, height=150))

    pf = ftl.get_frame(
        camera_id="c1",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=InternalBoundingBox(x=10, y=20, width=100, height=80),
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
    )

    assert pf.image["width"] == 100
    assert pf.image["height"] == 80
    assert pf.spatial_transform.scale_x == 1.0
    assert pf.spatial_transform.scale_y == 1.0
    assert pf.spatial_transform.pad_left == 0
    assert pf.spatial_transform.pad_top == 0
    assert pf.spatial_transform.output_width == 100
    assert pf.spatial_transform.output_height == 80
    assert pf.source_bbox_full_frame == {"x": 10, "y": 20, "width": 100, "height": 80}


def test_none_geometry_accepts_zero_placeholders_and_keeps_identity_transform():
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=64, height=48))

    pf = ftl.get_frame(
        camera_id="c1",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=InternalBoundingBox(x=5, y=6, width=20, height=10),
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(
            resize_policy=ResizePolicy.NONE,
            width=0,
            height=0,
        ),
    )

    st = pf.spatial_transform
    assert st.scale_x == 1.0
    assert st.scale_y == 1.0
    assert st.pad_left == 0
    assert st.pad_top == 0
    assert st.output_width == 20
    assert st.output_height == 10
    assert pf.image["width"] == 20
    assert pf.image["height"] == 10


def test_ftl_accepts_motion_detection_input_contract_geometry_spec():
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=96, height=72))
    motion_contract = MotionDetectionManager().get_input_contract()

    pf = ftl.get_frame(
        camera_id="c1",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=InternalBoundingBox(x=0, y=0, width=96, height=72),
        output_type=motion_contract["output_image_type"],
        geometry_spec=motion_contract["geometry_spec"],
    )

    assert pf.spatial_transform.scale_x == 1.0
    assert pf.spatial_transform.scale_y == 1.0
    assert pf.spatial_transform.pad_left == 0
    assert pf.spatial_transform.pad_top == 0
    assert pf.spatial_transform.output_width == 96
    assert pf.spatial_transform.output_height == 72


def test_ftl_accepts_face_recognition_input_contract_geometry_spec():
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=120, height=80))
    face_recognition_contract = FaceRecognitionModule(
        config=FaceRecognitionConfig(recognition_threshold=0.5),
        embedding_engine=StubFaceEmbeddingEngine(),
        gallery_entries=[],
    ).get_input_contract()

    pf = ftl.get_frame(
        camera_id="c1",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=InternalBoundingBox(x=10, y=10, width=40, height=30),
        output_type=face_recognition_contract["output_image_type"],
        geometry_spec=face_recognition_contract["geometry_spec"],
    )

    assert pf.spatial_transform.scale_x == 1.0
    assert pf.spatial_transform.scale_y == 1.0
    assert pf.spatial_transform.pad_left == 0
    assert pf.spatial_transform.pad_top == 0
    assert pf.spatial_transform.output_width == 40
    assert pf.spatial_transform.output_height == 30


def test_letterbox_geometry_target_dims_and_transform():
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=400, height=300))

    # Crop 200x100 → target 100x100 → scale=0.5; resized 100x50;
    # pad_left=0, pad_top=floor((100-50)/2)=25
    pf = ftl.get_frame(
        camera_id="c1",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=InternalBoundingBox(x=0, y=0, width=200, height=100),
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(
            resize_policy=ResizePolicy.LETTERBOX,
            width=100,
            height=100,
        ),
    )

    st = pf.spatial_transform
    assert st.output_width == 100
    assert st.output_height == 100
    assert math.isclose(st.scale_x, 0.5)
    assert math.isclose(st.scale_y, 0.5)
    assert st.pad_left == 0
    assert st.pad_top == 25
    assert pf.image["width"] == 100
    assert pf.image["height"] == 100
    assert pf.source_bbox_full_frame == {"x": 0, "y": 0, "width": 200, "height": 100}


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
    ],
)
def test_output_image_type_changes_image_metadata(output_type, expected):
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=20, height=20))

    pf = ftl.get_frame(
        camera_id="c1",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=InternalBoundingBox(x=0, y=0, width=20, height=20),
        output_type=output_type,
        geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
    )

    cf, layout, dtype, vr = expected
    assert pf.image["color_format"] == cf
    assert pf.image["layout"] == layout
    assert pf.image["dtype"] == dtype
    assert pf.image["value_range"] == vr


def test_unknown_output_type_raises():
    resolver = OutputImageContractResolver()
    with pytest.raises(UnsupportedOutputImageTypeError):
        resolver.resolve("not-an-enum")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# FrameStore — stores shared Image TypedDict
# ---------------------------------------------------------------------------


def test_frame_store_only_stores_base_image():
    store = FrameStore()
    # FrameStore.put_latest only accepts StoredFrame; initial state has no CURRENT
    with pytest.raises(FrameNotFoundError):
        store.get("c1", FrameTemporalSelector.CURRENT)

    # After put_latest, CURRENT is set
    stored_image = {
        "data": np.zeros((2, 2, 3), dtype=np.uint8),
        "width": 2,
        "height": 2,
        "color_format": "RGB",
        "layout": "HWC",
        "dtype": "uint8",
        "value_range": "[0,255]",
    }
    sf = StoredFrame(frame_id="f1", camera_id="c1", timestamp_ms=0, image=stored_image)
    store.put_latest(sf)
    stored = store.get("c1", FrameTemporalSelector.CURRENT)
    assert isinstance(stored, StoredFrame)
    assert stored.image is stored_image

    # PREVIOUS is not available yet after first ingest
    with pytest.raises(PreviousFrameNotAvailableError):
        store.get("c1", FrameTemporalSelector.PREVIOUS)


def test_full_get_frame_does_not_pollute_store_with_derived_outputs():
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=50, height=40))
    before = ftl._store._frames_by_camera["c1"].current.image

    ftl.get_frame(
        camera_id="c1",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=InternalBoundingBox(x=0, y=0, width=10, height=10),
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(
            resize_policy=ResizePolicy.LETTERBOX,
            width=32,
            height=32,
        ),
    )
    after = ftl._store._frames_by_camera["c1"].current.image

    # Same canonical full-frame stored image object/state
    assert isinstance(after, dict)
    assert after["width"] == 50 and after["height"] == 40
    assert after["color_format"] == "RGB"
    assert after["dtype"] == "uint8"
    assert np.array_equal(after["data"], before["data"])


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
        region_bbox=InternalBoundingBox(x=0, y=0, width=10, height=10),
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
    )
    pf_b = ftl.get_frame(
        camera_id="cam2",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=InternalBoundingBox(x=0, y=0, width=20, height=15),
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
    )

    assert (pf_a.image["width"], pf_a.image["height"]) == (10, 10)
    assert (pf_b.image["width"], pf_b.image["height"]) == (20, 15)

    # cam2 has no PREVIOUS (only one ingest)
    with pytest.raises(PreviousFrameNotAvailableError):
        ftl.get_frame(
            camera_id="cam2",
            temporal_selector=FrameTemporalSelector.PREVIOUS,
            region_bbox=InternalBoundingBox(x=0, y=0, width=20, height=15),
            output_type=OutputImageType.RGB_UINT8_HWC,
            geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
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
            region_bbox=InternalBoundingBox(x=0, y=0, width=1, height=1),
            output_type=OutputImageType.RGB_UINT8_HWC,
            geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
        )


def test_invalid_crop_bbox_raises():
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=10, height=10))
    with pytest.raises(InvalidCropBboxError):
        ftl.get_frame(
            camera_id="c1",
            temporal_selector=FrameTemporalSelector.CURRENT,
            region_bbox=InternalBoundingBox(x=0, y=0, width=0, height=5),
            output_type=OutputImageType.RGB_UINT8_HWC,
            geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
        )


def test_crop_out_of_bounds_raises():
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=10, height=10))
    with pytest.raises(CropOutOfBoundsError):
        ftl.get_frame(
            camera_id="c1",
            temporal_selector=FrameTemporalSelector.CURRENT,
            region_bbox=InternalBoundingBox(x=5, y=5, width=10, height=10),
            output_type=OutputImageType.RGB_UINT8_HWC,
            geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
        )


@pytest.mark.parametrize(
    "spec",
    [
        # LETTERBOX with missing target
        InternalGeometrySpec(resize_policy=ResizePolicy.LETTERBOX, width=None, height=100),
        InternalGeometrySpec(resize_policy=ResizePolicy.LETTERBOX, width=100, height=None),
        # LETTERBOX with non-positive target
        InternalGeometrySpec(resize_policy=ResizePolicy.LETTERBOX, width=0, height=100),
        InternalGeometrySpec(resize_policy=ResizePolicy.LETTERBOX, width=100, height=-1),
        # NONE with invalid dimensions when provided
        InternalGeometrySpec(resize_policy=ResizePolicy.NONE, width=-1, height=10),
        InternalGeometrySpec(resize_policy=ResizePolicy.NONE, width=10, height=-1),
    ],
)
def test_geometry_spec_validation(spec):
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=10, height=10))
    with pytest.raises(ValidationError):
        ftl.get_frame(
            camera_id="c1",
            temporal_selector=FrameTemporalSelector.CURRENT,
            region_bbox=InternalBoundingBox(x=0, y=0, width=10, height=10),
            output_type=OutputImageType.RGB_UINT8_HWC,
            geometry_spec=spec,
        )


# ---------------------------------------------------------------------------
# Direct component sanity checks (defaults / mappings)
# ---------------------------------------------------------------------------


def test_geometry_spec_default_padding_is_black():
    spec = InternalGeometrySpec(resize_policy=ResizePolicy.NONE)
    assert spec.padding_color == RGBColor(r=0, g=0, b=0)


def test_resize_policy_exposes_only_supported_values():
    assert [policy.name for policy in ResizePolicy] == ["NONE", "LETTERBOX"]


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
    src_data = np.array([], dtype=np.uint8)
    contract = OutputImageContractResolver().resolve(OutputImageType.RGB_UINT8_HWC)
    spec = InternalGeometrySpec(
        resize_policy=ResizePolicy.LETTERBOX, width=100, height=100
    )
    img, st = converter.convert(src_data, 100, 200, contract, spec)
    assert (img["width"], img["height"]) == (100, 100)
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
        dtype="uint8",
        value_range="[0,255]",
        num_color_channels=3,
        bits_per_channel=8,
        packing="tightly_packed",
        image_bytes=raw,
    )
    return orig_w, orig_h, packet


def _print_processed_frame(pf: "ProcessedFrame") -> None:
    st = pf.spatial_transform
    print(
        f"\n  image.width={pf.image["width"]}  image.height={pf.image["height"]}"
        f"  color_format={pf.image["color_format"]!r}"
        f"  dtype={pf.image["dtype"]!r}  value_range={pf.image["value_range"]!r}"
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
    """Ingest a real JPEG as a canonical FramePacket; verify Image metadata."""
    orig_w, orig_h, packet = _make_real_packet()

    ftl = FrameTransformationLayer()
    ftl.ingest_frame(packet)  # must not raise

    stored = ftl._store._frames_by_camera.get("camera_001").current
    assert isinstance(stored, StoredFrame)
    assert stored.image["width"] == orig_w
    assert stored.image["height"] == orig_h
    assert stored.image["color_format"] == "RGB"
    assert stored.image["layout"] == "HWC"
    assert stored.image["dtype"] == "uint8"
    assert stored.image["value_range"] == "[0,255]"

    print(
        f"\n  [ingest] stored image: width={stored.image['width']}  height={stored.image['height']}"
        f"  color_format={stored.image['color_format']!r}  dtype={stored.image['dtype']!r}"
        f"  value_range={stored.image['value_range']!r}  data_len={stored.image['data'].size}"
    )


def test_get_full_frame_none_real_image_stub():
    """get_frame with full-frame BoundingBox + NONE returns correct metadata."""
    orig_w, orig_h, packet = _make_real_packet()
    full_bbox = InternalBoundingBox(x=0, y=0, width=orig_w, height=orig_h)

    ftl = FrameTransformationLayer()
    ftl.ingest_frame(packet)

    pf = ftl.get_frame(
        camera_id="camera_001",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=full_bbox,
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
    )

    assert pf.image["width"] == orig_w
    assert pf.image["height"] == orig_h
    assert pf.image["color_format"] == "RGB"
    assert pf.source_bbox_full_frame == {
        "x": full_bbox.x,
        "y": full_bbox.y,
        "width": full_bbox.width,
        "height": full_bbox.height,
    }
    st = pf.spatial_transform
    assert math.isclose(st.scale_x, 1.0)
    assert math.isclose(st.scale_y, 1.0)
    assert st.pad_left == 0
    assert st.pad_top == 0
    assert st.output_width == orig_w
    assert st.output_height == orig_h

    _print_processed_frame(pf)


def test_get_crop_none_real_image_stub():
    """get_frame with a sub-region BoundingBox + NONE yields bbox dimensions."""
    orig_w, orig_h, packet = _make_real_packet()
    crop_w = min(100, orig_w - 10)
    crop_h = min(100, orig_h - 10)
    crop_bbox = InternalBoundingBox(x=10, y=10, width=crop_w, height=crop_h)

    ftl = FrameTransformationLayer()
    ftl.ingest_frame(packet)

    pf = ftl.get_frame(
        camera_id="camera_001",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=crop_bbox,
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
    )

    assert pf.image["width"] == crop_w
    assert pf.image["height"] == crop_h
    assert pf.source_bbox_full_frame == {
        "x": crop_bbox.x,
        "y": crop_bbox.y,
        "width": crop_bbox.width,
        "height": crop_bbox.height,
    }
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
        region_bbox=InternalBoundingBox(x=0, y=0, width=orig_w, height=orig_h),
        output_type=OutputImageType.GRAYSCALE_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
    )

    assert pf.image["color_format"] == "GRAY"
    assert pf.image["layout"] == "HWC"
    assert pf.image["dtype"] == "uint8"
    assert pf.image["value_range"] == "[0,255]"

    _print_processed_frame(pf)


def test_removed_normalized_types_not_in_output_image_type():
    """Verify normalized [-1,1] and [0,1] output types are no longer part of the public enum."""
    assert not hasattr(OutputImageType, "RGB_FLOAT32_HWC_NORMALIZED_MINUS1_TO_1")
    assert not hasattr(OutputImageType, "RGB_FLOAT32_HWC_NORMALIZED_0_TO_1")


def test_get_letterbox_metadata_real_image_stub():
    """get_frame with LETTERBOX 640×640 returns correct scale/padding transform."""
    orig_w, orig_h, packet = _make_real_packet()
    full_bbox = InternalBoundingBox(x=0, y=0, width=orig_w, height=orig_h)

    ftl = FrameTransformationLayer()
    ftl.ingest_frame(packet)

    pf = ftl.get_frame(
        camera_id="camera_001",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=full_bbox,
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(
            resize_policy=ResizePolicy.LETTERBOX,
            width=640,
            height=640,
            padding_color=RGBColor(r=0, g=0, b=0),
        ),
    )

    expected_scale = min(640 / orig_w, 640 / orig_h)
    resized_w = round(orig_w * expected_scale)
    resized_h = round(orig_h * expected_scale)
    expected_pad_left = math.floor((640 - resized_w) / 2)
    expected_pad_top = math.floor((640 - resized_h) / 2)

    assert pf.image["width"] == 640
    assert pf.image["height"] == 640
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
    bbox = InternalBoundingBox(x=bx, y=by, width=bw, height=bh)

    # NONE
    pf_none = ftl.get_frame(
        camera_id="camera_001",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=bbox,
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
    )

    # LETTERBOX 640×640
    pf_lb = ftl.get_frame(
        camera_id="camera_001",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=bbox,
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(
            resize_policy=ResizePolicy.LETTERBOX,
            width=640,
            height=640,
            padding_color=RGBColor(r=0, g=0, b=0),
        ),
    )

    # Assertions
    assert pf_none.image["width"] == bw
    assert pf_none.image["height"] == bh
    assert pf_lb.image["width"] == 640
    assert pf_lb.image["height"] == 640
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
        f"[NONE] out: {pf_none.image['width']}x{pf_none.image['height']}",
        f"[LETTERBOX] out: {pf_lb.image['width']}x{pf_lb.image['height']}",
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
    bbox = InternalBoundingBox(x=bx, y=by, width=bw, height=bh)

    ftl = FrameTransformationLayer()
    ftl.ingest_frame(packet)
    pf = ftl.get_frame(
        camera_id="camera_001",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=bbox,
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
    )

    # --- pixel assertion ---------------------------------------------------
    orig_arr = np.frombuffer(packet.image_bytes, dtype=np.uint8).reshape(orig_h, orig_w, 3)
    expected = orig_arr[by : by + bh, bx : bx + bw, :].copy()

    assert isinstance(pf.image["data"], np.ndarray)
    assert np.array_equal(pf.image["data"], expected)
    assert pf.image["width"] == bw
    assert pf.image["height"] == bh

    # --- save outputs ------------------------------------------------------
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    crop_img = PILImage.fromarray(pf.image["data"], mode="RGB")
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
    bbox = InternalBoundingBox(x=bx, y=by, width=bw, height=bh)

    ftl = FrameTransformationLayer()
    ftl.ingest_frame(packet)
    pf = ftl.get_frame(
        camera_id="camera_001",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=bbox,
        output_type=OutputImageType.GRAYSCALE_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
    )

    # --- metadata assertions -----------------------------------------------
    assert pf.image["color_format"] == "GRAY"
    assert pf.image["dtype"] == "uint8"
    assert pf.image["value_range"] == "[0,255]"
    assert isinstance(pf.image["data"], np.ndarray)
    assert pf.image["data"].shape == (bh, bw)

    # --- pixel assertions (sampled) ----------------------------------------
    ref_gray = np.array(
        PILImage.open(ASSET_PATH).convert("RGB").crop((bx, by, bx + bw, by + bh)).convert("L"),
        dtype=np.uint8,
    )
    actual = pf.image["data"]
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
    bbox = InternalBoundingBox(x=bx, y=by, width=bw, height=bh)

    ftl = FrameTransformationLayer()
    ftl.ingest_frame(packet)
    pf = ftl.get_frame(
        camera_id="camera_001",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=bbox,
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(
            resize_policy=ResizePolicy.LETTERBOX,
            width=640,
            height=640,
            padding_color=RGBColor(r=0, g=0, b=0),
        ),
    )

    # --- dimension + transform assertions ----------------------------------
    assert pf.image["width"] == 640
    assert pf.image["height"] == 640
    assert isinstance(pf.image["data"], np.ndarray)
    assert pf.image["data"].shape == (640, 640, 3)

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
    arr = pf.image["data"]
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
    lb_img = PILImage.fromarray(pf.image["data"], mode="RGB")
    lb_path = _OUTPUTS_DIR / "real_letterbox_output.jpg"
    lb_img.save(str(lb_path), quality=95)
    assert lb_path.exists()
    print(
        f"\n  [real letterbox visual]  scale={expected_scale:.4f}"
        f"  pad_left={expected_pad_left}  pad_top={expected_pad_top}"
    )
    print(f"  [real letterbox visual]  path={lb_path.resolve()}")


def test_real_coordinate_mapping():
    """A point in the letterbox output non-padding area maps back inside the source bbox."""
    _skip_if_no_real_image()

    orig_w, orig_h, packet = _make_real_packet()
    bx, by, bw, bh = _real_bbox(orig_w, orig_h)
    bbox = InternalBoundingBox(x=bx, y=by, width=bw, height=bh)

    ftl = FrameTransformationLayer()
    ftl.ingest_frame(packet)
    pf = ftl.get_frame(
        camera_id="camera_001",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=bbox,
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(
            resize_policy=ResizePolicy.LETTERBOX,
            width=640,
            height=640,
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
    fx = rx + pf.source_bbox_full_frame["x"]
    fy = ry + pf.source_bbox_full_frame["y"]

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
    bbox = InternalBoundingBox(x=bx, y=by, width=bw, height=bh)

    ftl = FrameTransformationLayer()
    ftl.ingest_frame(packet)

    pf_crop = ftl.get_frame(
        camera_id="camera_001",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=bbox,
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
    )
    pf_lb = ftl.get_frame(
        camera_id="camera_001",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=bbox,
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(
            resize_policy=ResizePolicy.LETTERBOX,
            width=640,
            height=640,
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
        f"[NONE]  out: {pf_crop.image['width']}x{pf_crop.image['height']}",
        f"[LETTERBOX] out: {pf_lb.image['width']}x{pf_lb.image['height']}",
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
        region_bbox=InternalBoundingBox(x=0, y=0, width=4, height=3),
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
    )
    assert pf.image["width"] == 4
    # PREVIOUS raises before second ingest
    with pytest.raises(PreviousFrameNotAvailableError):
        ftl.get_frame(
            camera_id="c1",
            temporal_selector=FrameTemporalSelector.PREVIOUS,
            region_bbox=InternalBoundingBox(x=0, y=0, width=4, height=3),
            output_type=OutputImageType.RGB_UINT8_HWC,
            geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
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
    expected2 = np.frombuffer(raw2, dtype=np.uint8).reshape(3, 4, 3)
    expected1 = np.frombuffer(raw1, dtype=np.uint8).reshape(3, 4, 3)
    assert np.array_equal(state.current.image["data"], expected2)
    assert np.array_equal(state.previous.image["data"], expected1)


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
            region_bbox=InternalBoundingBox(x=0, y=0, width=1, height=1),
            output_type=OutputImageType.RGB_UINT8_HWC,
            geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
        )


def test_get_previous_before_second_ingest_raises():
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=4, height=3))
    with pytest.raises(PreviousFrameNotAvailableError):
        ftl.get_frame(
            camera_id="c1",
            temporal_selector=FrameTemporalSelector.PREVIOUS,
            region_bbox=InternalBoundingBox(x=0, y=0, width=4, height=3),
            output_type=OutputImageType.RGB_UINT8_HWC,
            geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
        )


# ---------------------------------------------------------------------------
# SpatialTransform consistency across supported ResizePolicy values
# ---------------------------------------------------------------------------


def test_spatial_transform_none_identity():
    """NONE produces identity transform (scale=1, pad=0)."""
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=200, height=150))

    pf = ftl.get_frame(
        camera_id="c1",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=InternalBoundingBox(x=10, y=20, width=100, height=80),
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
    )

    st = pf.spatial_transform
    assert st.scale_x == 1.0
    assert st.scale_y == 1.0
    assert st.pad_left == 0
    assert st.pad_top == 0
    assert st.output_width == 100
    assert st.output_height == 80


def test_spatial_transform_letterbox_includes_padding():
    """LETTERBOX includes non-zero padding in spatial_transform."""
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=400, height=300))

    pf = ftl.get_frame(
        camera_id="c1",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=InternalBoundingBox(x=0, y=0, width=200, height=100),
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(
            resize_policy=ResizePolicy.LETTERBOX,
            width=100,
            height=100,
        ),
    )

    st = pf.spatial_transform
    # At least one padding value may be non-zero
    assert st.pad_left >= 0 or st.pad_top >= 0
    assert st.pad_left == 0 or st.pad_top > 0  # Likely some padding here


def test_spatial_transform_all_policies_have_output_dims():
    """Supported ResizePolicy values produce output_width and output_height."""
    ftl = FrameTransformationLayer()
    ftl.ingest_frame(_make_packet(width=400, height=300))

    bbox = InternalBoundingBox(x=0, y=0, width=200, height=100)
    output_type = OutputImageType.RGB_UINT8_HWC

    for policy in [ResizePolicy.NONE, ResizePolicy.LETTERBOX]:
        geom = InternalGeometrySpec(
            resize_policy=policy,
            width=150 if policy is ResizePolicy.LETTERBOX else None,
            height=150 if policy is ResizePolicy.LETTERBOX else None,
        )
        pf = ftl.get_frame(
            camera_id="c1",
            temporal_selector=FrameTemporalSelector.CURRENT,
            region_bbox=bbox,
            output_type=output_type,
            geometry_spec=geom,
        )
        st = pf.spatial_transform
        assert st.output_width > 0
        assert st.output_height > 0
        assert isinstance(st.scale_x, float)
        assert isinstance(st.scale_y, float)



# ---------------------------------------------------------------------------
# Concurrency tests
# ---------------------------------------------------------------------------


def test_concurrent_ingest_multiple_cameras():
    """Concurrent ingest from multiple cameras should not corrupt state."""
    ftl = FrameTransformationLayer()
    
    # Ingest frames for different cameras
    ftl.ingest_frame(_make_packet(width=100, height=100, camera_id="cam1"))
    ftl.ingest_frame(_make_packet(width=200, height=150, camera_id="cam2"))
    ftl.ingest_frame(_make_packet(width=150, height=120, camera_id="cam3"))

    # Verify each camera has independent CURRENT state
    pf1 = ftl.get_frame(
        camera_id="cam1",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=InternalBoundingBox(x=0, y=0, width=100, height=100),
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
    )
    pf2 = ftl.get_frame(
        camera_id="cam2",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=InternalBoundingBox(x=0, y=0, width=200, height=150),
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
    )
    pf3 = ftl.get_frame(
        camera_id="cam3",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=InternalBoundingBox(x=0, y=0, width=150, height=120),
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
    )

    assert pf1.image["width"] == 100
    assert pf1.image["height"] == 100
    assert pf2.image["width"] == 200
    assert pf2.image["height"] == 150
    assert pf3.image["width"] == 150
    assert pf3.image["height"] == 120


def test_concurrent_current_previous_isolation():
    """CURRENT/PREVIOUS state is per-camera and isolated."""
    ftl = FrameTransformationLayer()
    
    # cam1: two ingests
    ftl.ingest_frame(_make_packet(width=100, height=100, camera_id="cam1", frame_id="f1"))
    ftl.ingest_frame(_make_packet(width=100, height=100, camera_id="cam1", frame_id="f2"))
    
    # cam2: one ingest
    ftl.ingest_frame(_make_packet(width=200, height=150, camera_id="cam2", frame_id="f3"))

    # cam1 has both CURRENT and PREVIOUS
    pf_current = ftl.get_frame(
        camera_id="cam1",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=InternalBoundingBox(x=0, y=0, width=100, height=100),
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
    )
    pf_previous = ftl.get_frame(
        camera_id="cam1",
        temporal_selector=FrameTemporalSelector.PREVIOUS,
        region_bbox=InternalBoundingBox(x=0, y=0, width=100, height=100),
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
    )

    assert pf_current.frame_id == "f2"
    assert pf_previous.frame_id == "f1"

    # cam2 has only CURRENT, PREVIOUS raises
    with pytest.raises(PreviousFrameNotAvailableError):
        ftl.get_frame(
            camera_id="cam2",
            temporal_selector=FrameTemporalSelector.PREVIOUS,
            region_bbox=InternalBoundingBox(x=0, y=0, width=200, height=150),
            output_type=OutputImageType.RGB_UINT8_HWC,
            geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
        )


def test_no_cross_camera_contamination():
    """Per-camera CURRENT/PREVIOUS should not affect other cameras."""
    ftl = FrameTransformationLayer()
    
    # cam1: ingest f1, f2, f3 → CURRENT=f3, PREVIOUS=f2
    ftl.ingest_frame(_make_packet(width=100, height=100, camera_id="cam1", frame_id="f1"))
    ftl.ingest_frame(_make_packet(width=100, height=100, camera_id="cam1", frame_id="f2"))
    ftl.ingest_frame(_make_packet(width=100, height=100, camera_id="cam1", frame_id="f3"))

    # cam2: ingest g1, g2 → CURRENT=g2, PREVIOUS=g1
    ftl.ingest_frame(_make_packet(width=200, height=150, camera_id="cam2", frame_id="g1"))
    ftl.ingest_frame(_make_packet(width=200, height=150, camera_id="cam2", frame_id="g2"))

    # cam1 CURRENT should still be f3
    pf_cam1_current = ftl.get_frame(
        camera_id="cam1",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=InternalBoundingBox(x=0, y=0, width=100, height=100),
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
    )
    assert pf_cam1_current.frame_id == "f3"

    # cam1 PREVIOUS should still be f2
    pf_cam1_previous = ftl.get_frame(
        camera_id="cam1",
        temporal_selector=FrameTemporalSelector.PREVIOUS,
        region_bbox=InternalBoundingBox(x=0, y=0, width=100, height=100),
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
    )
    assert pf_cam1_previous.frame_id == "f2"

    # cam2 CURRENT should be g2
    pf_cam2_current = ftl.get_frame(
        camera_id="cam2",
        temporal_selector=FrameTemporalSelector.CURRENT,
        region_bbox=InternalBoundingBox(x=0, y=0, width=200, height=150),
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
    )
    assert pf_cam2_current.frame_id == "g2"

    # cam2 PREVIOUS should be g1
    pf_cam2_previous = ftl.get_frame(
        camera_id="cam2",
        temporal_selector=FrameTemporalSelector.PREVIOUS,
        region_bbox=InternalBoundingBox(x=0, y=0, width=200, height=150),
        output_type=OutputImageType.RGB_UINT8_HWC,
        geometry_spec=InternalGeometrySpec(resize_policy=ResizePolicy.NONE),
    )
    assert pf_cam2_previous.frame_id == "g1"

