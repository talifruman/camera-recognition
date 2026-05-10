from __future__ import annotations

import sys
import time
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_processing.frame_ingestion_gateway import (  # type: ignore[import-not-found]
    CameraSlotExhaustedError,
    FrameDecodeError,
    FrameIngestionGateway,
    FrameIngestionGatewayConfig,
    FrameIngestionInputValidator,
    FrameNormalizationError,
    FramePacket,
    FramePacketBuilder,
    FramePayloadNormalizer,
    FrameStore,
    FrameValidator,
    GatewayLifecycleError,
    IngressFrameMessage,
    NormalizedFrameBuffer,
    PayloadSizeMismatchError,
    StubFrameIngressTransport,
    StructuralValidationError,
    UnsupportedSourceFormatError,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_ALL_FORMATS = ["RGB", "BGR", "GRAY8", "YUV420", "NV12", "YUY2", "JPEG", "MJPEG"]


def _make_jpeg_bytes(w: int, h: int, bgr: tuple[int, int, int] = (128, 128, 128)) -> bytes:
    """Encode a solid-colour BGR image as a valid JPEG byte string."""
    b, g, r = bgr
    img = np.full((h, w, 3), [b, g, r], dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", img)
    assert ok, "cv2.imencode failed in test helper"
    return encoded.tobytes()


def _make_config(**overrides) -> FrameIngestionGatewayConfig:
    defaults = dict(
        bind_address="0.0.0.0:50061",
        service_name="test-service",
        max_concurrent_streams=10,
        max_camera_slots=4,
        camera_order=[],
        supported_source_formats=list(_ALL_FORMATS),
    )
    defaults.update(overrides)
    return FrameIngestionGatewayConfig(**defaults)


def _make_rgb_message(
    frame_id: str = "frame-1",
    camera_id: str = "cam-a",
    timestamp_ms: int = 1000,
    width: int = 4,
    height: int = 4,
) -> IngressFrameMessage:
    return IngressFrameMessage(
        frame_id=frame_id,
        camera_id=camera_id,
        timestamp_ms=timestamp_ms,
        width=width,
        height=height,
        source_format="RGB",
        payload_bytes=bytes(width * height * 3),
    )


def _poll_frame(
    gateway: FrameIngestionGateway,
    timeout: float = 2.0,
) -> FramePacket | None:
    """Poll get_next_frame() until a frame arrives or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = gateway.get_next_frame()
        if frame is not None:
            return frame
        time.sleep(0.01)
    return None


def _start_gateway(transport: StubFrameIngressTransport, **config_overrides) -> FrameIngestionGateway:
    gw = FrameIngestionGateway(transport=transport)
    gw.configure(_make_config(**config_overrides))
    gw.start()
    return gw


# ---------------------------------------------------------------------------
# TestFrameIngestionInputValidator
# ---------------------------------------------------------------------------


class TestFrameIngestionInputValidator(unittest.TestCase):
    def setUp(self):
        self.validator = FrameIngestionInputValidator()

    def _valid_message(self) -> IngressFrameMessage:
        return _make_rgb_message()

    def test_valid_message_passes(self):
        # Should not raise
        self.validator.validate(self._valid_message())

    def test_missing_frame_id_raises(self):
        msg = self._valid_message()
        msg.frame_id = ""
        with self.assertRaises(StructuralValidationError):
            self.validator.validate(msg)

    def test_none_frame_id_raises(self):
        msg = self._valid_message()
        msg.frame_id = None  # type: ignore[assignment]
        with self.assertRaises(StructuralValidationError):
            self.validator.validate(msg)

    def test_missing_camera_id_raises(self):
        msg = self._valid_message()
        msg.camera_id = ""
        with self.assertRaises(StructuralValidationError):
            self.validator.validate(msg)

    def test_zero_width_raises(self):
        msg = self._valid_message()
        msg.width = 0
        with self.assertRaises(StructuralValidationError):
            self.validator.validate(msg)

    def test_negative_width_raises(self):
        msg = self._valid_message()
        msg.width = -1
        with self.assertRaises(StructuralValidationError):
            self.validator.validate(msg)

    def test_zero_height_raises(self):
        msg = self._valid_message()
        msg.height = 0
        with self.assertRaises(StructuralValidationError):
            self.validator.validate(msg)

    def test_empty_source_format_raises(self):
        msg = self._valid_message()
        msg.source_format = ""
        with self.assertRaises(StructuralValidationError):
            self.validator.validate(msg)

    def test_empty_payload_raises(self):
        msg = self._valid_message()
        msg.payload_bytes = b""
        with self.assertRaises(StructuralValidationError):
            self.validator.validate(msg)

    def test_none_payload_raises(self):
        msg = self._valid_message()
        msg.payload_bytes = None  # type: ignore[assignment]
        with self.assertRaises(StructuralValidationError):
            self.validator.validate(msg)


# ---------------------------------------------------------------------------
# TestFrameValidator
# ---------------------------------------------------------------------------


class TestFrameValidator(unittest.TestCase):
    def setUp(self):
        self.validator = FrameValidator(_ALL_FORMATS)

    def _msg(self, fmt: str, w: int = 4, h: int = 4, payload: bytes | None = None) -> IngressFrameMessage:
        if payload is None:
            if fmt == "RGB":
                payload = bytes(w * h * 3)
            elif fmt == "BGR":
                payload = bytes(w * h * 3)
            elif fmt == "GRAY8":
                payload = bytes(w * h)
            elif fmt in ("YUV420", "NV12"):
                payload = bytes(w * h * 3 // 2)
            elif fmt == "YUY2":
                payload = bytes(w * h * 2)
            else:
                # Encoded: arbitrary bytes
                payload = b"\xff\xd8\xff" + bytes(10)
        return IngressFrameMessage(
            frame_id="f1",
            camera_id="cam-a",
            timestamp_ms=1000,
            width=w,
            height=h,
            source_format=fmt,
            payload_bytes=payload,
        )

    def test_valid_rgb_passes(self):
        self.validator.validate(self._msg("RGB"))

    def test_valid_bgr_passes(self):
        self.validator.validate(self._msg("BGR"))

    def test_valid_gray8_passes(self):
        self.validator.validate(self._msg("GRAY8"))

    def test_valid_yuv420_passes(self):
        self.validator.validate(self._msg("YUV420"))

    def test_valid_nv12_passes(self):
        self.validator.validate(self._msg("NV12"))

    def test_valid_yuy2_passes(self):
        self.validator.validate(self._msg("YUY2"))

    def test_valid_jpeg_passes(self):
        self.validator.validate(self._msg("JPEG"))

    def test_valid_mjpeg_passes(self):
        self.validator.validate(self._msg("MJPEG"))

    def test_unsupported_format_raises(self):
        msg = self._msg("RGB")
        msg.source_format = "H264"
        with self.assertRaises(UnsupportedSourceFormatError):
            self.validator.validate(msg)

    def test_rgb_size_mismatch_raises(self):
        msg = self._msg("RGB", payload=bytes(10))  # wrong size
        with self.assertRaises(PayloadSizeMismatchError):
            self.validator.validate(msg)

    def test_bgr_size_mismatch_raises(self):
        msg = self._msg("BGR", w=4, h=4, payload=bytes(10))
        with self.assertRaises(PayloadSizeMismatchError):
            self.validator.validate(msg)

    def test_gray8_size_mismatch_raises(self):
        # Correct GRAY8 size is 4*4=16; pass 20 bytes
        msg = self._msg("GRAY8", w=4, h=4, payload=bytes(20))
        with self.assertRaises(PayloadSizeMismatchError):
            self.validator.validate(msg)

    def test_yuv420_odd_width_raises(self):
        # odd width=3 with even height=4 → must raise
        msg = self._msg("YUV420", w=3, h=4, payload=bytes(3 * 4 * 3 // 2))
        with self.assertRaises(PayloadSizeMismatchError):
            self.validator.validate(msg)

    def test_yuv420_odd_height_raises(self):
        msg = self._msg("YUV420", w=4, h=3, payload=bytes(4 * 3 * 3 // 2))
        with self.assertRaises(PayloadSizeMismatchError):
            self.validator.validate(msg)

    def test_nv12_odd_dimension_raises(self):
        msg = self._msg("NV12", w=3, h=3, payload=bytes(3 * 3 * 3 // 2))
        with self.assertRaises(PayloadSizeMismatchError):
            self.validator.validate(msg)

    def test_yuy2_size_mismatch_raises(self):
        # Correct YUY2 size is 4*4*2=32; pass 10 bytes
        msg = self._msg("YUY2", w=4, h=4, payload=bytes(10))
        with self.assertRaises(PayloadSizeMismatchError):
            self.validator.validate(msg)

    def test_encoded_format_skips_size_check(self):
        # JPEG: payload can be any non-empty bytes — no size formula applied
        small_payload = b"\xff\xd8\xff\xe0"  # 4 bytes JPEG header
        msg = self._msg("JPEG", w=640, h=480, payload=small_payload)
        # Should not raise PayloadSizeMismatchError
        self.validator.validate(msg)

    def test_unsupported_validator_rejects_when_not_in_list(self):
        restricted = FrameValidator(["RGB"])
        msg = self._msg("BGR")
        with self.assertRaises(UnsupportedSourceFormatError):
            restricted.validate(msg)


# ---------------------------------------------------------------------------
# TestFramePayloadNormalizer
# ---------------------------------------------------------------------------


class TestFramePayloadNormalizer(unittest.TestCase):
    def setUp(self):
        self.normalizer = FramePayloadNormalizer()

    def _msg(self, fmt: str, w: int = 4, h: int = 4, payload: bytes | None = None) -> IngressFrameMessage:
        if payload is None:
            payload = bytes(w * h * 3) if fmt == "RGB" else bytes(w * h * 3)
        return IngressFrameMessage(
            frame_id="f1",
            camera_id="cam-a",
            timestamp_ms=1000,
            width=w,
            height=h,
            source_format=fmt,
            payload_bytes=payload,
        )

    def test_rgb_preserves_payload_bytes(self):
        payload = bytes(range(48))  # 4*4*3 = 48
        msg = self._msg("RGB", payload=payload)
        buf = self.normalizer.normalize(msg)
        self.assertEqual(buf.image_bytes, payload)

    def test_rgb_output_contract(self):
        msg = self._msg("RGB")
        buf = self.normalizer.normalize(msg)
        self.assertEqual(buf.color_format, "RGB")
        self.assertEqual(buf.layout, "HWC")
        self.assertEqual(buf.dtype, "uint8")
        self.assertEqual(buf.value_range, "[0,255]")
        self.assertEqual(buf.num_color_channels, 3)
        self.assertEqual(buf.bits_per_channel, 8)
        self.assertEqual(buf.width, 4)
        self.assertEqual(buf.height, 4)

    def test_bgr_produces_rgb_of_correct_size(self):
        # BGR 4x4 needs exactly 4*4*3=48 bytes
        msg = self._msg("BGR", w=4, h=4)
        buf = self.normalizer.normalize(msg)
        self.assertEqual(len(buf.image_bytes), 4 * 4 * 3)
        self.assertEqual(buf.color_format, "RGB")

    def test_gray8_correct_payload_produces_rgb(self):
        # GRAY8 4x4 needs exactly 4*4=16 bytes (not 48)
        msg = self._msg("GRAY8", w=4, h=4, payload=bytes(4 * 4))
        buf = self.normalizer.normalize(msg)
        self.assertEqual(len(buf.image_bytes), 4 * 4 * 3)
        self.assertEqual(buf.color_format, "RGB")

    def test_jpeg_invalid_bytes_raises_decode_error(self):
        # Truncated / corrupt JPEG must raise FrameDecodeError
        msg = IngressFrameMessage(
            frame_id="f1",
            camera_id="cam-a",
            timestamp_ms=1000,
            width=8,
            height=6,
            source_format="JPEG",
            payload_bytes=b"\xff\xd8\xff" + bytes(50),
        )
        with self.assertRaises(FrameDecodeError):
            self.normalizer.normalize(msg)

    def test_output_always_has_canonical_fields(self):
        # Each format gets a correctly-sized payload so the real normalizer succeeds
        w, h = 4, 4
        format_payloads = {
            "BGR":   bytes(w * h * 3),
            "GRAY8": bytes(w * h),
            "YUV420": bytes(w * h * 3 // 2),
            "JPEG":  _make_jpeg_bytes(w, h),
            "MJPEG": _make_jpeg_bytes(w, h),
        }
        for fmt, payload in format_payloads.items():
            msg = IngressFrameMessage(
                frame_id="f1",
                camera_id="cam-a",
                timestamp_ms=1000,
                width=w,
                height=h,
                source_format=fmt,
                payload_bytes=payload,
            )
            buf = self.normalizer.normalize(msg)
            with self.subTest(fmt=fmt):
                self.assertEqual(buf.color_format, "RGB")
                self.assertEqual(buf.layout, "HWC")
                self.assertEqual(buf.dtype, "uint8")
                self.assertEqual(buf.num_color_channels, 3)
                self.assertEqual(buf.bits_per_channel, 8)
                self.assertEqual(buf.width, w)
                self.assertEqual(buf.height, h)
                self.assertEqual(len(buf.image_bytes), w * h * 3)


# ---------------------------------------------------------------------------
# TestFramePayloadNormalizerReal
# ---------------------------------------------------------------------------


class TestFramePayloadNormalizerReal(unittest.TestCase):
    """Real-normalizer tests: correct payloads, pixel accuracy, error paths."""

    def setUp(self):
        self.normalizer = FramePayloadNormalizer()

    # A. JPEG colour decode -----------------------------------------------

    def test_jpeg_colour_decode_correct_shape_and_format(self):
        # Solid red in BGR terms -> cv2.imencode -> decode -> RGB
        jpeg = _make_jpeg_bytes(4, 4, bgr=(0, 0, 255))  # B=0,G=0,R=255 => red in RGB
        msg = IngressFrameMessage(
            frame_id="f1", camera_id="cam-a", timestamp_ms=0,
            width=4, height=4, source_format="JPEG", payload_bytes=jpeg,
        )
        buf = self.normalizer.normalize(msg)
        self.assertEqual(buf.color_format, "RGB")
        self.assertEqual(buf.width, 4)
        self.assertEqual(buf.height, 4)
        self.assertEqual(len(buf.image_bytes), 4 * 4 * 3)
        arr = np.frombuffer(buf.image_bytes, dtype=np.uint8).reshape(4, 4, 3)
        # JPEG is lossy; just verify red channel is dominant
        self.assertGreater(int(arr[:, :, 0].mean()), 200)

    # B. JPEG grayscale decoded to RGB (R==G==B) --------------------------

    def test_jpeg_grayscale_expands_to_rgb_with_equal_channels(self):
        gray_img = np.full((4, 4), 128, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", gray_img)
        self.assertTrue(ok)
        msg = IngressFrameMessage(
            frame_id="f1", camera_id="cam-a", timestamp_ms=0,
            width=4, height=4, source_format="JPEG", payload_bytes=encoded.tobytes(),
        )
        buf = self.normalizer.normalize(msg)
        arr = np.frombuffer(buf.image_bytes, dtype=np.uint8).reshape(4, 4, 3)
        # All three channels must be equal at every pixel
        np.testing.assert_array_equal(arr[:, :, 0], arr[:, :, 1])
        np.testing.assert_array_equal(arr[:, :, 1], arr[:, :, 2])

    # C. GRAY8 all channels equal ----------------------------------------

    def test_gray8_all_channels_equal(self):
        gray_val = 192
        payload = bytes([gray_val] * 4 * 4)
        msg = IngressFrameMessage(
            frame_id="f1", camera_id="cam-a", timestamp_ms=0,
            width=4, height=4, source_format="GRAY8", payload_bytes=payload,
        )
        buf = self.normalizer.normalize(msg)
        arr = np.frombuffer(buf.image_bytes, dtype=np.uint8).reshape(4, 4, 3)
        np.testing.assert_array_equal(arr[:, :, 0], arr[:, :, 1])
        np.testing.assert_array_equal(arr[:, :, 1], arr[:, :, 2])
        self.assertEqual(int(arr[0, 0, 0]), gray_val)

    # D. BGR known pixel -> correct RGB swap ------------------------------

    def test_bgr_single_pixel_swapped_to_rgb(self):
        # 1x1 BGR: B=0, G=0, R=255 -> after swap: RGB = (255, 0, 0)
        payload = bytes([0, 0, 255])  # B, G, R
        msg = IngressFrameMessage(
            frame_id="f1", camera_id="cam-a", timestamp_ms=0,
            width=1, height=1, source_format="BGR", payload_bytes=payload,
        )
        buf = self.normalizer.normalize(msg)
        self.assertEqual(buf.image_bytes[0], 255)  # R
        self.assertEqual(buf.image_bytes[1], 0)    # G
        self.assertEqual(buf.image_bytes[2], 0)    # B

    # E. YUV420 shape + no crash ------------------------------------------

    def test_yuv420_output_shape_and_format(self):
        payload = bytes(4 * 4 * 3 // 2)
        msg = IngressFrameMessage(
            frame_id="f1", camera_id="cam-a", timestamp_ms=0,
            width=4, height=4, source_format="YUV420", payload_bytes=payload,
        )
        buf = self.normalizer.normalize(msg)
        self.assertEqual(buf.color_format, "RGB")
        self.assertEqual(len(buf.image_bytes), 4 * 4 * 3)

    # F. NV12 shape + no crash --------------------------------------------

    def test_nv12_output_shape_and_format(self):
        payload = bytes(4 * 4 * 3 // 2)
        msg = IngressFrameMessage(
            frame_id="f1", camera_id="cam-a", timestamp_ms=0,
            width=4, height=4, source_format="NV12", payload_bytes=payload,
        )
        buf = self.normalizer.normalize(msg)
        self.assertEqual(buf.color_format, "RGB")
        self.assertEqual(len(buf.image_bytes), 4 * 4 * 3)

    # G. YUY2 shape + no crash --------------------------------------------

    def test_yuy2_output_shape_and_format(self):
        payload = bytes(4 * 4 * 2)
        msg = IngressFrameMessage(
            frame_id="f1", camera_id="cam-a", timestamp_ms=0,
            width=4, height=4, source_format="YUY2", payload_bytes=payload,
        )
        buf = self.normalizer.normalize(msg)
        self.assertEqual(buf.color_format, "RGB")
        self.assertEqual(len(buf.image_bytes), 4 * 4 * 3)

    # H. Invalid JPEG raises FrameDecodeError (unit level) ----------------

    def test_invalid_jpeg_raises_frame_decode_error(self):
        msg = IngressFrameMessage(
            frame_id="f1", camera_id="cam-a", timestamp_ms=0,
            width=4, height=4, source_format="JPEG",
            payload_bytes=b"\x00\x01\x02\x03",
        )
        with self.assertRaises(FrameDecodeError):
            self.normalizer.normalize(msg)

    # I. CHW layout RGB transposed to HWC ---------------------------------

    def test_rgb_chw_layout_transposed_to_hwc(self):
        # Build a 2x2 RGB image in CHW order: C=3, H=2, W=2
        # Pixel (0,0)=red, (0,1)=green, (1,0)=blue, (1,1)=white
        chw = np.zeros((3, 2, 2), dtype=np.uint8)
        chw[0] = [[255, 0], [0, 255]]    # R channel
        chw[1] = [[0, 255], [0, 255]]    # G channel
        chw[2] = [[0, 0], [255, 255]]    # B channel
        hwc_expected = chw.transpose(1, 2, 0)  # expected HWC output
        msg = IngressFrameMessage(
            frame_id="f1", camera_id="cam-a", timestamp_ms=0,
            width=2, height=2, source_format="RGB",
            source_layout="CHW",
            payload_bytes=chw.tobytes(),
        )
        buf = self.normalizer.normalize(msg)
        arr = np.frombuffer(buf.image_bytes, dtype=np.uint8).reshape(2, 2, 3)
        np.testing.assert_array_equal(arr, hwc_expected)


# ---------------------------------------------------------------------------
# TestFramePacketBuilder
# ---------------------------------------------------------------------------


class TestFramePacketBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = FramePacketBuilder()

    def _make_buffer(self, w: int = 4, h: int = 4) -> NormalizedFrameBuffer:
        return NormalizedFrameBuffer(
            width=w,
            height=h,
            color_format="RGB",
            layout="HWC",
            dtype="uint8",
            value_range="[0,255]",
            num_color_channels=3,
            bits_per_channel=8,
            image_bytes=bytes(w * h * 3),
        )

    def test_preserves_identity_fields(self):
        msg = _make_rgb_message(frame_id="f-99", camera_id="cam-z", timestamp_ms=9999)
        buf = self._make_buffer()
        pkt = self.builder.build(msg, buf)
        self.assertEqual(pkt.frame_id, "f-99")
        self.assertEqual(pkt.camera_id, "cam-z")
        self.assertEqual(pkt.timestamp_ms, 9999)

    def test_always_rgb_layout_fields(self):
        msg = _make_rgb_message()
        buf = self._make_buffer()
        pkt = self.builder.build(msg, buf)
        self.assertEqual(pkt.pixel_format, "RGB")
        self.assertEqual(pkt.layout, "HWC")
        self.assertEqual(pkt.num_color_channels, 3)
        self.assertEqual(pkt.bits_per_channel, 8)

    def test_image_bytes_taken_from_buffer(self):
        msg = _make_rgb_message()
        image_data = bytes(range(48))
        buf = self._make_buffer()
        buf.image_bytes = image_data
        pkt = self.builder.build(msg, buf)
        self.assertEqual(pkt.image_bytes, image_data)

    def test_dimensions_taken_from_buffer(self):
        msg = _make_rgb_message(width=8, height=6)
        buf = self._make_buffer(w=8, h=6)
        pkt = self.builder.build(msg, buf)
        self.assertEqual(pkt.width, 8)
        self.assertEqual(pkt.height, 6)


# ---------------------------------------------------------------------------
# TestFramePacketImmutability
# ---------------------------------------------------------------------------


class TestFramePacketImmutability(unittest.TestCase):
    def _make_packet(self) -> FramePacket:
        return FramePacket(
            frame_id="f1",
            camera_id="cam-a",
            timestamp_ms=1000,
            width=4,
            height=4,
            pixel_format="RGB",
            layout="HWC",
            num_color_channels=3,
            bits_per_channel=8,
            image_bytes=bytes(48),
        )

    def test_frame_id_is_frozen(self):
        pkt = self._make_packet()
        with self.assertRaises(FrozenInstanceError):
            pkt.frame_id = "other"  # type: ignore[misc]

    def test_image_bytes_is_frozen(self):
        pkt = self._make_packet()
        with self.assertRaises(FrozenInstanceError):
            pkt.image_bytes = bytes(48)  # type: ignore[misc]

    def test_pixel_format_is_frozen(self):
        pkt = self._make_packet()
        with self.assertRaises(FrozenInstanceError):
            pkt.pixel_format = "BGR"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestFrameStore
# ---------------------------------------------------------------------------


class TestFrameStore(unittest.TestCase):
    def _make_packet(self, camera_id: str, frame_id: str = "f1") -> FramePacket:
        return FramePacket(
            frame_id=frame_id,
            camera_id=camera_id,
            timestamp_ms=1000,
            width=4,
            height=4,
            pixel_format="RGB",
            layout="HWC",
            num_color_channels=3,
            bits_per_channel=8,
            image_bytes=bytes(48),
        )

    def test_empty_store_returns_none(self):
        store = FrameStore(max_camera_slots=4, camera_order=["cam-a"])
        store.start()
        self.assertIsNone(store.get_next_frame())

    def test_fifo_ordering_single_camera(self):
        store = FrameStore(max_camera_slots=4, camera_order=["cam-a"])
        store.start()
        p1 = self._make_packet("cam-a", "f1")
        p2 = self._make_packet("cam-a", "f2")
        p3 = self._make_packet("cam-a", "f3")
        store.enqueue(p1)
        store.enqueue(p2)
        store.enqueue(p3)
        self.assertEqual(store.get_next_frame().frame_id, "f1")
        self.assertEqual(store.get_next_frame().frame_id, "f2")
        self.assertEqual(store.get_next_frame().frame_id, "f3")
        self.assertIsNone(store.get_next_frame())

    def test_round_robin_across_cameras(self):
        store = FrameStore(max_camera_slots=4, camera_order=["cam-a", "cam-b"])
        store.start()
        a1 = self._make_packet("cam-a", "a1")
        a2 = self._make_packet("cam-a", "a2")
        b1 = self._make_packet("cam-b", "b1")
        store.enqueue(a1)
        store.enqueue(a2)
        store.enqueue(b1)

        # Round-robin: cam-a → cam-b → cam-a
        r1 = store.get_next_frame()
        r2 = store.get_next_frame()
        r3 = store.get_next_frame()
        r4 = store.get_next_frame()

        self.assertEqual(r1.frame_id, "a1")
        self.assertEqual(r2.frame_id, "b1")
        self.assertEqual(r3.frame_id, "a2")
        self.assertIsNone(r4)

    def test_max_camera_slots_rejection(self):
        store = FrameStore(max_camera_slots=1, camera_order=["cam-a"])
        store.start()
        store.enqueue(self._make_packet("cam-a", "a1"))
        with self.assertRaises(CameraSlotExhaustedError):
            store.enqueue(self._make_packet("cam-b", "b1"))  # second camera rejected

    def test_new_camera_appended_when_slot_available(self):
        store = FrameStore(max_camera_slots=2, camera_order=["cam-a"])
        store.start()
        # cam-b is unknown but a slot is available
        store.enqueue(self._make_packet("cam-b", "b1"))
        pkt = store.get_next_frame()
        # cam-a is first, empty; cam-b is second, has frame
        self.assertIsNotNone(pkt)
        self.assertEqual(pkt.camera_id, "cam-b")

    def test_stop_preserves_queued_frames(self):
        store = FrameStore(max_camera_slots=4, camera_order=["cam-a"])
        store.start()
        store.enqueue(self._make_packet("cam-a", "f1"))
        store.stop()
        # Frames must still be dequeue-able after stop()
        pkt = store.get_next_frame()
        self.assertIsNotNone(pkt)
        self.assertEqual(pkt.frame_id, "f1")

    def test_health_reflects_state(self):
        store = FrameStore(max_camera_slots=4, camera_order=["cam-a"])
        self.assertEqual(store.health().state, "STOPPED")
        store.start()
        self.assertEqual(store.health().state, "READY")
        store.enqueue(self._make_packet("cam-a", "f1"))
        h = store.health()
        self.assertEqual(h.active_cameras, 1)
        self.assertEqual(h.total_queued_frames, 1)
        self.assertEqual(h.queue_depth_per_camera["cam-a"], 1)
        store.stop()
        self.assertEqual(store.health().state, "STOPPED")


# ---------------------------------------------------------------------------
# TestFrameIngestionGatewayLifecycle
# ---------------------------------------------------------------------------


class TestFrameIngestionGatewayLifecycle(unittest.TestCase):
    def test_get_next_frame_before_start_returns_none(self):
        transport = StubFrameIngressTransport()
        gw = FrameIngestionGateway(transport=transport)
        gw.configure(_make_config())
        self.assertIsNone(gw.get_next_frame())

    def test_configure_after_start_raises(self):
        transport = StubFrameIngressTransport()
        gw = FrameIngestionGateway(transport=transport)
        gw.configure(_make_config())
        gw.start()
        try:
            with self.assertRaises(GatewayLifecycleError):
                gw.configure(_make_config())
        finally:
            gw.stop()

    def test_start_is_idempotent(self):
        transport = StubFrameIngressTransport()
        gw = FrameIngestionGateway(transport=transport)
        gw.configure(_make_config())
        gw.start()
        try:
            gw.start()  # second call must be a no-op, not raise
        finally:
            gw.stop()

    def test_health_running_after_start(self):
        transport = StubFrameIngressTransport()
        gw = FrameIngestionGateway(transport=transport)
        gw.configure(_make_config())
        gw.start()
        try:
            h = gw.health()
            self.assertEqual(h.transport_state, "RUNNING")
            self.assertEqual(h.store_state, "READY")
        finally:
            gw.stop()

    def test_health_stopped_after_stop(self):
        transport = StubFrameIngressTransport()
        gw = FrameIngestionGateway(transport=transport)
        gw.configure(_make_config())
        gw.start()
        gw.stop()
        h = gw.health()
        self.assertEqual(h.transport_state, "STOPPED")


# ---------------------------------------------------------------------------
# TestFrameIngestionGatewayIntegration
# ---------------------------------------------------------------------------


class TestFrameIngestionGatewayIntegration(unittest.TestCase):
    """End-to-end integration tests using StubFrameIngressTransport."""

    def setUp(self):
        self.transport = StubFrameIngressTransport()
        self.gw: FrameIngestionGateway | None = None

    def tearDown(self):
        if self.gw is not None:
            self.gw.stop()

    def _start(self, **config_overrides) -> FrameIngestionGateway:
        self.gw = _start_gateway(self.transport, **config_overrides)
        return self.gw

    # --- valid RGB end-to-end flow ---

    def test_valid_rgb_end_to_end(self):
        gw = self._start()
        msg = _make_rgb_message(frame_id="f-42", camera_id="cam-1", timestamp_ms=5000)
        self.transport.inject_message(msg)
        pkt = _poll_frame(gw)
        self.assertIsNotNone(pkt)
        self.assertEqual(pkt.frame_id, "f-42")
        self.assertEqual(pkt.camera_id, "cam-1")
        self.assertEqual(pkt.timestamp_ms, 5000)
        self.assertEqual(pkt.pixel_format, "RGB")
        self.assertEqual(pkt.layout, "HWC")
        self.assertEqual(pkt.num_color_channels, 3)
        self.assertEqual(pkt.bits_per_channel, 8)
        self.assertEqual(pkt.width, 4)
        self.assertEqual(pkt.height, 4)
        self.assertEqual(len(pkt.image_bytes), 4 * 4 * 3)

    def test_frame_packet_canonical_invariants(self):
        gw = self._start()
        self.transport.inject_message(_make_rgb_message())
        pkt = _poll_frame(gw)
        self.assertIsNotNone(pkt)
        self.assertEqual(pkt.pixel_format, "RGB")
        self.assertEqual(pkt.layout, "HWC")
        self.assertEqual(pkt.num_color_channels, 3)
        self.assertEqual(pkt.bits_per_channel, 8)

    # --- rejection tests (silent discard) ---

    def _inject_then_marker(self, bad_msg: IngressFrameMessage) -> FramePacket | None:
        """Inject bad message then a valid marker. Return the marker or None if it never arrives."""
        marker = _make_rgb_message(frame_id="marker", camera_id="cam-marker")
        self.transport.inject_message(bad_msg)
        self.transport.inject_message(marker)
        return _poll_frame(self.gw)

    def test_missing_frame_id_is_discarded(self):
        self._start()
        bad = _make_rgb_message()
        bad.frame_id = ""
        result = self._inject_then_marker(bad)
        self.assertIsNotNone(result)
        self.assertEqual(result.frame_id, "marker")

    def test_unsupported_source_format_is_discarded(self):
        gw = self._start(supported_source_formats=["RGB"])
        bad = _make_rgb_message()
        bad.source_format = "H264"
        bad.payload_bytes = bytes(48)
        result = self._inject_then_marker(bad)
        self.assertIsNotNone(result)
        self.assertEqual(result.frame_id, "marker")

    def test_payload_size_mismatch_is_discarded(self):
        self._start()
        bad = _make_rgb_message()
        bad.payload_bytes = bytes(10)  # wrong size for 4x4 RGB (expected 48)
        result = self._inject_then_marker(bad)
        self.assertIsNotNone(result)
        self.assertEqual(result.frame_id, "marker")

    # --- FIFO ordering ---

    def test_fifo_ordering_same_camera(self):
        gw = self._start()
        for i in range(5):
            self.transport.inject_message(
                _make_rgb_message(frame_id=f"f-{i}", camera_id="cam-a")
            )
        frames = []
        for _ in range(5):
            pkt = _poll_frame(gw)
            self.assertIsNotNone(pkt)
            frames.append(pkt.frame_id)
        self.assertEqual(frames, ["f-0", "f-1", "f-2", "f-3", "f-4"])

    # --- round-robin retrieval ---

    def test_round_robin_across_two_cameras(self):
        gw = self._start(camera_order=["cam-a", "cam-b"])
        self.transport.inject_message(_make_rgb_message(frame_id="a1", camera_id="cam-a"))
        self.transport.inject_message(_make_rgb_message(frame_id="a2", camera_id="cam-a"))
        self.transport.inject_message(_make_rgb_message(frame_id="b1", camera_id="cam-b"))

        # Wait until all 3 are enqueued
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            h = gw._store.health()
            if h.total_queued_frames == 3:
                break
            time.sleep(0.01)

        r1 = gw.get_next_frame()
        r2 = gw.get_next_frame()
        r3 = gw.get_next_frame()
        r4 = gw.get_next_frame()

        self.assertIsNotNone(r1)
        self.assertIsNotNone(r2)
        self.assertIsNotNone(r3)
        self.assertIsNone(r4)
        # cam-a is first in camera_order → first dequeue from cam-a
        self.assertEqual(r1.frame_id, "a1")
        # next: cam-b
        self.assertEqual(r2.frame_id, "b1")
        # next: cam-a again
        self.assertEqual(r3.frame_id, "a2")

    # --- max_camera_slots ---

    def test_max_camera_slots_rejection(self):
        gw = self._start(max_camera_slots=1)
        # cam-a occupies the single slot
        self.transport.inject_message(_make_rgb_message(frame_id="a1", camera_id="cam-a"))
        # cam-b should be rejected silently
        self.transport.inject_message(_make_rgb_message(frame_id="b1", camera_id="cam-b"))
        # marker from cam-a to ensure the pipeline has processed both
        self.transport.inject_message(_make_rgb_message(frame_id="a2", camera_id="cam-a"))

        frames = []
        for _ in range(3):
            pkt = _poll_frame(gw, timeout=2.0)
            if pkt is not None:
                frames.append(pkt)

        frame_ids = [p.frame_id for p in frames]
        self.assertIn("a1", frame_ids)
        self.assertIn("a2", frame_ids)
        self.assertNotIn("b1", frame_ids)

    # --- non-RGB format produces canonical RGB output ---

    def test_bgr_format_produces_canonical_rgb_output(self):
        gw = self._start()
        # BGR 4x4 correct payload size = 48
        bgr_msg = IngressFrameMessage(
            frame_id="bgr-1",
            camera_id="cam-a",
            timestamp_ms=1000,
            width=4,
            height=4,
            source_format="BGR",
            payload_bytes=bytes(48),
        )
        self.transport.inject_message(bgr_msg)
        pkt = _poll_frame(gw)
        self.assertIsNotNone(pkt)
        self.assertEqual(pkt.pixel_format, "RGB")
        self.assertEqual(pkt.layout, "HWC")
        self.assertEqual(pkt.num_color_channels, 3)
        self.assertEqual(pkt.bits_per_channel, 8)
        self.assertEqual(len(pkt.image_bytes), 4 * 4 * 3)

    def test_jpeg_format_produces_canonical_rgb_output(self):
        gw = self._start()
        jpeg_msg = IngressFrameMessage(
            frame_id="jpeg-1",
            camera_id="cam-a",
            timestamp_ms=1000,
            width=4,
            height=4,
            source_format="JPEG",
            payload_bytes=_make_jpeg_bytes(4, 4),
        )
        self.transport.inject_message(jpeg_msg)
        pkt = _poll_frame(gw)
        self.assertIsNotNone(pkt)
        self.assertEqual(pkt.pixel_format, "RGB")
        self.assertEqual(len(pkt.image_bytes), 4 * 4 * 3)

    # --- stop preserves queued frames ---

    def test_stop_preserves_queued_frames(self):
        gw = self._start()
        self.transport.inject_message(_make_rgb_message(frame_id="f1"))
        self.transport.inject_message(_make_rgb_message(frame_id="f2"))

        # Wait for both to be enqueued
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            h = gw._store.health()
            if h.total_queued_frames >= 2:
                break
            time.sleep(0.01)

        gw.stop()
        self.gw = None  # prevent double stop in tearDown

        # Frames must still be retrievable after stop
        pkt1 = gw.get_next_frame()
        pkt2 = gw.get_next_frame()
        self.assertIsNotNone(pkt1)
        self.assertIsNotNone(pkt2)
        self.assertIsNone(gw.get_next_frame())

    # --- invalid JPEG is silently discarded ---

    def test_invalid_jpeg_is_discarded_and_marker_arrives(self):
        gw = self._start()
        invalid_jpeg = IngressFrameMessage(
            frame_id="bad-jpeg",
            camera_id="cam-a",
            timestamp_ms=1000,
            width=4,
            height=4,
            source_format="JPEG",
            payload_bytes=b"\x00\x01\x02\x03",  # not a valid JPEG
        )
        marker = _make_rgb_message(frame_id="marker", camera_id="cam-a")
        self.transport.inject_message(invalid_jpeg)
        self.transport.inject_message(marker)
        pkt = _poll_frame(gw)
        self.assertIsNotNone(pkt)
        self.assertEqual(pkt.frame_id, "marker")
        # The gateway must have counted the decode failure
        self.assertEqual(gw._decode_failed_total, 1)


if __name__ == "__main__":
    unittest.main()
