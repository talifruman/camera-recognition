"""
Real Visual Image Test — Frame Ingestion Gateway
=================================================

NOTE: This is NOT a unit test. Do not run it with pytest.
Run it directly:

    python tests/frame_ingestion_gateway/real_visual_image_test.py

Purpose
-------
Visually verify that the REAL FramePayloadNormalizer can:
  1. Accept a real JPEG image file as payload.
  2. Decode and normalize it into canonical RGB / HWC / uint8 / [0,255].
  3. Build a FramePacket, enqueue it, and retrieve it through the full
     FrameIngestionGateway pipeline.
  4. Save the output back as a PNG for visual inspection.
  5. Save a side-by-side comparison of the original and the output.

Prerequisites
-------------
Place a JPEG test image at:

    tests/frame_ingestion_gateway/assets/input_test_image.jpg

Expected output (written to tests/frame_ingestion_gateway/visual_outputs/)
---------------------------------------------------------------------------
  real_jpeg_output.png      — the decoded FramePacket rendered as PNG
  real_jpeg_comparison.png  — original (left) vs. output (right) side-by-side
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from image_processing.frame_ingestion_gateway import (  # type: ignore[import-not-found]
    FrameIngestionGateway,
    FrameIngestionGatewayConfig,
    FramePacket,
    IngressFrameMessage,
    StubFrameIngressTransport,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ASSETS_DIR = Path(__file__).parent / "assets"
_INPUT_IMAGE = _ASSETS_DIR / "input_test_image.jpg"
_OUTPUT_DIR = Path(__file__).parent / "visual_outputs"
_OUTPUT_IMAGE = _OUTPUT_DIR / "real_jpeg_output.png"
_COMPARISON_IMAGE = _OUTPUT_DIR / "real_jpeg_comparison.png"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _poll_frame(gateway: FrameIngestionGateway, timeout: float = 5.0) -> FramePacket | None:
    """Poll get_next_frame() until a packet arrives or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pkt = gateway.get_next_frame()
        if pkt is not None:
            return pkt
        time.sleep(0.005)
    return None


def _save_array_as_png(rgb_array: np.ndarray, path: Path) -> None:
    """Save an HWC RGB uint8 ndarray as a PNG file."""
    bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def _make_comparison(left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
    """Return a side-by-side BGR image suitable for imwrite."""
    # Resize right to match left's height if they differ (shouldn't happen here)
    if left_rgb.shape[0] != right_rgb.shape[0]:
        right_rgb = cv2.resize(
            right_rgb,
            (int(right_rgb.shape[1] * left_rgb.shape[0] / right_rgb.shape[0]),
             left_rgb.shape[0]),
        )

    gap = np.full((left_rgb.shape[0], 8, 3), 200, dtype=np.uint8)  # light gray divider
    side_by_side = np.concatenate([left_rgb, gap, right_rgb], axis=1)
    return cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # -----------------------------------------------------------------------
    # 1. Ensure assets directory exists; prompt user if image is missing
    # -----------------------------------------------------------------------
    _ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not _INPUT_IMAGE.exists():
        print()
        print("=" * 60)
        print("  INPUT IMAGE NOT FOUND")
        print("=" * 60)
        print()
        print(f"  Expected: {_INPUT_IMAGE}")
        print()
        print("  Please place a JPEG image at that path and re-run:")
        print()
        print("    python tests/frame_ingestion_gateway/real_visual_image_test.py")
        print()
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 2. Read JPEG bytes from disk
    # -----------------------------------------------------------------------
    jpeg_bytes = _INPUT_IMAGE.read_bytes()

    # -----------------------------------------------------------------------
    # 3. Decode once only to determine width / height
    # -----------------------------------------------------------------------
    raw = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    decoded_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        print(f"ERROR: cv2.imdecode could not read '{_INPUT_IMAGE}'. "
              "Ensure it is a valid JPEG file.")
        sys.exit(1)

    img_height, img_width = decoded_bgr.shape[:2]
    original_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)

    # -----------------------------------------------------------------------
    # 4. Build IngressFrameMessage
    # -----------------------------------------------------------------------
    message = IngressFrameMessage(
        frame_id="real_jpeg_frame_001",
        camera_id="cam_real_1",
        timestamp_ms=int(time.time() * 1000),
        width=img_width,
        height=img_height,
        source_format="JPEG",
        payload_bytes=jpeg_bytes,
    )

    # -----------------------------------------------------------------------
    # 5. Wire up the REAL FrameIngestionGateway
    # -----------------------------------------------------------------------
    transport = StubFrameIngressTransport()

    config = FrameIngestionGatewayConfig(
        bind_address="0.0.0.0:50061",
        service_name="real-visual-image-test",
        max_concurrent_streams=1,
        max_camera_slots=1,
        camera_order=["cam_real_1"],
        supported_source_formats=["JPEG"],
    )

    gateway = FrameIngestionGateway(transport=transport)
    gateway.configure(config)
    gateway.start()

    # -----------------------------------------------------------------------
    # 6. Inject message and retrieve FramePacket
    # -----------------------------------------------------------------------
    transport.inject_message(message)
    packet = _poll_frame(gateway)
    gateway.stop()

    if packet is None:
        print("ERROR: Gateway did not produce a FramePacket within the timeout.")
        print("       Check that the JPEG is valid and the format is supported.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 7. Reconstruct RGB array from FramePacket
    # -----------------------------------------------------------------------
    output_rgb = np.frombuffer(packet.image_bytes, dtype=np.uint8).reshape(
        (packet.height, packet.width, 3)
    )

    # -----------------------------------------------------------------------
    # 8. Save output PNG and side-by-side comparison
    # -----------------------------------------------------------------------
    _save_array_as_png(output_rgb, _OUTPUT_IMAGE)
    comparison_bgr = _make_comparison(original_rgb, output_rgb)
    cv2.imwrite(str(_COMPARISON_IMAGE), comparison_bgr)

    # -----------------------------------------------------------------------
    # 9. Print summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  Real Visual Image Test — Frame Ingestion Gateway")
    print("=" * 60)
    print()
    print(f"  Input image    : {_INPUT_IMAGE}")
    print(f"  Output image   : {_OUTPUT_IMAGE}")
    print(f"  Comparison     : {_COMPARISON_IMAGE}")
    print()
    print("  FramePacket fields:")
    print(f"    frame_id           : {packet.frame_id}")
    print(f"    camera_id          : {packet.camera_id}")
    print(f"    width              : {packet.width}")
    print(f"    height             : {packet.height}")
    print(f"    pixel_format       : {packet.pixel_format}")
    print(f"    layout             : {packet.layout}")
    print(f"    num_color_channels : {packet.num_color_channels}")
    print(f"    bits_per_channel   : {packet.bits_per_channel}")
    print(f"    image_bytes length : {len(packet.image_bytes)}")
    print()
    print("  Expected: real_jpeg_output.png matches the original image.")
    print("  Expected: real_jpeg_comparison.png shows original (left) vs. output (right).")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
