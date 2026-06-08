"""
Visual Smoke Test — Frame Ingestion Gateway (real normalizer)
=============================================================

NOTE: This is NOT a unit test. Do not run it with pytest.
Run it directly:

    python tests/frame_ingestion_gateway/visual_smoke_test.py

Purpose
-------
Visually verify the FrameIngestionGateway end-to-end by:
  1. Generating synthetic frames in multiple source formats:
       - Solid-color raw RGB frames
       - A JPEG-encoded frame
       - A GRAY8 frame
  2. Wrapping them as IngressFrameMessage objects.
  3. Feeding them through the StubFrameIngressTransport.
  4. Retrieving FramePacket objects via get_next_frame().
  5. Saving each packet as a PNG image for visual inspection.

Expected output: 6 PNG files.
  01_cam_1_frame_1.png  — solid red
  02_cam_2_frame_1.png  — solid green
  03_cam_1_frame_2.png  — solid blue
  04_cam_2_frame_2.png  — solid yellow
  05_cam_1_frame_3.png  — solid magenta (from JPEG source)
  06_cam_2_frame_3.png  — uniform gray (from GRAY8 source)
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
# Constants
# ---------------------------------------------------------------------------

WIDTH = 160
HEIGHT = 100
OUTPUT_DIR = Path(__file__).parent / "visual_outputs"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rgb_bytes(r: int, g: int, b: int, width: int, height: int) -> bytes:
    """Return raw HWC RGB bytes for a solid-color image."""
    pixel = bytes([r, g, b])
    return pixel * (width * height)


def _make_jpeg_bytes(r: int, g: int, b: int, width: int, height: int) -> bytes:
    """Encode a solid-color image as JPEG bytes (BGR order for cv2)."""
    bgr_img = np.full((height, width, 3), [b, g, r], dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", bgr_img)
    if not ok:
        raise RuntimeError("cv2.imencode failed while building JPEG smoke-test frame")
    return encoded.tobytes()


def _make_gray8_bytes(gray_val: int, width: int, height: int) -> bytes:
    """Return raw GRAY8 bytes for a uniform-brightness image."""
    return bytes([gray_val]) * (width * height)


def _save_png(packet: FramePacket, path: Path) -> None:
    """Save a FramePacket's RGB bytes as a PNG file using Pillow or OpenCV."""
    try:
        from PIL import Image  # type: ignore[import-not-found]
        img = Image.frombytes(
            "RGB",
            (packet.width, packet.height),
            packet.image_bytes,
        )
        img.save(path, format="PNG")
        return
    except ImportError:
        pass

    arr = np.frombuffer(packet.image_bytes, dtype=np.uint8).reshape(
        (packet.height, packet.width, 3)
    )
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def _poll_frames(gateway: FrameIngestionGateway, count: int, timeout: float = 5.0) -> list[FramePacket]:
    """Poll get_next_frame() until 'count' packets are retrieved or timeout expires."""
    packets: list[FramePacket] = []
    deadline = time.monotonic() + timeout
    while len(packets) < count and time.monotonic() < deadline:
        pkt = gateway.get_next_frame()
        if pkt is not None:
            packets.append(pkt)
        else:
            time.sleep(0.005)
    return packets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Build IngressFrameMessage objects for all 6 frames
    # -----------------------------------------------------------------------
    messages: list[IngressFrameMessage] = []

    # Frames 1–4: solid-color raw RGB
    rgb_frames = [
        ("cam_1", "frame_1", (255,   0,   0)),   # red
        ("cam_2", "frame_1", (  0, 255,   0)),   # green
        ("cam_1", "frame_2", (  0,   0, 255)),   # blue
        ("cam_2", "frame_2", (255, 255,   0)),   # yellow
    ]
    for camera_id, frame_id, (r, g, b) in rgb_frames:
        messages.append(IngressFrameMessage(
            frame_id=frame_id,
            camera_id=camera_id,
            timestamp_ms=0,
            width=WIDTH,
            height=HEIGHT,
            source_format="RGB",
            source_layout="HWC",
            source_num_color_channels=3,
            source_bits_per_channel=8,
            payload_bytes=_make_rgb_bytes(r, g, b, WIDTH, HEIGHT),
        ))

    # Frame 5: solid magenta encoded as JPEG (cam_1 / frame_3)
    messages.append(IngressFrameMessage(
        frame_id="frame_3",
        camera_id="cam_1",
        timestamp_ms=0,
        width=WIDTH,
        height=HEIGHT,
        source_format="JPEG",
        payload_bytes=_make_jpeg_bytes(255, 0, 255, WIDTH, HEIGHT),  # magenta
    ))

    # Frame 6: uniform gray as GRAY8 (cam_2 / frame_3)
    messages.append(IngressFrameMessage(
        frame_id="frame_3",
        camera_id="cam_2",
        timestamp_ms=0,
        width=WIDTH,
        height=HEIGHT,
        source_format="GRAY8",
        source_num_color_channels=1,
        source_bits_per_channel=8,
        payload_bytes=_make_gray8_bytes(192, WIDTH, HEIGHT),
    ))

    # -----------------------------------------------------------------------
    # Wire up the gateway with the STUB transport
    # -----------------------------------------------------------------------
    transport = StubFrameIngressTransport()

    config = FrameIngestionGatewayConfig(
        bind_address="0.0.0.0:50061",
        service_name="visual-smoke-test",
        max_concurrent_streams=4,
        max_camera_slots=4,
        camera_order=["cam_1", "cam_2"],
        supported_source_formats=["RGB", "JPEG", "GRAY8"],
    )

    gateway = FrameIngestionGateway(transport=transport)
    gateway.configure(config)
    gateway.start()

    # --- Inject all messages into the stub transport ---
    for msg in messages:
        transport.inject_message(msg)

    # --- Retrieve processed FramePackets ---
    packets = _poll_frames(gateway, count=len(messages))

    # --- Stop the gateway ---
    gateway.stop()

    # --- Save PNG images ---
    saved_filenames: list[str] = []
    for idx, packet in enumerate(packets, start=1):
        filename = f"{idx:02d}_{packet.camera_id}_{packet.frame_id}.png"
        output_path = OUTPUT_DIR / filename
        _save_png(packet, output_path)
        saved_filenames.append(filename)

    # --- Summary ---
    print()
    print("=" * 52)
    print("  Visual Smoke Test — STUB Frame Ingestion Gateway")
    print("=" * 52)
    print(f"  Messages processed : {len(messages)}")
    print(f"  FramePackets saved : {len(packets)}")
    print(f"  Output directory   : {OUTPUT_DIR}")
    print()
    print("  Saved images (in retrieval order):")
    for name in saved_filenames:
        print(f"    {name}")
    print("=" * 52)
    print()

    if len(packets) < len(messages):
        missing = len(messages) - len(packets)
        print(f"  WARNING: {missing} message(s) were not retrieved within the timeout.")
        sys.exit(1)


if __name__ == "__main__":
    main()
