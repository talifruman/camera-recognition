"""Visual smoke runner for Frame Ingestion Gateway.

This script discovers image assets, pushes multiple ingress format variants
through the real gateway normalize/build/publish path, captures published
FramePacket objects via an in-memory sink, writes reconstructed output images,
builds a contact sheet, and emits a JSON summary.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_processing.frame_ingestion_gateway import (  # type: ignore[import-not-found]
    FrameIngestionGateway,
    FrameIngestionGatewayConfig,
    InMemoryFrameIngressTransport,
    IngressFrameMessage,
)
from image_processing.shared.contracts import (  # type: ignore[import-not-found]
    EnqueueResult,
    FramePacket,
    FramePacketSink,
)

ASSETS_DIR = REPO_ROOT / "tests" / "frame_ingestion_gateway" / "assets"
OUTPUT_DIR = REPO_ROOT / "tests" / "results" / "frame_ingestion_gateway_visual"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
CONTACT_SHEET_PATH = OUTPUT_DIR / "contact_sheet.png"
CAMERA_ID = "camera_1"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CASE_ORDER = ["RGB", "BGR", "GRAY8", "JPEG", "MJPEG"]
MAX_WAIT_SECONDS = 20.0
WAIT_POLL_SECONDS = 0.01
IMAGE_TILE_WIDTH = 240
IMAGE_TILE_HEIGHT = 150
TILE_LABEL_HEIGHT = 24
ROW_GAP = 8
COL_GAP = 8
TEXT_COLOR_DARK = (25, 25, 25)
TEXT_COLOR_LIGHT = (250, 250, 250)
TILE_BG = (235, 235, 235)
PLACEHOLDER_BG = (210, 210, 210)
BASE_TIMESTAMP_MS = int(time.time() * 1000)


@dataclass(frozen=True)
class AssetRecord:
    """Decoded source asset and metadata used for case generation."""

    path: Path
    stem: str
    width: int
    height: int
    rgb_image: np.ndarray
    bgr_image: np.ndarray


@dataclass(frozen=True)
class IngressCase:
    """Single generated ingress case for one source asset."""

    input_file: Path
    asset_stem: str
    source_format: str
    frame_id: str
    camera_id: str
    timestamp_ms: int
    width: int
    height: int
    payload_bytes: bytes
    source_layout: str | None
    source_bits_per_channel: int | None


class CapturingSink(FramePacketSink):
    """In-memory sink that stores accepted FramePacket objects."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._packets: list[FramePacket] = []

    def enqueue(self, frame_packet: FramePacket) -> EnqueueResult:
        """Store a published packet and return accepted result."""
        with self._lock:
            self._packets.append(frame_packet)
        return {"accepted": True, "reason": None}

    def count(self) -> int:
        """Return number of captured packets."""
        with self._lock:
            return len(self._packets)

    def snapshot(self) -> list[FramePacket]:
        """Return a copy of captured packets."""
        with self._lock:
            return list(self._packets)


class SkippedCase(TypedDict):
    """Skipped ingress case entry persisted to summary."""

    case: str
    reason: str


class CanonicalValidation(TypedDict):
    """Canonical packet validation result for one case."""

    passed: bool
    errors: list[str]


class CaseResult(TypedDict):
    """Per-case execution and validation result."""

    source_format: str
    frame_id: str
    camera_id: str
    width: int | None
    height: int | None
    canonical_validation: CanonicalValidation
    output_file: str | None
    errors: list[str]


class AssetSummary(TypedDict):
    """Per-asset summary persisted to summary JSON."""

    input_file: str
    generated_cases: list[str]
    skipped_cases: list[SkippedCase]
    output_file_paths: dict[str, str]
    cases: list[CaseResult]
    errors: list[str]
    width: int | None
    height: int | None


class TotalsSummary(TypedDict):
    """Aggregate smoke run counters."""

    input_images_found: int
    cases_generated: int
    outputs_saved: int
    skipped_cases: int
    failures: int
    packets_captured: int


class RunSummary(TypedDict):
    """Top-level summary artifact structure."""

    input_directory: str
    output_directory: str
    contact_sheet: str
    summary_json: str
    totals: TotalsSummary
    assets: list[AssetSummary]


def discover_asset_files(directory: Path) -> list[Path]:
    """Return all supported input image files in deterministic order."""
    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name.lower())


def clean_output_directory(directory: Path) -> None:
    """Delete previous output directory and recreate it empty."""
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def sanitize_stem(stem: str) -> str:
    """Return filesystem-safe lowercase stem for frame_id and file names."""
    normalized = re.sub(r"[^A-Za-z0-9_-]", "_", stem).strip("_")
    return normalized.lower() or "asset"


def load_asset_record(path: Path) -> AssetRecord | None:
    """Load one asset from disk and normalize it to RGB and BGR arrays."""
    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        return None
    if decoded.ndim == 2:
        bgr = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)
    elif decoded.ndim == 3 and decoded.shape[2] == 4:
        bgr = cv2.cvtColor(decoded, cv2.COLOR_BGRA2BGR)
    elif decoded.ndim == 3 and decoded.shape[2] == 3:
        bgr = decoded
    else:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return AssetRecord(
        path=path,
        stem=sanitize_stem(path.stem),
        width=int(bgr.shape[1]),
        height=int(bgr.shape[0]),
        rgb_image=rgb,
        bgr_image=bgr,
    )


def build_ingress_cases(
    asset: AssetRecord,
    case_start_index: int,
) -> tuple[list[IngressCase], list[SkippedCase]]:
    """Build ingress cases and skipped-case entries for one asset."""
    generated: list[IngressCase] = []
    skipped: list[SkippedCase] = []
    gray = cv2.cvtColor(asset.bgr_image, cv2.COLOR_BGR2GRAY)
    jpeg_ok, jpeg_encoded = cv2.imencode(".jpg", asset.bgr_image)
    timestamp_cursor = case_start_index
    for case_name in CASE_ORDER:
        payload: bytes | None = None
        source_layout: str | None = None
        source_bits: int | None = None
        if case_name == "RGB":
            payload = asset.rgb_image.tobytes()
            source_layout = "HWC"
            source_bits = 8
        elif case_name == "BGR":
            payload = asset.bgr_image.tobytes()
            source_layout = "HWC"
            source_bits = 8
        elif case_name == "GRAY8":
            payload = gray.tobytes()
            source_layout = "HWC"
            source_bits = 8
        elif case_name in {"JPEG", "MJPEG"}:
            if jpeg_ok:
                payload = jpeg_encoded.tobytes()
            else:
                skipped.append(
                    {
                        "case": case_name,
                        "reason": "opencv_jpeg_encode_failed",
                    }
                )
        if payload is None:
            if case_name not in {"JPEG", "MJPEG"}:
                skipped.append(
                    {
                        "case": case_name,
                        "reason": "case_payload_generation_failed",
                    }
                )
            continue
        frame_id = f"{asset.stem}__{case_name.lower()}"
        generated.append(
            IngressCase(
                input_file=asset.path,
                asset_stem=asset.stem,
                source_format=case_name,
                frame_id=frame_id,
                camera_id=CAMERA_ID,
                timestamp_ms=BASE_TIMESTAMP_MS + timestamp_cursor,
                width=asset.width,
                height=asset.height,
                payload_bytes=payload,
                source_layout=source_layout,
                source_bits_per_channel=source_bits,
            )
        )
        timestamp_cursor += 1
    return generated, skipped


def make_gateway_config(sink: FramePacketSink) -> FrameIngestionGatewayConfig:
    """Build visual-smoke gateway config for in-memory ingestion execution."""
    return FrameIngestionGatewayConfig(
        bind_address="127.0.0.1:50061",
        service_name="fig-visual-smoke",
        max_concurrent_streams=16,
        configured_camera_ids=[CAMERA_ID],
        supported_source_formats=list(CASE_ORDER),
        max_frame_width=8192,
        max_frame_height=8192,
        max_payload_bytes=80 * 1024 * 1024,
        allowed_timestamp_skew_ms=60_000,
        allowed_timestamp_lag_ms=60_000,
        strict_timestamp_validation_enabled=False,
        duplicate_frame_tracking_enabled=False,
        duplicate_frame_tracking_window_size=64,
        stop_timeout_ms=1_000,
        log_rate_limit_window_ms=5_000,
        log_rate_limit_max_per_key=5,
        sink=sink,
    )


def wait_for_packets(
    sink: CapturingSink,
    expected_count: int,
    timeout_seconds: float,
) -> bool:
    """Wait until sink captures expected packet count or timeout occurs."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if sink.count() >= expected_count:
            return True
        time.sleep(WAIT_POLL_SECONDS)
    return sink.count() >= expected_count


def validate_canonical_packet(packet: FramePacket) -> list[str]:
    """Return validation errors for required canonical FramePacket fields."""
    checks = {
        "pixel_format": (packet.pixel_format, "RGB"),
        "layout": (packet.layout, "HWC"),
        "dtype": (packet.dtype, "uint8"),
        "value_range": (packet.value_range, "[0,255]"),
        "num_color_channels": (packet.num_color_channels, 3),
        "bits_per_channel": (packet.bits_per_channel, 8),
        "packing": (packet.packing, "tightly_packed"),
    }
    errors: list[str] = []
    for field_name, (actual, expected) in checks.items():
        if actual != expected:
            errors.append(
                f"canonical_mismatch:{field_name}:expected={expected}:actual={actual}"
            )
    return errors


def reconstruct_rgb_image(packet: FramePacket) -> tuple[np.ndarray | None, str | None]:
    """Reconstruct RGB image array from packet bytes with shape checks."""
    expected_size = packet.width * packet.height * 3
    if len(packet.image_bytes) != expected_size:
        return None, (
            "image_bytes_size_mismatch:"
            f"expected={expected_size}:actual={len(packet.image_bytes)}"
        )
    rgb = np.frombuffer(packet.image_bytes, dtype=np.uint8)
    rgb = rgb.reshape(packet.height, packet.width, 3)
    return rgb, None


def save_rgb_png(rgb_image: np.ndarray, output_path: Path) -> None:
    """Write RGB image as PNG via OpenCV."""
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr)


def fit_into_tile(image_rgb: np.ndarray) -> np.ndarray:
    """Resize and pad an RGB image into fixed tile size for contact sheet."""
    src_h, src_w = image_rgb.shape[:2]
    scale = min(IMAGE_TILE_WIDTH / src_w, IMAGE_TILE_HEIGHT / src_h)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized_rgb = cast(
        np.ndarray,
        cv2.resize(
        image_rgb,
        (resized_w, resized_h),
        interpolation=cv2.INTER_AREA,
        ),
    )
    canvas = np.full(
        (IMAGE_TILE_HEIGHT, IMAGE_TILE_WIDTH, 3),
        TILE_BG,
        dtype=np.uint8,
    )
    start_x = (IMAGE_TILE_WIDTH - resized_w) // 2
    start_y = (IMAGE_TILE_HEIGHT - resized_h) // 2
    canvas[start_y:start_y + resized_h, start_x:start_x + resized_w] = resized_rgb
    return canvas


def render_contact_tile(
    image_rgb: np.ndarray | None,
    label: str,
    status: str,
) -> np.ndarray:
    """Render one contact-sheet tile with label and status text."""
    if image_rgb is None:
        tile_rgb = np.full(
            (IMAGE_TILE_HEIGHT, IMAGE_TILE_WIDTH, 3),
            PLACEHOLDER_BG,
            dtype=np.uint8,
        )
        cv2.putText(
            tile_rgb,
            "missing",
            (14, IMAGE_TILE_HEIGHT // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            TEXT_COLOR_DARK,
            2,
            cv2.LINE_AA,
        )
    else:
        tile_rgb = fit_into_tile(image_rgb)
    label_band = np.full(
        (TILE_LABEL_HEIGHT, IMAGE_TILE_WIDTH, 3),
        (40, 40, 40),
        dtype=np.uint8,
    )
    text = f"{label} | {status}"
    cv2.putText(
        label_band,
        text,
        (8, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        TEXT_COLOR_LIGHT,
        1,
        cv2.LINE_AA,
    )
    full_rgb = np.vstack([label_band, tile_rgb])
    return cv2.cvtColor(full_rgb, cv2.COLOR_RGB2BGR)


def build_contact_sheet(
    asset_originals: dict[str, np.ndarray],
    output_images: dict[tuple[str, str], np.ndarray],
    case_statuses: dict[tuple[str, str], str],
    output_path: Path,
) -> None:
    """Build and save comparison contact sheet for all assets and cases."""
    has_mjpeg = any(key[1] == "MJPEG" for key in case_statuses)
    case_columns = ["RGB", "BGR", "GRAY8", "JPEG"]
    if has_mjpeg:
        case_columns.append("MJPEG")
    rows: list[np.ndarray] = []
    for asset_stem in sorted(asset_originals):
        tile_list: list[np.ndarray] = []
        tile_list.append(
            render_contact_tile(asset_originals[asset_stem], f"{asset_stem}:orig", "ok")
        )
        for case_name in case_columns:
            image = output_images.get((asset_stem, case_name))
            status = case_statuses.get((asset_stem, case_name), "missing")
            tile_list.append(render_contact_tile(image, case_name, status))
        spacer = np.full((tile_list[0].shape[0], COL_GAP, 3), 245, dtype=np.uint8)
        row = tile_list[0]
        for tile in tile_list[1:]:
            row = np.hstack([row, spacer, tile])
        rows.append(row)
    if not rows:
        blank = np.full((120, 300, 3), 240, dtype=np.uint8)
        cv2.putText(
            blank,
            "No assets discovered",
            (16, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(output_path), blank)
        return
    spacer_row = np.full((ROW_GAP, rows[0].shape[1], 3), 245, dtype=np.uint8)
    sheet = rows[0]
    for row in rows[1:]:
        sheet = np.vstack([sheet, spacer_row, row])
    cv2.imwrite(str(output_path), sheet)


def run_visual_smoke() -> RunSummary:
    """Execute visual smoke flow and return final summary object."""
    clean_output_directory(OUTPUT_DIR)
    input_files = discover_asset_files(ASSETS_DIR)
    summary_assets: list[AssetSummary] = []
    all_cases: list[IngressCase] = []
    output_images: dict[tuple[str, str], np.ndarray] = {}
    original_images: dict[str, np.ndarray] = {}
    case_statuses: dict[tuple[str, str], str] = {}
    case_index = 0
    for input_file in input_files:
        asset_summary: AssetSummary = {
            "input_file": str(input_file.relative_to(REPO_ROOT).as_posix()),
            "generated_cases": list(),
            "skipped_cases": list(),
            "output_file_paths": {},
            "cases": list(),
            "errors": list(),
            "width": None,
            "height": None,
        }
        asset = load_asset_record(input_file)
        if asset is None:
            asset_summary["errors"] = ["asset_decode_failed"]
            asset_summary["width"] = None
            asset_summary["height"] = None
            summary_assets.append(asset_summary)
            continue
        original_images[asset.stem] = asset.rgb_image
        asset_summary["width"] = asset.width
        asset_summary["height"] = asset.height
        generated_cases_for_asset, skipped_cases_for_asset = build_ingress_cases(
            asset,
            case_index,
        )
        case_index += len(generated_cases_for_asset)
        asset_summary["generated_cases"] = [
            gen_case.source_format for gen_case in generated_cases_for_asset
        ]
        asset_summary["skipped_cases"] = skipped_cases_for_asset
        summary_assets.append(asset_summary)
        for gen_case in generated_cases_for_asset:
            all_cases.append(gen_case)
            case_statuses[(gen_case.asset_stem, gen_case.source_format)] = "pending"
    sink = CapturingSink()
    transport = InMemoryFrameIngressTransport()
    gateway = FrameIngestionGateway(transport=transport)
    gateway.configure(make_gateway_config(sink=sink))
    gateway.start()
    try:
        for case in all_cases:
            message = IngressFrameMessage(
                frame_id=case.frame_id,
                camera_id=case.camera_id,
                timestamp_ms=case.timestamp_ms,
                width=case.width,
                height=case.height,
                source_format=case.source_format,
                source_layout=case.source_layout,
                source_num_color_channels=None,
                source_bits_per_channel=case.source_bits_per_channel,
                payload_bytes=case.payload_bytes,
            )
            transport.inject_message(message)
        wait_for_packets(
            sink=sink,
            expected_count=len(all_cases),
            timeout_seconds=MAX_WAIT_SECONDS,
        )
    finally:
        gateway.stop()
    packets = sink.snapshot()
    packet_by_frame_id = {packet.frame_id: packet for packet in packets}
    for asset_summary in summary_assets:
        file_path = str(asset_summary["input_file"])
        generated_cases = list(asset_summary["generated_cases"])
        per_case: list[CaseResult] = []
        output_file_paths: dict[str, str] = {}
        for source_format in generated_cases:
            stem = sanitize_stem(Path(file_path).stem)
            frame_id = f"{stem}__{source_format.lower()}"
            case_key = (stem, source_format)
            case_data: CaseResult = {
                "source_format": source_format,
                "frame_id": frame_id,
                "camera_id": CAMERA_ID,
                "width": asset_summary["width"],
                "height": asset_summary["height"],
                "canonical_validation": {"passed": False, "errors": []},
                "output_file": None,
                "errors": [],
            }
            packet = packet_by_frame_id.get(frame_id)
            if packet is None:
                case_data["errors"] = ["packet_not_published"]
                case_statuses[case_key] = "missing"
                per_case.append(case_data)
                continue
            canonical_errors = validate_canonical_packet(packet)
            image_rgb, reconstruct_error = reconstruct_rgb_image(packet)
            errors = list(canonical_errors)
            if reconstruct_error is not None:
                errors.append(reconstruct_error)
            if image_rgb is not None:
                output_file = OUTPUT_DIR / (
                    f"{stem}__{source_format}__gateway_output.png"
                )
                save_rgb_png(image_rgb, output_file)
                output_path_rel = str(output_file.relative_to(REPO_ROOT).as_posix())
                case_data["output_file"] = output_path_rel
                output_file_paths[source_format] = output_path_rel
                output_images[case_key] = image_rgb
            case_data["canonical_validation"] = {
                "passed": len(canonical_errors) == 0,
                "errors": canonical_errors,
            }
            case_data["errors"] = errors
            case_statuses[case_key] = "ok" if len(errors) == 0 else "error"
            per_case.append(case_data)
        asset_summary["cases"] = per_case
        asset_summary["output_file_paths"] = output_file_paths
    for asset_summary in summary_assets:
        generated_set = set(asset_summary["generated_cases"])
        for skipped_case in asset_summary["skipped_cases"]:
            case_key = (
                sanitize_stem(Path(asset_summary["input_file"]).stem),
                skipped_case["case"],
            )
            if skipped_case["case"] not in generated_set:
                case_statuses[case_key] = "skipped"
    build_contact_sheet(
        asset_originals=original_images,
        output_images=output_images,
        case_statuses=case_statuses,
        output_path=CONTACT_SHEET_PATH,
    )
    outputs_saved = sum(
        len(asset_summary["output_file_paths"])
        for asset_summary in summary_assets
    )
    skipped_count = sum(
        len(asset_summary["skipped_cases"])
        for asset_summary in summary_assets
    )
    failure_count = 0
    for asset_summary in summary_assets:
        for case_result in asset_summary["cases"]:
            if case_result["errors"]:
                failure_count += 1
    summary: RunSummary = {
        "input_directory": str(ASSETS_DIR.relative_to(REPO_ROOT).as_posix()),
        "output_directory": str(OUTPUT_DIR.relative_to(REPO_ROOT).as_posix()),
        "contact_sheet": str(CONTACT_SHEET_PATH.relative_to(REPO_ROOT).as_posix()),
        "summary_json": str(SUMMARY_PATH.relative_to(REPO_ROOT).as_posix()),
        "totals": {
            "input_images_found": len(input_files),
            "cases_generated": len(all_cases),
            "outputs_saved": outputs_saved,
            "skipped_cases": skipped_count,
            "failures": failure_count,
            "packets_captured": len(packets),
        },
        "assets": summary_assets,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def print_run_report(summary: RunSummary) -> None:
    """Print compact run report for terminal usage."""
    totals = summary["totals"]
    print("Frame Ingestion Gateway visual smoke test complete")
    print(f"input images found: {totals['input_images_found']}")
    print(f"cases generated: {totals['cases_generated']}")
    print(f"outputs saved: {totals['outputs_saved']}")
    print(f"output directory: {summary['output_directory']}")
    print(f"contact sheet: {summary['contact_sheet']}")
    print(f"summary json: {summary['summary_json']}")
    print(f"failures: {totals['failures']}")
    print(f"skipped cases: {totals['skipped_cases']}")


def main() -> int:
    """Entry point for standalone visual smoke execution."""
    if not ASSETS_DIR.exists():
        print(f"Assets directory not found: {ASSETS_DIR}")
        return 1
    summary = run_visual_smoke()
    print_run_report(summary=summary)
    failures = int(summary["totals"]["failures"])
    return 1 if failures > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())