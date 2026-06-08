from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_processing.shared.contracts import Image  # type: ignore[import-not-found]
from image_processing.motion_detection import (  # type: ignore[import-not-found]
    MotionDetectionInput,
    MotionDetectionManager,
    MotionInputFrame,
    MotionResult,
)

FRAMES_ROOT = Path(__file__).parent / "assets" / "frames"
OUTPUTS_ROOT = Path(__file__).parent / "outputs"

CATEGORIES = ("no_motion", "small_motion", "clear_motion")


# ---------------------------------------------------------------------------
# Asset discovery
# ---------------------------------------------------------------------------


def iter_cases(
    frames_root: Path,
) -> Generator[tuple[str, str, Path], None, None]:
    """Yield (category, case_name, case_path) for every case folder found."""
    for category in CATEGORIES:
        cat_path = frames_root / category
        if not cat_path.is_dir():
            continue
        for case_path in sorted(cat_path.iterdir()):
            if case_path.is_dir():
                yield category, case_path.name, case_path


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------


def load_frame_pair(
    case_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load previous.png, current.png, and optional meta.json from a case folder."""
    prev_path = case_path / "previous.png"
    curr_path = case_path / "current.png"

    prev_img = cv2.imread(str(prev_path), cv2.IMREAD_GRAYSCALE)
    curr_img = cv2.imread(str(curr_path), cv2.IMREAD_GRAYSCALE)

    if prev_img is None:
        raise FileNotFoundError(f"Could not load: {prev_path}")
    if curr_img is None:
        raise FileNotFoundError(f"Could not load: {curr_path}")

    meta: dict = {}
    meta_path = case_path / "meta.json"
    if meta_path.exists():
        with meta_path.open() as f:
            meta = json.load(f)

    return prev_img, curr_img, meta


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------


def _ndarray_to_image(arr: np.ndarray) -> Image:
    """Wrap a 2-D or (H,W,1) uint8 grayscale ndarray in a shared Image struct."""
    if arr.ndim == 3:
        h, w = arr.shape[0], arr.shape[1]
    else:
        h, w = arr.shape[0], arr.shape[1]
    return Image(
        data=arr,
        width=w,
        height=h,
        color_format="GRAY",
        layout="HWC",
        dtype="uint8",
        value_range="[0,255]",
    )


def build_motion_input(
    prev_img: np.ndarray,
    curr_img: np.ndarray,
    case_name: str,
) -> MotionDetectionInput:
    """Construct a MotionDetectionInput from a grayscale frame pair."""
    camera_id = case_name

    previous_frame: MotionInputFrame = {
        "frame_id": "frame_0001",
        "camera_id": camera_id,
        "timestamp_ms": 1000,
        "image": _ndarray_to_image(prev_img),
    }
    current_frame: MotionInputFrame = {
        "frame_id": "frame_0002",
        "camera_id": camera_id,
        "timestamp_ms": 2000,
        "image": _ndarray_to_image(curr_img),
    }
    motion_input: MotionDetectionInput = {
        "previous_frame": previous_frame,
        "current_frame": current_frame,
    }
    return motion_input


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def draw_result(
    curr_img: np.ndarray,
    result: MotionResult,
    case_name: str,
) -> np.ndarray:
    """Return a BGR visualization of the motion detection result on the current frame."""
    # Convert grayscale to BGR for drawing
    if len(curr_img.shape) == 2:
        vis = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2BGR)
    else:
        vis = curr_img.copy()

    detected: bool = result["detected"]
    bboxes = result["bboxes"]

    # Draw each bounding box
    for bbox in bboxes:
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

    # Draw detection label at top-left
    if detected:
        label = "MOTION"
        label_color = (0, 255, 0)
    else:
        label = "NO MOTION"
        label_color = (0, 0, 255)

    cv2.putText(
        vis,
        label,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        label_color,
        2,
        cv2.LINE_AA,
    )

    # Draw case name at bottom-left
    h_img = vis.shape[0]
    cv2.putText(
        vis,
        case_name,
        (8, h_img - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return vis


def build_debug_image(
    prev_img: np.ndarray,
    curr_img: np.ndarray,
    vis_img: np.ndarray,
) -> np.ndarray:
    """Return a side-by-side debug image: previous | current | annotated."""
    def to_bgr(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img.copy()

    prev_bgr = to_bgr(prev_img)
    curr_bgr = to_bgr(curr_img)

    # Ensure all panels have identical height before stacking
    target_h = vis_img.shape[0]
    panels = []
    for panel in (prev_bgr, curr_bgr, vis_img):
        if panel.shape[0] != target_h:
            scale = target_h / panel.shape[0]
            new_w = int(panel.shape[1] * scale)
            panel = cv2.resize(panel, (new_w, target_h), interpolation=cv2.INTER_AREA)
        panels.append(panel)

    return np.hstack(panels)


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------


def save_output(
    output_root: Path,
    category: str,
    case_name: str,
    vis_img: np.ndarray,
    debug_img: np.ndarray | None = None,
) -> None:
    cat_out = output_root / category
    cat_out.mkdir(parents=True, exist_ok=True)

    out_path = cat_out / f"{case_name}.png"
    cv2.imwrite(str(out_path), vis_img)

    if debug_img is not None:
        debug_path = cat_out / f"{case_name}_debug.png"
        cv2.imwrite(str(debug_path), debug_img)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    manager = MotionDetectionManager()

    total = 0
    detected_count = 0
    not_detected_count = 0
    failed_count = 0

    for category, case_name, case_path in iter_cases(FRAMES_ROOT):
        total += 1
        try:
            prev_img, curr_img, _meta = load_frame_pair(case_path)
            motion_input = build_motion_input(prev_img, curr_img, case_name)
            result: MotionResult = manager.process(motion_input)

            vis_img = draw_result(curr_img, result, case_name)
            debug_img = build_debug_image(prev_img, curr_img, vis_img)
            save_output(OUTPUTS_ROOT, category, case_name, vis_img, debug_img)

            status = "DETECTED" if result["detected"] else "NO MOTION"
            bbox_count = len(result["bboxes"])
            print(f"  [{category}] {case_name}: {status} ({bbox_count} bbox(s))")

            if result["detected"]:
                detected_count += 1
            else:
                not_detected_count += 1

        except Exception as exc:  # noqa: BLE001
            failed_count += 1
            print(f"  ERROR [{category}] {case_name}: {exc}")

    print()
    print("=" * 60)
    print("Motion Detection Visual Test — Summary")
    print("=" * 60)
    print(f"  Cases processed : {total}")
    print(f"  Detected        : {detected_count}")
    print(f"  Not detected    : {not_detected_count}")
    print(f"  Errors          : {failed_count}")
    print(f"  Outputs saved to: {OUTPUTS_ROOT.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
