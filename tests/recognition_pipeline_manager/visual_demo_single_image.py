"""Single-image visual demo for the Recognition Pipeline Manager.

This script loads one demo image, runs the real RecognitionPipelineManager
through the production startup path, and writes one annotated output image.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _PROJECT_ROOT / "src"
_TESTS_DIR = _PROJECT_ROOT / "tests"

for _path in (_PROJECT_ROOT, _SRC_DIR, _TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

_INPUT_IMAGE_PATH = (
    _PROJECT_ROOT / "tests" / "recognition_pipeline_manager" / "images" / "demo_input.png"
)
_OUTPUT_IMAGE_PATH = (
    _PROJECT_ROOT / "tests" / "recognition_pipeline_manager" / "images" / "demo_output.png"
)
_RPM_CONFIG_PATH = _PROJECT_ROOT / "config" / "image_processing_service" / "local_debug.yaml"
_PERSON_JSON_PATH = _PROJECT_ROOT / "data" / "person_directory.json"
_CAMERA_ID = "camera_1"
_FRAME_ID = "demo_single_image_0001"
_TIMESTAMP_MS = 0

_GREEN = (0, 255, 0)
_YELLOW = (255, 255, 0)
_CYAN = (0, 255, 255)
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)


def _get_font() -> Any:
    """Return the small default PIL font used by the demo overlays."""
    from PIL import ImageFont

    return ImageFont.load_default()


def _load_rgb_image(image_path: Path) -> np.ndarray:
    """Load one RGB image as a uint8 HWC ndarray."""
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as opened_image:
        return np.array(opened_image.convert("RGB"))


def _build_frame_packet(image: np.ndarray) -> Any:
    """Convert one RGB ndarray into the canonical FramePacket contract."""
    from image_processing.frame_transformation_layer.contracts import FramePacket

    height, width = image.shape[:2]
    return FramePacket(
        frame_id=_FRAME_ID,
        camera_id=_CAMERA_ID,
        timestamp_ms=_TIMESTAMP_MS,
        width=width,
        height=height,
        pixel_format="RGB",
        layout="HWC",
        dtype="uint8",
        value_range="[0,255]",
        num_color_channels=3,
        bits_per_channel=8,
        packing="tightly_packed",
        image_bytes=image.tobytes(),
    )


def _draw_text_box(
    draw: Any,
    image_size: tuple[int, int],
    text: str,
    origin: tuple[int, int],
    text_color: tuple[int, int, int],
    background_color: tuple[int, int, int] = _BLACK,
) -> None:
    """Draw a text label with a solid backing rectangle."""
    if not text:
        return
    x, y = origin
    pad = 4
    font = _get_font()
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    top_left = (max(0, x - pad), max(0, y - text_height - pad))
    bottom_right = (
        min(image_size[0] - 1, x + text_width + pad),
        min(image_size[1] - 1, y + pad),
    )
    draw.rectangle([top_left, bottom_right], fill=background_color)
    draw.text((x, y - text_height), text, fill=text_color, font=font)


def _draw_legend(image: Image.Image) -> None:
    """Draw a small legend in the top-left corner."""
    from PIL import ImageDraw

    lines = [
        ("Person Detection", _GREEN),
        ("Face Detection", _YELLOW),
        ("Face Recognition", _CYAN),
    ]
    draw = ImageDraw.Draw(image)
    panel_x = 12
    panel_y = 14
    line_height = 22
    panel_width = 190
    panel_height = 18 + (line_height * len(lines))
    draw.rectangle(
        [(panel_x - 8, panel_y - 18), (panel_x + panel_width, panel_y + panel_height)],
        fill=(0, 0, 0),
        outline=(90, 90, 90),
        width=1,
    )
    for index, (label, color) in enumerate(lines):
        y = panel_y + (index * line_height)
        draw.rectangle([(panel_x, y - 10), (panel_x + 14, y + 2)], fill=color)
        draw.text((panel_x + 22, y - 10), label, fill=_WHITE, font=_get_font())


def _extract_identity_label(face: dict[str, Any]) -> str:
    """Build the visible identity label for one recognized face."""
    person_name = str(face.get("person_name", "")).strip()
    if not person_name or person_name == "UNKNOWN":
        person_name = str(face.get("person_id", "UNKNOWN")).strip() or "UNKNOWN"
    confidence = face.get("confidence")
    if confidence is None:
        return person_name
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        return person_name
    return f"{person_name} ({confidence_value:.2f})"


def _draw_person_box(draw: Any, bbox: dict[str, Any]) -> None:
    """Draw one green person bounding box."""
    x = int(bbox["x"])
    y = int(bbox["y"])
    width = int(bbox["width"])
    height = int(bbox["height"])
    draw.rectangle([(x, y), (x + width, y + height)], outline=_GREEN, width=3)


def _draw_face_box_and_label(draw: Any, image_size: tuple[int, int], face: dict[str, Any]) -> None:
    """Draw one yellow face box and a cyan identity label when present."""
    bbox = face.get("face_bbox")
    if not bbox:
        return
    x = int(bbox["x"])
    y = int(bbox["y"])
    width = int(bbox["width"])
    height = int(bbox["height"])
    draw.rectangle([(x, y), (x + width, y + height)], outline=_YELLOW, width=2)

    label = _extract_identity_label(face)
    if not label:
        return
    label_y = y - 6 if y >= 24 else y + height + 18
    _draw_text_box(
        draw,
        image_size,
        label,
        (x, label_y),
        text_color=_CYAN,
        background_color=_BLACK,
    )


def _annotate_output(image: np.ndarray, output: dict[str, Any]) -> np.ndarray:
    """Render persons, faces, labels, and legend onto one RGB image."""
    import numpy as np
    from PIL import Image, ImageDraw

    annotated_image = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(annotated_image)
    for person in output.get("persons", []):
        person_bbox = person.get("person_bbox")
        if person_bbox:
            _draw_person_box(draw, person_bbox)
        for face in person.get("recognized_faces", []):
            _draw_face_box_and_label(draw, annotated_image.size, face)
    _draw_legend(annotated_image)
    return np.array(annotated_image)


def _summarize_recognized_identities(output: dict[str, Any]) -> list[str]:
    """Collect unique recognized identity labels in first-seen order."""
    identities: list[str] = []
    seen: set[str] = set()
    for person in output.get("persons", []):
        for face in person.get("recognized_faces", []):
            label = _extract_identity_label(face)
            if label in seen:
                continue
            seen.add(label)
            identities.append(label)
    return identities


def _resolve_rpm_config_path(argv: list[str]) -> Path:
    """Resolve the RPM config path from CLI arguments or fallback default."""
    if not argv:
        return _RPM_CONFIG_PATH
    provided = Path(argv[0])
    if provided.is_absolute():
        return provided
    return _PROJECT_ROOT / provided


def main() -> int:
    """Run the single-image RPM demo and write the annotated output image."""
    argv = list(sys.argv[1:])

    try:
        from PIL import Image
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "This demo requires the project runtime dependencies to be installed, "
            "including numpy and Pillow."
        ) from exc

    if not _INPUT_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Demo input image not found: {_INPUT_IMAGE_PATH}")

    try:
        from scripts.run_rpm_on_frames import build_runtime_components, load_yaml_config
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Unable to import the RPM runtime helpers. Ensure the repository "
            "dependencies are installed and the project root is on PYTHONPATH."
        ) from exc

    rpm_config_path = _resolve_rpm_config_path(argv)
    if not rpm_config_path.exists():
        raise FileNotFoundError(f"RPM config not found: {rpm_config_path}")
    rpm_config = load_yaml_config(rpm_config_path)
    print(f"Using RPM config: {rpm_config_path}")
    rgb_image = _load_rgb_image(_INPUT_IMAGE_PATH)
    frame_packet = _build_frame_packet(rgb_image)

    person_json_path = _PERSON_JSON_PATH
    pd_path_str = rpm_config.get("recognition_pipeline", {}).get("person_directory_json_path")
    if pd_path_str:
        candidate = _PROJECT_ROOT / pd_path_str
        if candidate.exists():
            person_json_path = candidate

    components = build_runtime_components(
        rpm_config=rpm_config,
        person_json=person_json_path,
        frame_width=rgb_image.shape[1],
        frame_height=rgb_image.shape[0],
        real_motion_detection=True,
        real_object_detection=True,
        real_face_detection=True,
        real_face_recognition=True,
        demo_bypass_motion_gate=True,
    )

    # Warmup call: FTL requires a previous frame before it can serve motion detection.
    # The first call always returns cold_start; discard it and flush all logging wrappers
    # so their call lists start clean before the real demo pass.
    print("Warming up pipeline (frame 1 is cold-start, discarding)...")
    components.rpm.process_frame(frame_packet)
    components.motion.flush_calls()
    components.obj_det.flush_calls()
    components.face_det.flush_calls()
    components.face_rec.flush_calls()
    print("Warmup complete. Running real demo pass...")

    rpm_output = components.rpm.process_frame(frame_packet)

    # Collect stage execution evidence from real runtime metrics and logging wrappers.
    frame_metrics = components.rpm.get_last_frame_metrics()
    components.motion.flush_calls()
    components.obj_det.flush_calls()
    components.face_det.flush_calls()
    components.face_rec.flush_calls()

    stop_reason = frame_metrics.get("stop_reason", "unknown")
    motion_detected = frame_metrics.get("motion_detected", False)
    bypass_used = frame_metrics.get("demo_bypass_motion_gate", False)
    od_calls = int(frame_metrics.get("object_detection_call_count", 0))
    fd_calls = int(frame_metrics.get("face_detection_call_count", 0))
    fr_calls = int(frame_metrics.get("face_recognition_call_count", 0))

    print("")
    print("[STAGE EXECUTION]")
    print(f"  motion_detection : bypassed  (motion_detected={motion_detected}, demo_bypass={bypass_used})")
    print(f"  object_detection : {'executed' if od_calls > 0 else 'skipped '}  ({od_calls} calls)")
    print(f"  face_detection   : {'executed' if fd_calls > 0 else 'skipped '}  ({fd_calls} calls)")
    print(f"  face_recognition : {'executed' if fr_calls > 0 else 'skipped '}  ({fr_calls} calls)")
    print(f"  stop_reason      : {stop_reason}")
    print("")

    annotated = _annotate_output(rgb_image, rpm_output)

    _OUTPUT_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(annotated).save(_OUTPUT_IMAGE_PATH)

    persons = rpm_output.get("persons", [])
    total_persons = len(persons)
    total_faces = sum(len(person.get("recognized_faces", [])) for person in persons)
    identities = _summarize_recognized_identities(rpm_output)

    print(f"Number of persons detected: {total_persons}")
    print(f"Number of faces detected: {total_faces}")
    print("Recognized identities:")
    if identities:
        for identity in identities:
            print(f"  - {identity}")
    else:
        print("  - none")
    print(f"Saved annotated image to: {_OUTPUT_IMAGE_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())