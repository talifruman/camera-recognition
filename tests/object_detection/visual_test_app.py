from __future__ import annotations

from itertools import count
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from object_detection_stub import BoundingBox, FrameMetadata, process

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FRAME_COUNTER = count(1)


def iter_image_files(input_root: Path):
    for path in sorted(input_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def draw_person_boxes(image: Image.Image, persons: list[BoundingBox]) -> Image.Image:
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()

    for bbox in persons:
        left = bbox["x"]
        top = bbox["y"]
        right = left + bbox["width"]
        bottom = top + bbox["height"]
        draw.rectangle((left, top, right, bottom), outline=(255, 255, 0), width=4)
        label_bottom = min(image.height - 1, top + 12)
        draw.rectangle((left, top, min(image.width - 1, left + 42), label_bottom), fill=(255, 255, 0))
        draw.text((left + 2, top), "person", fill=(0, 0, 0), font=font)

    return annotated_image


def process_images(input_root: Path, output_root: Path) -> int:
    processed_count = 0
    input_root = Path(input_root)
    output_root = Path(output_root)

    for image_path in iter_image_files(input_root):
        try:
            with Image.open(image_path) as opened_image:
                rgb_image = opened_image.convert("RGB")
                width, height = rgb_image.size
                metadata: FrameMetadata = {
                    "camera_id": "visual-test-camera",
                    "frame_id": next(FRAME_COUNTER),
                    "width": width,
                    "height": height,
                }
                result = process(np.array(rgb_image), metadata)
                rendered_image = draw_person_boxes(rgb_image, result["persons"])
        except (OSError, ValueError, UnidentifiedImageError):
            continue

        output_path = output_root / image_path.relative_to(input_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rendered_image.save(output_path)
        processed_count += 1

    return processed_count


def main() -> int:
    input_root = Path("tests/object_detection/assets")
    output_root = Path("tests/object_detection/rendered_assets")
    process_images(input_root, output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())