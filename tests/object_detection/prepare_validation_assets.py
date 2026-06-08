from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, TypedDict

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

ROOT_DIR = Path(__file__).resolve().parents[2]
ASSETS_DIR = ROOT_DIR / "tests" / "object_detection" / "assets"
SEARCH_ENDPOINT = "https://api.openverse.org/v1/images/"
REQUEST_HEADERS = {
    "User-Agent": "camera-recognition-validation-assets/1.0",
}
PAGE_SIZE = 20
MAX_PAGES_PER_QUERY = 5
DOWNLOAD_TIMEOUT_SECONDS = 20
MIN_IMAGE_SIDE = 320
MAX_ASPECT_RATIO = 3.0
MIN_DOWNLOAD_BYTES = 2048
MIN_FACE_SIDE = 40
MIN_FACE_AREA_RATIO = 0.008
MIN_FACE_SHARPNESS = 60.0
MIN_FACE_BRIGHTNESS = 45.0
MAX_FACE_BRIGHTNESS = 215.0
MIN_PERSON_HEIGHT_RATIO = 0.25
MIN_PERSON_AREA_RATIO = 0.06

HARD_REJECT_KEYWORDS = {
    "actor",
    "actress",
    "advertisement",
    "ai",
    "anime",
    "artwork",
    "billboard",
    "cartoon",
    "celebrity",
    "cgi",
    "comic",
    "dall-e",
    "display",
    "doll",
    "drawing",
    "famous",
    "generated",
    "hologram",
    "illustration",
    "mannequin",
    "marble",
    "midjourney",
    "monitor",
    "painting",
    "poster",
    "printed",
    "reflection",
    "render",
    "rendered",
    "screen",
    "screenshot",
    "sculpture",
    "singer",
    "stable diffusion",
    "statue",
    "television",
    "toy",
    "tv",
    "virtual",
    "wax",
}

MULTIPLE_HINTS = {
    "audience",
    "couple",
    "crowd",
    "family",
    "festival",
    "friends",
    "group",
    "groups",
    "many",
    "multiple",
    "pair",
    "parade",
    "people",
    "team",
    "three",
    "two",
}
ONE_HINTS = {
    "alone",
    "girl",
    "individual",
    "man",
    "one",
    "pedestrian",
    "person",
    "runner",
    "single",
    "walker",
    "walking",
    "woman",
}
NO_PERSON_HINTS = {
    "building",
    "corridor",
    "empty",
    "hallway",
    "interior",
    "office",
    "road",
    "room",
    "street",
    "vacant",
}

_PERSON_DETECTOR: cv2.HOGDescriptor | None = None
_FACE_CASCADE: cv2.CascadeClassifier | None = None


class TargetConfig(TypedDict):
    count: int
    queries: list[str]


TARGETS: dict[str, TargetConfig] = {
    "one_person": {
        "count": 4,
        "queries": [
            "single person standing full body",
            "person walking alone",
            "single person outdoors",
            "one person sidewalk",
            "single person looking at camera",
            "one person face visible",
        ],
    },
    "multiple_people": {
        "count": 3,
        "queries": [
            "two people standing",
            "group of people walking",
            "people on sidewalk",
            "crowd crossing street",
            "two people looking at camera",
            "group of people looking at camera",
            "family portrait outdoors",
            "friends standing together looking at camera",
        ],
    },
    "no_person": {
        "count": 3,
        "queries": [
            "empty room",
            "empty street",
            "empty hallway",
            "empty office interior",
        ],
    },
}


def reset_asset_dirs() -> None:
    if ASSETS_DIR.exists():
        shutil.rmtree(ASSETS_DIR)

    for category in TARGETS:
        (ASSETS_DIR / category).mkdir(parents=True, exist_ok=True)


def build_request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers=REQUEST_HEADERS)


def get_metadata_text(result: dict[str, Any]) -> str:
    parts = [
        str(result.get("title", "")),
        str(result.get("description", "")),
        str(result.get("creator", "")),
        str(result.get("attribution", "")),
    ]

    for tag in result.get("tags", []):
        parts.append(str(tag.get("name", "")))

    return " ".join(parts).lower()


def should_reject_result(result: dict[str, Any]) -> bool:
    metadata_text = get_metadata_text(result)
    tokens = set(re.findall(r"[a-z0-9]+", metadata_text))
    for keyword in HARD_REJECT_KEYWORDS:
        keyword_tokens = keyword.split()
        if len(keyword_tokens) == 1:
            if keyword_tokens[0] in tokens:
                return True
            continue

        if keyword in metadata_text:
            return True

    return False


def search_results(query: str):
    for page in range(1, MAX_PAGES_PER_QUERY + 1):
        params = urllib.parse.urlencode(
            {
                "q": query,
                "page": page,
                "page_size": PAGE_SIZE,
                "mature": "false",
            }
        )
        request = build_request(f"{SEARCH_ENDPOINT}?{params}")

        try:
            with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            break

        for result in payload.get("results", []):
            image_url = result.get("url")
            if image_url:
                yield result

        if page >= payload.get("page_count", 0):
            break


def download_image(url: str) -> bytes | None:
    request = build_request(url)

    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            content_type = response.headers.get("Content-Type", "")
            if content_type and "image" not in content_type.lower():
                return None

            data = response.read()
    except Exception:
        return None

    if len(data) < MIN_DOWNLOAD_BYTES:
        return None

    return data


def decode_image_bytes(image_bytes: bytes) -> np.ndarray | None:
    try:
        with Image.open(io.BytesIO(image_bytes)) as verify_image:
            verify_image.verify()

        with Image.open(io.BytesIO(image_bytes)) as pil_image:
            pil_image.load()
            rgb_image = pil_image.convert("RGB")
            width, height = rgb_image.size

            if min(width, height) < MIN_IMAGE_SIDE:
                return None

            aspect_ratio = max(width, height) / max(1, min(width, height))
            if aspect_ratio > MAX_ASPECT_RATIO:
                return None

            image_array = np.array(rgb_image, dtype=np.uint8)
    except (OSError, ValueError, UnidentifiedImageError):
        return None

    return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)


def get_person_detector() -> cv2.HOGDescriptor:
    global _PERSON_DETECTOR

    if _PERSON_DETECTOR is None:
        detector = cv2.HOGDescriptor()
        default_people_detector = getattr(cv2, "HOGDescriptor_getDefaultPeopleDetector")
        detector.setSVMDetector(default_people_detector())
        _PERSON_DETECTOR = detector

    return _PERSON_DETECTOR


def get_face_cascade() -> cv2.CascadeClassifier | None:
    global _FACE_CASCADE

    if _FACE_CASCADE is None:
        cascade_root = getattr(getattr(cv2, "data"), "haarcascades")
        cascade_path = cascade_root + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return None
        _FACE_CASCADE = cascade

    return _FACE_CASCADE


def resize_for_detection(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= 1280:
        return image

    scale = 1280 / longest_side
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    return cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def flatten_indices(selected: Any) -> list[int]:
    if selected is None or len(selected) == 0:
        return []

    indices: list[int] = []
    for item in selected:
        if isinstance(item, (list, tuple)):
            indices.append(int(item[0]))
        elif isinstance(item, np.ndarray):
            indices.append(int(item.flatten()[0]))
        else:
            indices.append(int(item))
    return indices


def detect_people(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    detector = get_person_detector()
    resized = resize_for_detection(image)
    rects, weights = detector.detectMultiScale(
        resized,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )

    if len(rects) == 0:
        return []

    scale_x = image.shape[1] / resized.shape[1]
    scale_y = image.shape[0] / resized.shape[0]
    image_area = image.shape[0] * image.shape[1]
    boxes: list[list[int]] = []
    scores: list[float] = []

    for (x, y, width, height), weight in zip(rects, weights):
        score = float(weight)
        if score < 0.4:
            continue

        scaled_x = int(x * scale_x)
        scaled_y = int(y * scale_y)
        scaled_width = int(width * scale_x)
        scaled_height = int(height * scale_y)

        if scaled_height / image.shape[0] < MIN_PERSON_HEIGHT_RATIO:
            continue

        if (scaled_width * scaled_height) / image_area < MIN_PERSON_AREA_RATIO:
            continue

        boxes.append([scaled_x, scaled_y, scaled_width, scaled_height])
        scores.append(score)

    if not boxes:
        return []

    selected = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.4, nms_threshold=0.35)
    indices = flatten_indices(selected)
    return [
        (boxes[index][0], boxes[index][1], boxes[index][2], boxes[index][3])
        for index in indices
    ]


def detect_faces(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    cascade = get_face_cascade()
    if cascade is None:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    raw_faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(MIN_FACE_SIDE, MIN_FACE_SIDE),
    )

    if len(raw_faces) == 0:
        return []

    image_area = image.shape[0] * image.shape[1]
    valid_faces: list[tuple[int, int, int, int]] = []

    for x, y, width, height in raw_faces:
        face_area_ratio = (width * height) / image_area
        if face_area_ratio < MIN_FACE_AREA_RATIO:
            continue

        aspect_ratio = width / max(1, height)
        if not 0.6 <= aspect_ratio <= 1.6:
            continue

        roi = gray[y : y + height, x : x + width]
        if roi.size == 0:
            continue

        sharpness = cv2.Laplacian(roi, cv2.CV_64F).var()
        brightness = float(roi.mean())

        if sharpness < MIN_FACE_SHARPNESS:
            continue

        if brightness < MIN_FACE_BRIGHTNESS or brightness > MAX_FACE_BRIGHTNESS:
            continue

        valid_faces.append((int(x), int(y), int(width), int(height)))

    return valid_faces


def validate_image(image_bytes: bytes, category: str) -> bool:
    image = decode_image_bytes(image_bytes)
    if image is None:
        return False

    faces = detect_faces(image)
    people = detect_people(image)

    if category == "no_person":
        return len(faces) == 0 and len(people) == 0

    if category == "one_person":
        return len(faces) == 1

    if category == "multiple_people":
        return len(faces) >= 2

    return False


def metadata_hint(result: dict[str, Any], query: str) -> str | None:
    tokens = set(re.findall(r"[a-z]+", query.lower()))
    tokens.update(re.findall(r"[a-z]+", get_metadata_text(result)))

    if tokens & MULTIPLE_HINTS:
        return "multiple_people"
    if tokens & ONE_HINTS:
        return "one_person"
    if tokens & NO_PERSON_HINTS:
        return "no_person"
    return None


def classify_image(result: dict[str, Any], query: str, image: np.ndarray) -> str | None:
    faces = detect_faces(image)
    people = detect_people(image)
    hinted_category = metadata_hint(result, query)

    if len(faces) >= 2:
        return "multiple_people"

    if len(faces) == 1:
        return "one_person"

    if len(faces) == 0 and len(people) == 0:
        return "no_person"

    if len(faces) >= 1 and hinted_category == "multiple_people":
        return "multiple_people"

    if len(faces) == 1 and hinted_category == "one_person":
        return "one_person"

    return None


def save_image(category: str, index: int, image: np.ndarray) -> Path:
    output_path = ASSETS_DIR / category / f"{category}_{index:02d}.jpg"
    if not cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"Failed to write image: {output_path}")
    return output_path


def target_complete(saved_counts: dict[str, int]) -> bool:
    return all(saved_counts[name] >= TARGETS[name]["count"] for name in TARGETS)


def main() -> int:
    reset_asset_dirs()

    saved_counts = {name: 0 for name in TARGETS}
    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()

    for expected_category, config in TARGETS.items():
        for query in config["queries"]:
            if saved_counts[expected_category] >= config["count"]:
                break

            for result in search_results(query):
                if saved_counts[expected_category] >= config["count"]:
                    break

                image_url = result.get("url")
                if not image_url:
                    continue

                if should_reject_result(result):
                    continue

                if image_url in seen_urls:
                    continue

                seen_urls.add(image_url)
                image_bytes = download_image(image_url)
                if image_bytes is None:
                    continue

                image_hash = hashlib.sha256(image_bytes).hexdigest()
                if image_hash in seen_hashes:
                    continue

                if not validate_image(image_bytes, expected_category):
                    continue

                image = decode_image_bytes(image_bytes)
                if image is None:
                    continue

                actual_category = classify_image(result, query, image)
                if actual_category != expected_category:
                    continue

                seen_hashes.add(image_hash)
                saved_counts[actual_category] += 1
                output_path = save_image(actual_category, saved_counts[actual_category], image)
                print(f"saved {actual_category}: {output_path.name} <- {image_url}")

                if target_complete(saved_counts):
                    summary = ", ".join(f"{name}={saved_counts[name]}" for name in TARGETS)
                    print(f"complete: {summary}")
                    return 0

    missing = [
        f"{name}={TARGETS[name]['count'] - saved_counts[name]}"
        for name in TARGETS
        if saved_counts[name] < TARGETS[name]["count"]
    ]
    print("incomplete dataset: " + ", ".join(missing), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())