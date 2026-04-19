from pathlib import Path
import cv2
import json

VIDEO_ROOT = Path(r"C:\Users\talif\Desktop\videos")
FRAMES_ROOT = Path("tests/motion_detection/assets/frames")

FRAME_STEP = 3
MAX_CASES_PER_VIDEO = 20
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def is_video_file(path: Path):
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def extract_pairs_from_video(video_path, output_dir):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return 0

    frames = []
    index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append((index, gray))
        index += 1

    cap.release()

    if len(frames) <= FRAME_STEP:
        return 0

    created = 0

    for i in range(0, len(frames) - FRAME_STEP, FRAME_STEP):
        if created >= MAX_CASES_PER_VIDEO:
            break

        prev_idx, prev_img = frames[i]
        curr_idx, curr_img = frames[i + FRAME_STEP]

        case_dir = output_dir / f"{video_path.stem}_case_{created+1:03d}"
        ensure_dir(case_dir)

        cv2.imwrite(str(case_dir / "previous.png"), prev_img)
        cv2.imwrite(str(case_dir / "current.png"), curr_img)

        meta = {
            "video": video_path.name,
            "previous_frame_index": prev_idx,
            "current_frame_index": curr_idx,
            "frame_gap": FRAME_STEP,
            "expected_motion": output_dir.name != "no_motion"
        }

        with open(case_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        created += 1

    return created


def main():
    ensure_dir(FRAMES_ROOT)

    total = 0

    for category in VIDEO_ROOT.iterdir():
        if not category.is_dir():
            continue

        print(f"Processing category: {category.name}")

        out_dir = FRAMES_ROOT / category.name
        ensure_dir(out_dir)

        for video in category.iterdir():
            if not is_video_file(video):
                continue

            count = extract_pairs_from_video(video, out_dir)
            print(f"{video.name}: {count} pairs")

            total += count

    print(f"\nDone. Total pairs: {total}")


if __name__ == "__main__":
    main()
