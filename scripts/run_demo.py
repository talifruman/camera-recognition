from pathlib import Path
import cv2


INPUT_VIDEO = Path("assets/demo_annotated.mp4")
OUTPUT_DIR = Path("debug_outputs")
OUTPUT_VIDEO = OUTPUT_DIR / "docker_demo_output.mp4"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO}")

    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {INPUT_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.putText(
            frame,
            "Docker demo is running",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        frame_count += 1

    cap.release()
    writer.release()

    print("Demo completed successfully.")
    print(f"Input: {INPUT_VIDEO}")
    print(f"Output: {OUTPUT_VIDEO}")
    print(f"Processed frames: {frame_count}")


if __name__ == "__main__":
    main()
