import os
import glob
import argparse
import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_dir", required=True)
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--model_size", default="n", choices=["n", "s", "m", "l", "x"],
                        help="YOLO model size: n=nano, s=small, m=medium, l=large, x=xlarge")
    args = parser.parse_args()

    img_dir = os.path.join(args.seq_dir, "img1")

    # Support both jpg and png
    frames = sorted(
        glob.glob(os.path.join(img_dir, "*.jpg")) +
        glob.glob(os.path.join(img_dir, "*.png"))
    )

    if len(frames) == 0:
        print("ERROR: no frames found in:", img_dir)
        return

    print("img_dir:", img_dir)
    print("num_frames:", len(frames))

    img = cv2.imread(frames[0])
    if img is None:
        print("ERROR: could not read first image:", frames[0])
        return

    # Load model with specified size
    model_name = f"yolov8{args.model_size}.pt"
    model = YOLO(model_name)

    results = model(img, classes=[0], conf=0.3)
    
    print(f"Model: {model_name}")

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"person {conf:.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    seq_name = os.path.basename(args.seq_dir.rstrip("\\/"))
    out_seq_dir = os.path.join(args.out_dir, f"{seq_name}_yolov8{args.model_size}")
    os.makedirs(out_seq_dir, exist_ok=True)

    out_path = os.path.join(out_seq_dir, "det_example.jpg")
    cv2.imwrite(out_path, img)

    print(f"Saved detection image to: {out_path}")


if __name__ == "__main__":
    main()
