"""
inference.py — run detection on a single image, a folder of images, or a
video, and report average FPS (for the paper's real-time deployment claims).

Usage:
    # single image or folder of images
    python inference.py --weights best.pt --source path/to/images --imgsz 640 --conf 0.25

    # video (e.g. dashcam footage)
    python inference.py --weights best.pt --source path/to/video.mp4
"""

import argparse
import time
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="Image, folder, or video path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--save_dir", default="runs/inference")
    args = parser.parse_args()

    model = YOLO(args.weights)

    start = time.time()
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        save=True,
        project=args.save_dir,
        name="predictions",
    )
    elapsed = time.time() - start

    n_frames = len(results)
    fps = n_frames / elapsed if elapsed > 0 else 0.0

    print(f"\nProcessed {n_frames} frame(s)/image(s) in {elapsed:.2f}s -> {fps:.2f} FPS")
    print(f"Annotated outputs saved under: {args.save_dir}/predictions")

    # per-image detection summary
    for r in results:
        img_name = Path(r.path).name
        counts = {}
        for cls_id in r.boxes.cls.tolist():
            name = model.names[int(cls_id)]
            counts[name] = counts.get(name, 0) + 1
        print(f"  {img_name}: {counts if counts else 'no damage detected'}")


if __name__ == "__main__":
    main()
