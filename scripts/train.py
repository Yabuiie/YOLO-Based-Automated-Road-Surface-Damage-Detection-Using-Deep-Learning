"""
train.py — fine-tune a YOLO model on RDD2022.

Default base model is yolo11s.pt (Ultralytics YOLO11). YOLO11 uses the same
training/inference API as YOLOv8 (same `ultralytics.YOLO` class, same
COCO-pretrained transfer-learning workflow the paper describes) but its
backbone (C3k2 blocks) and neck (C2PSA partial self-attention block) give a
consistently better accuracy/parameter-count and accuracy/speed tradeoff
than YOLOv8 at the same size (s/m) — the same kind of gain the "attention
enhanced YOLOv8" papers in Related Work report, without a custom
architecture. If you need strict YOLOv8 parity (e.g. to match a cited
YOLOv8 baseline number), pass --model yolov8s.pt or yolov8m.pt instead —
nothing else about this script changes.

Usage:
    python train.py --data ../rdd2022.yaml --model yolo11s.pt --epochs 100 --imgsz 640 --batch 16

Run `yolo settings` once beforehand if you want to disable analytics/telemetry.
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../rdd2022.yaml", help="Path to dataset yaml")
    parser.add_argument("--model", default="yolo11s.pt",
                         help="Base checkpoint: yolo11n/s/m/l.pt (recommended) or "
                              "yolov8n/s/m/l.pt for strict YOLOv8 parity")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--project", default="runs/rdd2022")
    parser.add_argument("--name", default="yolo_rdd2022")
    parser.add_argument("--device", default="0", help="'0' for first GPU, 'cpu' for CPU")
    args = parser.parse_args()

    model = YOLO(args.model)  # loads COCO-pretrained weights

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=args.project,
        name=args.name,
        device=args.device,
        optimizer="auto",
        lr0=0.01,
        cos_lr=True,
        mosaic=1.0,
        close_mosaic=10,   # disable mosaic for the last 10 epochs
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5,
        plots=True,
        save=True,
    )

    print(f"\nTraining complete. Best weights saved under: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
