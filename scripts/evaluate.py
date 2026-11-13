"""
evaluate.py — run validation and print/save the metrics needed for the
paper's Results tables (Precision, Recall, mAP@0.5, mAP@0.5:0.95, per class).

Usage:
    python evaluate.py --weights runs/rdd2022/yolo_rdd2022/weights/best.pt --data ../rdd2022.yaml
"""

import argparse
import json
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to trained .pt weights")
    parser.add_argument("--data", default="../rdd2022.yaml")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split=args.split, imgsz=args.imgsz,
                         conf=args.conf, iou=args.iou, plots=True)

    class_names = model.names
    results = {
        "overall": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
        },
        "per_class": {},
    }

    # metrics.box.ap50 / metrics.box.ap are per-class arrays aligned with class indices
    for i, name in class_names.items():
        try:
            results["per_class"][name] = {
                "AP50": float(metrics.box.ap50[i]),
                "AP50-95": float(metrics.box.ap[i]),
            }
        except IndexError:
            pass

    print(json.dumps(results, indent=2))

    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to evaluation_results.json")
    print("Confusion matrix and PR curves saved under the runs/ directory (see 'plots=True' output path above).")


if __name__ == "__main__":
    main()
