"""
prepare_data.py

Converts the *raw* RDD2022 release (Pascal VOC XML) into YOLO-format labels
and builds a standard YOLO dataset directory:

    output/
        images/{train,val,test}/*.jpg
        labels/{train,val,test}/*.txt

Use this ONLY against the original RDD2022 download (per-country folders
with annotations/xmls/*.xml). If you're using a dataset that is already in
YOLO .txt format (e.g. a pre-converted Kaggle mirror such as
"rdd2022-yolo-crackscan-v2"), skip this script entirely and use
adapt_kaggle_dataset.py instead.

Only the four primary CRDDC-2022 challenge classes are kept:
    D00 -> longitudinal crack   (class 0)
    D10 -> transverse crack     (class 1)
    D20 -> alligator crack      (class 2)
    D40 -> pothole              (class 3)
Any other class label present in the raw XMLs (e.g. D01, D43, D44, D50) is
ignored, since they are not part of the standard 4-class benchmark.

Usage:
    python prepare_data.py \
        --source /path/to/RDD2022 \
        --output /path/to/rdd2022_yolo_dataset \
        --train_ratio 0.7 --val_ratio 0.15 --seed 42
"""

import argparse
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

CLASS_MAP = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}
IMG_EXTS = (".jpg", ".jpeg", ".png")


def find_image_for_xml(xml_path: Path) -> Path | None:
    """RDD2022 layout is typically .../annotations/xmls/foo.xml with a
    sibling .../images/foo.jpg. Falls back to searching the same folder."""
    candidates = []
    images_dir_guess = xml_path.parent.parent.parent / "images"
    for ext in IMG_EXTS:
        candidates.append(images_dir_guess / (xml_path.stem + ext))
        candidates.append(xml_path.parent / (xml_path.stem + ext))
    for c in candidates:
        if c.exists():
            return c
    return None


def parse_voc_xml(xml_path: Path):
    """Returns (img_w, img_h, [(class_id, xmin, ymin, xmax, ymax), ...])."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.findtext("width"))
    img_h = int(size.findtext("height"))

    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in CLASS_MAP:
            continue
        bnd = obj.find("bndbox")
        xmin = float(bnd.findtext("xmin"))
        ymin = float(bnd.findtext("ymin"))
        xmax = float(bnd.findtext("xmax"))
        ymax = float(bnd.findtext("ymax"))
        boxes.append((CLASS_MAP[name], xmin, ymin, xmax, ymax))

    return img_w, img_h, boxes


def voc_box_to_yolo_line(class_id, xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = ((xmin + xmax) / 2) / img_w
    y_center = ((ymin + ymax) / 2) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"


def collect_samples(source_root: Path):
    """Walk the whole RDD2022 tree and collect (image_path, yolo_lines)
    for every XML that has at least one box in one of the 4 target classes."""
    samples = []
    xml_files = list(source_root.rglob("*.xml"))
    print(f"Found {len(xml_files)} XML annotation files under {source_root}")

    if not xml_files:
        print("No XML files found. If your dataset is already in YOLO .txt "
              "format, use adapt_kaggle_dataset.py instead of this script.")

    skipped_no_image = 0
    skipped_no_boxes = 0

    for xml_path in xml_files:
        img_path = find_image_for_xml(xml_path)
        if img_path is None:
            skipped_no_image += 1
            continue

        img_w, img_h, boxes = parse_voc_xml(xml_path)
        if not boxes:
            skipped_no_boxes += 1
            continue

        lines = [voc_box_to_yolo_line(*b, img_w, img_h) for b in boxes]
        samples.append((img_path, lines))

    print(f"Usable samples: {len(samples)} "
          f"(skipped {skipped_no_image} missing images, "
          f"{skipped_no_boxes} with no target-class boxes)")
    return samples


def split_and_write(samples, output_root: Path, train_ratio, val_ratio, seed):
    random.seed(seed)
    random.shuffle(samples)

    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:],
    }

    for split_name, split_samples in splits.items():
        img_out = output_root / "images" / split_name
        lbl_out = output_root / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, lines in split_samples:
            dest_img = img_out / img_path.name
            shutil.copy2(img_path, dest_img)
            label_path = lbl_out / (img_path.stem + ".txt")
            label_path.write_text("\n".join(lines))

        print(f"{split_name}: {len(split_samples)} images written to {img_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to raw RDD2022 dataset root")
    parser.add_argument("--output", required=True, help="Path to write YOLO-formatted dataset")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_root = Path(args.source)
    output_root = Path(args.output)

    samples = collect_samples(source_root)
    if not samples:
        raise SystemExit("No usable samples found — check --source path and folder structure.")

    split_and_write(samples, output_root, args.train_ratio, args.val_ratio, args.seed)
    print("\nDone. Point rdd2022.yaml's `path:` field at:", output_root.resolve())


if __name__ == "__main__":
    main()
