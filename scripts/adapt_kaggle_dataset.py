"""
adapt_kaggle_dataset.py

For datasets that are ALREADY in YOLO format — e.g. the Kaggle mirror
"rdd2022-yolo-crackscan-v2" (https://www.kaggle.com/datasets/sreekaraditya/
rdd2022-yolo-crackscan-v2), Roboflow exports, or any other pre-converted
YOLO dataset. Do NOT run prepare_data.py against these: prepare_data.py
expects raw Pascal-VOC XML annotations, and a pre-converted dataset has none
— it would silently find 0 XML files and produce nothing.

This script does not copy or convert any images. It:
  1. Downloads nothing — point it at a folder you already extracted
     (e.g. via `kaggle datasets download -d sreekaraditya/rdd2022-yolo-crackscan-v2 --unzip`).
  2. Auto-detects the split layout, handling both common conventions:
       images/train, images/val|valid, images/test  (+ matching labels/)
     and
       train/images, train/labels, valid/..., test/...
  3. Looks for a data.yaml / dataset.yaml / classes.txt shipped with the
     dataset to read the REAL class list and count — it does not assume
     the 4-class CRDDC-2022 set, because a "CrackScan" mirror may include
     extra classes (e.g. D0w0, D43, D44, D50, repair patches, manholes).
  4. Writes a corrected rdd2022.yaml pointing straight at the dataset
     (no file copying needed — train.py/evaluate.py/inference.py can use
     it as-is).
  5. Spot-checks that image/label counts per split are non-zero and sane.

Usage:
    python adapt_kaggle_dataset.py --source /path/to/extracted/kaggle_dataset --out ../rdd2022.yaml
"""

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
SPLIT_ALIASES = {"train": ["train"], "val": ["val", "valid", "validation"], "test": ["test"]}


def find_existing_data_yaml(source: Path):
    for name in ("data.yaml", "dataset.yaml", "data.yml"):
        p = source / name
        if p.exists():
            return p
    # sometimes nested one level down
    for p in source.glob("*/data.yaml"):
        return p
    return None


def load_classes_from_yaml(yaml_path: Path):
    if yaml is None:
        print("pyyaml not installed — cannot auto-read", yaml_path, "(pip install pyyaml)")
        return None
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if names is None:
        return None
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return None


def load_classes_from_txt(txt_path: Path):
    lines = [l.strip() for l in txt_path.read_text().splitlines() if l.strip()]
    return {i: n for i, n in enumerate(lines)}


def detect_split_dir(source: Path, split: str):
    """Returns (images_dir, labels_dir) for a split, or (None, None)."""
    aliases = SPLIT_ALIASES[split]

    # Layout A: images/<split>, labels/<split>
    for alias in aliases:
        img_dir = source / "images" / alias
        lbl_dir = source / "labels" / alias
        if img_dir.is_dir():
            return img_dir, lbl_dir if lbl_dir.is_dir() else None

    # Layout B: <split>/images, <split>/labels
    for alias in aliases:
        img_dir = source / alias / "images"
        lbl_dir = source / alias / "labels"
        if img_dir.is_dir():
            return img_dir, lbl_dir if lbl_dir.is_dir() else None

    return None, None


def count_files(d: Path, exts=None):
    if d is None or not d.is_dir():
        return 0
    if exts is None:
        return sum(1 for _ in d.iterdir() if _.is_file())
    return sum(1 for f in d.iterdir() if f.suffix.lower() in exts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                         help="Path to the extracted, already-YOLO-format dataset root")
    parser.add_argument("--out", default="../rdd2022.yaml",
                         help="Where to write the corrected dataset yaml")
    parser.add_argument("--fallback-names", nargs="*", default=None,
                         help="Class names to use, in class-index order, ONLY if no "
                              "data.yaml/classes.txt is found in --source")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    if not source.is_dir():
        raise SystemExit(f"--source {source} does not exist")

    # 1. classes
    names = None
    yaml_path = find_existing_data_yaml(source)
    if yaml_path:
        names = load_classes_from_yaml(yaml_path)
        print(f"Read class list from {yaml_path}")
    if names is None:
        txt_path = source / "classes.txt"
        if txt_path.exists():
            names = load_classes_from_txt(txt_path)
            print(f"Read class list from {txt_path}")
    if names is None and args.fallback_names:
        names = {i: n for i, n in enumerate(args.fallback_names)}
        print("Using --fallback-names (no data.yaml/classes.txt found in the dataset)")
    if names is None:
        print("\nERROR: could not find a data.yaml, dataset.yaml, or classes.txt inside "
              f"{source}, and no --fallback-names was given.")
        print("List the folder manually and re-run with --fallback-names "
              "\"D00_longitudinal_crack\" \"D10_transverse_crack\" ... in the correct class-index order.")
        print("Do NOT guess this — if the class order is wrong, every label will be silently "
              "mapped to the wrong damage type.")
        sys.exit(1)

    print("Classes:", names)

    # 2. splits
    split_paths = {}
    for split in ("train", "val", "test"):
        img_dir, lbl_dir = detect_split_dir(source, split)
        if img_dir is None:
            print(f"  [{split}] not found — skipping")
            continue
        n_img = count_files(img_dir, IMG_EXTS)
        n_lbl = count_files(lbl_dir, {".txt"}) if lbl_dir else 0
        print(f"  [{split}] images={n_img} ({img_dir})  labels={n_lbl} ({lbl_dir})")
        if n_img == 0:
            print(f"    WARNING: no images found for split '{split}'")
        if lbl_dir is None or n_lbl == 0:
            print(f"    WARNING: no labels found for split '{split}' — check the layout by hand")
        split_paths[split] = img_dir.relative_to(source).as_posix()

    if "train" not in split_paths:
        raise SystemExit("Could not locate a train split — inspect the folder structure under "
                          f"{source} and adjust SPLIT_ALIASES/detect_split_dir if needed.")

    # 3. write corrected yaml
    out_path = Path(args.out)
    lines = [
        "# Auto-generated by adapt_kaggle_dataset.py — points directly at a pre-converted",
        "# YOLO-format dataset (no copying performed).",
        "",
        f"path: {source.as_posix()}",
        f"train: {split_paths.get('train', 'images/train')}",
        f"val: {split_paths.get('val', split_paths.get('train'))}",
    ]
    if "test" in split_paths:
        lines.append(f"test: {split_paths['test']}")
    lines += [
        "",
        f"nc: {len(names)}",
        "names:",
    ]
    for i in sorted(names):
        lines.append(f"  {i}: {names[i]}")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_path.resolve()}")
    if "val" not in split_paths:
        print("NOTE: no separate val split was found — val: currently points at the train "
              "folder, which will overstate validation metrics. Fix this manually if the "
              "dataset doesn't ship one (e.g. carve one out of train).")


if __name__ == "__main__":
    main()
