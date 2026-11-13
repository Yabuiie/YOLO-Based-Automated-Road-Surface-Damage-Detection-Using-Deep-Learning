# YOLO Road Damage Detection — RDD2022

Full pipeline: get RDD2022 into YOLO format → train → evaluate → run inference.

There are **two possible data sources** and they need different first steps —
pick the one that matches what you actually downloaded.

| Source | Format | First step |
|---|---|---|
| Official RDD2022 release (sekilab) | Pascal VOC XML | `scripts/prepare_data.py` |
| Kaggle "rdd2022-yolo-crackscan-v2" (or any pre-converted YOLO export) | YOLO `.txt` already | `scripts/adapt_kaggle_dataset.py` |

Do not run `prepare_data.py` against the Kaggle mirror — it parses `*.xml`
files, and a pre-converted YOLO dataset ships `.txt` labels, not XML. It
will find 0 XML files and produce nothing usable.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Get the dataset into YOLO format

### Option A — official RDD2022 (Pascal VOC XML)

```bash
python scripts/prepare_data.py \
  --source /path/to/RDD2022 \
  --output /path/to/rdd2022_yolo_dataset \
  --train_ratio 0.7 --val_ratio 0.15 --seed 42
```

Then edit `rdd2022.yaml` and set `path:` to that output folder.

### Option B — Kaggle "rdd2022-yolo-crackscan-v2" (already YOLO format)

```bash
kaggle datasets download -d sreekaraditya/rdd2022-yolo-crackscan-v2 --unzip -p /path/to/kaggle_rdd2022

python scripts/adapt_kaggle_dataset.py \
  --source /path/to/kaggle_rdd2022 \
  --out ../rdd2022.yaml
```

This inspects the extracted folder, detects the train/val/test layout,
reads the real class list from whatever `data.yaml`/`classes.txt` the
dataset ships (don't assume it's the same 4 classes as the official
release — verify what the script prints against your paper's Table I), and
writes a corrected `rdd2022.yaml` for you. No images are copied.

If the script can't find a `data.yaml`/`classes.txt` inside the dataset,
it will stop and ask you to pass `--fallback-names` explicitly — **do not
guess the class order**, since a wrong order silently mislabels every
detection (e.g. potholes reported as alligator cracks).

## 3. Train

```bash
cd scripts
python train.py --data ../rdd2022.yaml --model yolo11s.pt --epochs 100 --imgsz 640 --batch 16
```

- Default base model is `yolo11s.pt` (Ultralytics YOLO11) — same training
  API as YOLOv8, but its C3k2 backbone and C2PSA attention block give a
  consistently better accuracy/parameter tradeoff than YOLOv8 at the same
  size. Use `yolo11n.pt` if you're GPU/memory constrained, `yolo11m.pt` if
  you have more headroom.
- If you need a literal YOLOv8 number to compare against (e.g. citing a
  paper's "YOLOv8 baseline" figure), pass `--model yolov8s.pt` instead —
  nothing else changes.
- Best weights land at `runs/rdd2022/yolo_rdd2022/weights/best.pt`.
- Training curves, confusion matrix, and sample predictions are auto-saved
  in the same run folder.

## 4. Evaluate (for the paper's Results tables)

```bash
python evaluate.py --weights runs/rdd2022/yolo_rdd2022/weights/best.pt --data ../rdd2022.yaml --split test
```

Prints and saves overall + per-class Precision, Recall, mAP@0.5, and mAP@0.5:0.95
to `evaluation_results.json`.

**Note:** if you're filling in a results table for a paper, these numbers
have to come from an actual completed training + evaluation run on your
data — there's no shortcut. Don't leave placeholder values in a submitted
table.

## 5. Run inference / measure FPS

```bash
python inference.py --weights runs/rdd2022/yolo_rdd2022/weights/best.pt --source /path/to/test_images
```

Annotated images are saved under `runs/inference/predictions`, and console output
reports FPS — useful for the deployment/real-time discussion in the paper.

## Notes

- The official RDD2022 pipeline (`prepare_data.py`) keeps only the 4
  primary CRDDC-2022 classes: D00, D10, D20, D40. Other labels present in
  some country subsets (e.g. D01, D43, D44, D50) are dropped since they
  aren't part of the standard benchmark.
- The Kaggle mirror may use a different class set/order than the official
  4-class benchmark — always check what `adapt_kaggle_dataset.py` prints
  before training, and update your paper's class table if it differs.
- If `ultralytics` can't find a GPU, training will silently fall back to CPU
  and be very slow — check `nvidia-smi` first, or set `--device cpu` explicitly
  to acknowledge it.
