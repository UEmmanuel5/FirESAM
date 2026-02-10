import argparse
import csv
import glob
import os
from pathlib import Path
from typing import Optional, Set

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FireSAM bbox CSV from deduplicated images + FASDD YOLO labels."
    )
    parser.add_argument(
        "--dedup-root",
        type=str,
        required=True,
        help="Root directory where deduplicated images live (e.g. fire2_dedu96_150).",
    )
    parser.add_argument(
        "--yolo-labels-root",
        type=str,
        required=True,
        help="Root of YOLO label files (e.g. .../annotations/YOLO_CV/labels).",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default="fire2_dedu96_boxes.csv",
        help="Output CSV path (default: fire2_dedu96_boxes.csv).",
    )
    parser.add_argument(
        "--fire-class-ids",
        type=int,
        nargs="+",
        default=[0],
        help=(
            "YOLO class IDs to include (e.g. 0 for fire only, or 0 1 for fire+smoke). "
            "Default: 0."
        ),
    )
    return parser.parse_args()


def find_yolo_file(stem: str, labels_root: Path) -> Optional[Path]:
    """
    Find YOLO txt file by stem under labels_root.
    Example stem: 'fire_CV023379' -> '.../fire_CV023379.txt'
    """
    pattern = str(labels_root / f"{stem}.txt")
    matches = glob.glob(pattern)
    if not matches:
        return None
    if len(matches) > 1:
        print(f"[WARN] multiple YOLO files for {stem}, using first: {matches[0]}")
    return Path(matches[0])


def yolo_line_to_xyxy(line: str, w: int, h: int):
    """
    Convert one YOLO-normalized line to (cls, x1,y1,x2,y2) in pixels.
    YOLO line: cls xc yc bw bh  (all 0..1, relative to width/height)
    For example:
    image_path,x1,y1,x2,y2
    folder_x/.../image.jpg,  x1,y1,x2,y2
    
    more concretely,
    
    image_path,x1,y1,x2,y2
    000000_bothFireAndSmoke_CV000000.jpg,433.00000000000006,503.99999999999994,470.00000000000006,527.0
    000001_bothFireAndSmoke_CV000001.jpg,453.0,476.0,477.0,519.0
    000002_bothFireAndSmoke_CV000002.jpg,451.0,454.00000000000006,495.00000000000006,506.00000000000006
    ...
    """
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    cls = int(float(parts[0]))
    xc, yc, bw, bh = map(float, parts[1:])

    x1 = (xc - bw / 2.0) * w
    x2 = (xc + bw / 2.0) * w
    y1 = (yc - bh / 2.0) * h
    y2 = (yc + bh / 2.0) * h
    x1 = max(0.0, min(x1, w - 1))
    x2 = max(0.0, min(x2, w - 1))
    y1 = max(0.0, min(y1, h - 1))
    y2 = max(0.0, min(y2, h - 1))
    return cls, x1, y1, x2, y2


def main():
    args = parse_args()

    dedup_root = Path(args.dedup_root)
    yolo_labels_root = Path(args.yolo_labels_root)
    csv_out = Path(args.csv_out)
    fire_class_ids: Set[int] = set(args.fire_class_ids)
    print(f"DEDUP_ROOT       : {dedup_root}")
    print(f"YOLO_LABELS_ROOT : {yolo_labels_root}")
    print(f"CSV_OUT          : {csv_out}")
    print(f"FIRE_CLASS_IDS   : {sorted(fire_class_ids)}\n")
    if not dedup_root.is_dir():
        raise SystemExit(f"DEDUP_ROOT not found: {dedup_root}")
    if not yolo_labels_root.is_dir():
        raise SystemExit(f"YOLO_LABELS_ROOT not found: {yolo_labels_root}")

    rows: list[list[object]] = []
    for img_path in dedup_root.rglob("*"):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        rel_img = img_path.relative_to(dedup_root)

        # Dedup name: '000123_fire_CV023379.jpg'
        # Recover original stem after first underscore: 'fire_CV023379'
        stem_new = img_path.stem
        parts = stem_new.split("_", 1)
        orig_stem = parts[1] if len(parts) == 2 else stem_new
        yolo_file = find_yolo_file(orig_stem, yolo_labels_root)
        if yolo_file is None:
            print(f"[WARN] no YOLO label for {orig_stem}, skipping {img_path}")
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] failed to read {img_path}, skipping.")
            continue
        h, w = img.shape[:2]
        with open(yolo_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parsed = yolo_line_to_xyxy(line, w, h)
                if parsed is None:
                    continue
                cls, x1, y1, x2, y2 = parsed
                if cls not in fire_class_ids:
                    continue
                rows.append([
                    str(rel_img).replace(os.sep, "/"),
                    x1,
                    y1,
                    x2,
                    y2,
                ])
    print(f"\nWriting {len(rows)} rows to {csv_out}")
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "x1", "y1", "x2", "y2"])
        w.writerows(rows)
    print("Done.")


if __name__ == "__main__":
    main()
