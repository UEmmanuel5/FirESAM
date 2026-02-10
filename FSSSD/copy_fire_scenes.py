import os
import re
import csv
import shutil
import argparse
from glob import glob

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
WANTED = {"fire_CV", "bothFireAndSmoke_CV"}

LABEL_RX = re.compile(
    r'^(?P<label>neitherFireNorSmoke_CV|smoke_CV|fire_CV|bothFireAndSmoke_CV)\s*(?P<id>\d+)$',
    re.IGNORECASE,
)


def list_images(root: str):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob(os.path.join(root, f"**/*{ext}"), recursive=True))
    return sorted(set(paths))


def parse_label_from_stem(stem: str):
    m = LABEL_RX.match(stem)
    if not m:
        return None
    label = m.group("label")
    mapping = {
        "neitherfirenorsmoke_cv": "neitherFireNorSmoke_CV",
        "smoke_cv": "smoke_CV",
        "fire_cv": "fire_CV",
        "bothfireandsmoke_cv": "bothFireAndSmoke_CV",
    }
    return mapping.get(label.lower(), label)


def unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while True:
        cand = f"{base}_{i}{ext}"
        if not os.path.exists(cand):
            return cand
        i += 1


def copy_preserving_structure(src_path: str, images_root: str, out_root: str) -> str:
    """
    Copy src_path into out_root, preserving relative path under images_root when possible.
    Falls back to flat copy if src_path is outside images_root.
    """
    try:
        rel = os.path.relpath(src_path, images_root)
        if rel.startswith(".."):
            rel = os.path.basename(src_path)
    except Exception:
        rel = os.path.basename(src_path)

    dst_path = os.path.join(out_root, rel)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    dst_path = unique_path(dst_path)
    shutil.copy2(src_path, dst_path)
    return dst_path


def main():
    parser = argparse.ArgumentParser(
        description="Copy fire_CV and bothFireAndSmoke_CV images into a new folder."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="images",
        help="Root folder containing FASDD_CV images.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="fasdd_cv_filename_audit.csv",
        help="CSV from fasdd_cv_filename_audit.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="fire2",
        help="Destination folder to create for fire images.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not use CSV; infer labels directly from filenames.",
    )

    args = parser.parse_args()

    images_dir = os.path.abspath(args.images_dir)
    csv_path = os.path.abspath(args.csv_path)
    output_dir = os.path.abspath(args.output_dir)
    use_csv = not args.no_csv

    if not os.path.isdir(images_dir):
        raise SystemExit(f"Images directory not found: {images_dir}")

    to_copy = []

    if use_csv:
        if not os.path.isfile(csv_path):
            raise SystemExit(f"CSV not found: {csv_path}")
        print(f"Using CSV: {csv_path}")
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = row.get("image_path")
                lab = row.get("label_from_filename")
                if not p or not lab:
                    continue
                if lab in WANTED:
                    to_copy.append(p)
    else:
        print("No CSV mode: inferring labels from filenames.")
        images = list_images(images_dir)
        if not images:
            raise SystemExit(f"No images found under: {images_dir}")
        for p in images:
            stem = os.path.splitext(os.path.basename(p))[0]
            lab = parse_label_from_stem(stem)
            if lab in WANTED:
                to_copy.append(p)
    os.makedirs(output_dir, exist_ok=True)
    copied, missing = 0, 0

    print(f"Found {len(to_copy)} candidate images to copy.")
    for src in to_copy:
        if not os.path.isfile(src):
            missing += 1
            continue
        copy_preserving_structure(src, images_dir, output_dir)
        copied += 1

    print("=== Copy report ===")
    print(f"Selected (fire_CV + bothFireAndSmoke_CV): {len(to_copy)}")
    print(f"Copied: {copied}")
    print(f"Missing at source: {missing}")
    print(f"Destination: {output_dir}")


if __name__ == "__main__":
    main()
