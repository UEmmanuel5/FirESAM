import os
import re
import csv
import argparse
from glob import glob
from collections import Counter

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

LABEL_RX = re.compile(
    r'^(?P<label>neitherFireNorSmoke_CV|smoke_CV|fire_CV|bothFireAndSmoke_CV)\s*(?P<id>\d+)$',
    re.IGNORECASE,
)


def list_images(root):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob(os.path.join(root, f"**/*{ext}"), recursive=True))
    return sorted(set(paths))


def parse_label(stem):
    m = LABEL_RX.match(stem)
    if not m:
        return None
    label = m.group("label").lower()
    if label == "neitherfirenorsmoke_cv":
        label = "neitherFireNorSmoke_CV"
    elif label == "smoke_cv":
        label = "smoke_CV"
    elif label == "fire_cv":
        label = "fire_CV"
    elif label == "bothfireandsmoke_cv":
        label = "bothFireAndSmoke_CV"
    return label


def main():
    parser = argparse.ArgumentParser(
        description="FASDD_CV filename audit."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="images",
        help="Root folder containing FASDD_CV images.",
    )
    parser.add_argument(
        "--expect-images",
        type=int,
        default=95_314,
        help="Expected total number of images.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="fasdd_cv_filename_audit.csv",
        help="Path to write the per-image CSV.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="If set, print the first few rows using pandas (if available).",
    )

    args = parser.parse_args()

    images_dir = os.path.abspath(args.images_dir)
    expect_images = args.expect_images
    out_csv = os.path.abspath(args.output_csv)

    if not os.path.isdir(images_dir):
        raise SystemExit(f"Images directory not found: {images_dir}")

    images = list_images(images_dir)
    if not images:
        raise SystemExit(f"No images found under: {images_dir}")

    groups = Counter()
    unknown = []
    rows = []

    for p in images:
        stem = os.path.splitext(os.path.basename(p))[0]
        label = parse_label(stem)
        if label is None:
            unknown.append(os.path.basename(p))
            group = "unknown_pattern"
        else:
            group = label
            groups[label] += 1
        rows.append((p, group))

    # write CSV
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "label_from_filename"])
        for p, g in rows:
            w.writerow([p, g])

    # then report values
    total = len(images)
    assigned = sum(groups.values())
    print("=== FASDD_CV Filename Audit (images only) ===")
    print(f"Images found: {total} | expected: {expect_images} | match: {total == expect_images}")
    print(f"fire_CV: {groups['fire_CV']}")
    print(f"smoke_CV: {groups['smoke_CV']}")
    print(f"bothFireAndSmoke_CV: {groups['bothFireAndSmoke_CV']}")
    print(f"neitherFireNorSmoke_CV: {groups['neitherFireNorSmoke_CV']}")
    print(f"Unmatched filenames: {total - assigned} (see first 10 below)")
    print(f"Per-image CSV: {out_csv}")

    print("\nSample unmatched:")
    for name in unknown[:10]:
        print("  ", name)

    print(
        "\nNote: Instance totals (e.g. 73,297 fire, 53,080 smoke) cannot be "
        "verified from filenames alone."
    )
    print("annotations required to count object instances.")

    if args.preview:
        try:
            import pandas as pd

            df = pd.read_csv(out_csv)
            print("\nPreview of first 20 rows:")
            print(df.head(20).to_string(index=False))
        except Exception as e:
            print("\nInstall pandas to preview: pip install pandas")
            print(f"Pandas error: {e}")


if __name__ == "__main__":
    main()
