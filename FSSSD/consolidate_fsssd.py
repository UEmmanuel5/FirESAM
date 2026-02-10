import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate accepted FSSSD images/masks and build re_annotate set."
    )
    parser.add_argument(
        "--good-overlay-dir",
        type=str,
        required=True,
        help="Directory containing accepted overlay images.",
    )
    parser.add_argument(
        "--masks-root",
        type=str,
        required=True,
        help="Root directory containing all auto-generated PNG masks.",
    )
    parser.add_argument(
        "--source-images-dir",
        type=str,
        required=True,
        help="Root directory of original deduplicated images.",
    )
    return parser.parse_args()


def collect_good_overlay_stems(overlay_dir: Path) -> Set[str]:
    """
    Collect basenames (without extension) of accepted overlay images
    """
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    stems: Set[str] = set()
    for ext in exts:
        for p in overlay_dir.glob(f"*{ext}"):
            if p.is_file():
                stems.add(p.stem)
    return stems


def collect_masks(mask_root: Path) -> List[Path]:
    """
    Recursively collect all PNG masks under MASKS_ROOT.
    """
    return [p for p in mask_root.rglob("*.png") if p.is_file()]

def index_source_images(src_root: Path) -> Dict[str, Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    index: Dict[str, Path] = {}
    for root, _, files in os.walk(src_root):
        for name in files:
            if not name.lower().endswith(exts):
                continue
            p = Path(root) / name
            stem = p.stem
            if stem in index:
                continue
            index[stem] = p
    return index


def main() -> None:
    args = parse_args()
    good_overlay_dir = Path(args.good_overlay_dir)
    masks_root = Path(args.masks_root)
    source_images_dir = Path(args.source_images_dir)

    if not good_overlay_dir.is_dir():
        raise SystemExit(f"GOOD_OVERLAY_DIR not found: {good_overlay_dir}")
    if not masks_root.is_dir():
        raise SystemExit(f"MASKS_ROOT not found: {masks_root}")
    if not source_images_dir.is_dir():
        raise SystemExit(f"SOURCE_IMAGES_DIR not found: {source_images_dir}")
    print(f"Scanning overlays in: {good_overlay_dir}")
    good_stems = collect_good_overlay_stems(good_overlay_dir)
    print(f"Found {len(good_stems)} good overlay images.")
    masks_out = good_overlay_dir / "masks"
    images_out = good_overlay_dir / "images"
    reannotate_out = good_overlay_dir / "re_annotate"
    masks_out.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(parents=True, exist_ok=True)
    reannotate_out.mkdir(parents=True, exist_ok=True)
    print(f"Scanning masks under: {masks_root}")
    all_masks = collect_masks(masks_root)
    print(f"Found {len(all_masks)} mask files (.png).")
    used_stems: Set[str] = set()
    for mask_path in all_masks:
        stem = mask_path.stem
        if stem not in good_stems:
            continue
        dst_mask = masks_out / mask_path.name
        if not dst_mask.exists():
            shutil.copy2(mask_path, dst_mask)
        used_stems.add(stem)
    print(f"Copied {len(used_stems)} masks matching good overlays.")
    print(f"Indexing source images in: {source_images_dir}")
    image_index = index_source_images(source_images_dir)
    print(f"Indexed {len(image_index)} source images.")
    copied_images = 0
    for stem in used_stems:
        src_img = image_index.get(stem)
        if src_img is None:
            print(f"[WARN] No source image found for stem '{stem}'")
            continue
        dst_img = images_out / src_img.name
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)
            copied_images += 1

    print(f"Copied {copied_images} images for good masks.")
    all_image_stems = set(image_index.keys())
    reannotate_stems = all_image_stems - used_stems
    print(f"{len(reannotate_stems)} images will go to re_annotate.")

    copied_reannotate = 0
    for stem in sorted(reannotate_stems):
        src_img = image_index[stem]
        dst_img = reannotate_out / src_img.name
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)
            copied_reannotate += 1

    print(f"Copied {copied_reannotate} images into re_annotate.")
    print("\nDone.")
    print(f"FSSSD masks       : {masks_out}")
    print(f"FSSSD images      : {images_out}")
    print(f"FSSSD re_annotate : {reannotate_out}")


if __name__ == "__main__":
    main()
