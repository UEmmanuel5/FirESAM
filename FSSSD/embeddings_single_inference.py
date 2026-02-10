import argparse
from pathlib import Path

import numpy as np
import torch

from evfr.embeddings.hf_dinov2 import HuggingFaceDinoV2Embedding
from evfr.datasets.image_folder import ImageFolderDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract visual embeddings from images using DINOv2 (EVFR)."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input images (e.g. fire2).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where embeddings.npz will be saved.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to run inference on. "
             "If omitted, uses 'cuda' if available, otherwise 'cpu'.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="HuggingFace model name. If omitted, uses EVFR default "
             "(typically 'facebook/dinov2-base').",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    # Auto-detect device if not provided
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Input dir   : {input_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Device      : {device}")
    print(f"Batch size  : {args.batch_size}")
    if args.model_name:
        print(f"Model name  : {args.model_name}")
    else:
        print("Model name  : (EVFR default DINOv2)")

    # Initialize model
    print("\nInitializing embedding model...")
    model = HuggingFaceDinoV2Embedding(
        model_name=args.model_name,
        device=device,
        use_processor=True,
    )

    # Build dataset
    print(f"\nLoading images from: {input_dir}")
    dataset = ImageFolderDataset(
        root_dir=str(input_dir),
        disable_transform=True,
    )
    num_images = len(dataset)
    if num_images == 0:
        raise SystemExit(f"No images found under: {input_dir}")
    print(f"Found {num_images} images.")

    # Extract embeddings
    print("\nExtracting embeddings...")
    num_workers = 4 if device == "cuda" else 0
    embeddings = model.extract_dataset(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
    )

    # Save
    output_path = output_dir / "embeddings.npz"
    image_paths = [str(p) for p in dataset.image_paths]

    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        image_paths=image_paths,
    )

    print("\nDone.")
    print(f"Saved embeddings to: {output_path}")
    print(f"Embeddings shape   : {embeddings.shape}")


if __name__ == "__main__":
    main()
