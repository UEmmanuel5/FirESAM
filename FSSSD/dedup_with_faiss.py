import argparse
import os
import shutil
from typing import Tuple

import faiss
import numpy as np

from evfr.datasets.image_folder import ImageFolderDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate images using cosine similarity on embeddings + FAISS."
    )
    parser.add_argument(
        "--image-root",
        type=str,
        required=True,
        help="Root directory of images (must match what was used to compute embeddings).",
    )
    parser.add_argument(
        "--emb-path",
        type=str,
        required=True,
        help="Path to embeddings .npz file (e.g. ../fire_embeddings/embeddings.npz).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory where deduplicated images will be copied.",
    )
    parser.add_argument(
        "--sim-thresh",
        type=float,
        default=0.96,
        help=(
            "Cosine similarity threshold to treat two images as duplicates. "
            "Higher keeps fewer images. Default: 0.96."
        ),
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="How many nearest neighbors to inspect per image. Default: 20.",
    )
    return parser.parse_args()


def load_embeddings(path: str) -> Tuple[np.ndarray, int, int]:
    """
    Load embeddings from a .npz file and return (embeddings, N, D).
    Tries keys: 'embeddings', 'features', 'arr_0'.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    data = np.load(path)
    for key in ["embeddings", "features", "arr_0"]:
        if key in data:
            emb = data[key]
            break
    else:
        raise KeyError(
            f"No expected key ('embeddings', 'features', 'arr_0') found in {path}. "
            f"Available keys: {list(data.keys())}"
        )

    emb = emb.astype(np.float32)
    N, D = emb.shape
    return emb, N, D


def dedup_with_faiss(
    embeddings: np.ndarray,
    sim_thresh: float,
    topk: int,
) -> np.ndarray:
    """
    embeddings: (N, D) float32
    sim_thresh: cosine similarity threshold to consider "duplicate"
    topk:       number of neighbors to inspect for each point

    Returns:
        keep_indices: 1D array of indices to keep (unique images)
    """
    N, D = embeddings.shape

    # L2-normalize so dot product = cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    X = embeddings / norms

    index = faiss.IndexFlatIP(D)
    index.add(X)
    keep = np.ones(N, dtype=bool)
    for i in range(N):
        if not keep[i]:
            continue
        sims, idxs = index.search(X[i:i + 1], topk)
        sims = sims[0]
        idxs = idxs[0]

        for sim, j in zip(sims, idxs):
            if j <= i:
                continue
            if sim >= sim_thresh:
                keep[j] = False
    keep_indices = np.where(keep)[0]
    return keep_indices


def main():
    args = parse_args()
    image_root = os.path.abspath(args.image_root)
    emb_path = os.path.abspath(args.emb_path)
    out_dir = os.path.abspath(args.out_dir)
    sim_thresh = args.sim_thresh
    topk = args.topk
    print(f"Image root       : {image_root}")
    print(f"Embeddings path  : {emb_path}")
    print(f"Output directory : {out_dir}")
    print(f"SIM_THRESH       : {sim_thresh}")
    print(f"TOPK             : {topk}")
    print()
    print("Loading embeddings...")
    embeddings, N, D = load_embeddings(emb_path)
    print(f"Embeddings shape: {N} x {D}")
    print(f"\nBuilding dataset from: {image_root}")
    dataset = ImageFolderDataset(root_dir=image_root)
    if len(dataset) != N:
        print(
            f"WARNING: dataset length ({len(dataset)}) != embeddings rows ({N}). "
            "Check that IMAGE_ROOT and EMB_PATH were produced from the same data."
        )
    print(f"\nRunning deduplication with SIM_THRESH={sim_thresh}, TOPK={topk}")
    keep_indices = dedup_with_faiss(embeddings, sim_thresh, topk)
    print(f"Keeping {len(keep_indices)} out of {N} images")

    os.makedirs(out_dir, exist_ok=True)
    print(f"\nCopying kept images to: {out_dir}")
    for rank, idx in enumerate(keep_indices):
        src = dataset.image_paths[idx]
        dst_name = f"{rank:06d}_" + os.path.basename(str(src))
        dst = os.path.join(out_dir, dst_name)
        shutil.copy2(str(src), dst)

    print("Done.")

if __name__ == "__main__":
    main()
