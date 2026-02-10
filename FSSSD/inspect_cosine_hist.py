import argparse
import os

import faiss
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect cosine similarity distribution of nearest neighbors "
                    "and 2D PCA projection of embeddings."
    )
    parser.add_argument(
        "--emb-path",
        type=str,
        required=True,
        help="Path to embeddings .npz file",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help="Number of neighbors to retrieve per point (excluding self in stats). "
             "Default: 50.",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=5000,
        help="Number of points to subsample for speed. Use -1 for no subsampling. "
             "Default: 5000.",
    )
    parser.add_argument(
        "--save-prefix",
        type=str,
        default=None,
        help="If set, save plots as '<save-prefix>_hist.png' and "
             "'<save-prefix>_pca.png' instead of (or in addition to) interactive display.",
    )
    return parser.parse_args()


def load_embeddings(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    data = np.load(path)
    for key in ["embeddings", "features", "arr_0"]:
        if key in data:
            emb = data[key]
            break
    else:
        raise KeyError(f"No expected key in {path}. Keys: {list(data.keys())}")
    emb = emb.astype(np.float32)
    return emb


def main():
    args = parse_args()

    emb_path = args.emb_path
    topk = args.topk
    subsample_n = None if args.subsample == -1 else args.subsample
    save_prefix = args.save_prefix

    print(f"Loading embeddings from: {emb_path}")
    X = load_embeddings(emb_path)
    N, D = X.shape
    print(f"Embeddings shape: {N} x {D}")
    if subsample_n is not None and subsample_n < N:
        idx = np.random.choice(N, size=subsample_n, replace=False)
        X = X[idx]
        N = X.shape[0]
        print(f"Subsampled to {N} points")

    # L2-normalize to dot product = cosine similarity
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    Xn = X / norms

    print("Building FAISS index (IndexFlatIP)...")
    index = faiss.IndexFlatIP(D)
    index.add(Xn)

    print(f"Searching top-{topk} neighbors for each point...")
    sims, idxs = index.search(Xn, topk + 1)
    neighbor_sims = sims[:, 1:].reshape(-1)
    print("\nStats on neighbor cosine similarities:")
    for q in [0, 10, 25, 50, 75, 90, 95, 99, 100]:
        val = float(np.percentile(neighbor_sims, q))
        print(f"  {q:2d}th percentile: {val:.4f}")

    # Histogram of neighbor similarities
    plt.figure()
    plt.hist(neighbor_sims, bins=100)
    plt.xlabel("Cosine similarity to nearest neighbors")
    plt.ylabel("Count")
    plt.title(f"Neighbor cosine similarities (N={N}, TOPK={topk})")
    plt.tight_layout()

    if save_prefix is not None:
        hist_path = f"{save_prefix}_hist.png"
        plt.savefig(hist_path, dpi=200)
        print(f"Saved histogram to: {hist_path}")

    # PCA projection of embeddings
    print("\nComputing 2D PCA projection...")
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(Xn)

    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], s=2, alpha=0.3)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA projection of embeddings")
    plt.tight_layout()

    if save_prefix is not None:
        pca_path = f"{save_prefix}_pca.png"
        plt.savefig(pca_path, dpi=200)
        print(f"Saved PCA scatter to: {pca_path}")

    plt.show()


if __name__ == "__main__":
    main()
