import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan  # type: ignore
except ImportError as exc:
    raise ImportError(
        "hdbscan is required. Install with: pip install hdbscan"
    ) from exc


def pool_hidden_state(hidden: torch.Tensor, method: str) -> np.ndarray:
    # Expected shapes like [batch, seq, hidden] or [seq, hidden]
    if hidden.dim() == 3:
        # use first batch
        hidden = hidden[0]
    if hidden.dim() != 2:
        raise ValueError(f"Unexpected hidden state shape: {tuple(hidden.shape)}")

    if method == "mean":
        vec = hidden.mean(dim=0)
    elif method == "last":
        vec = hidden[-1]
    elif method == "max":
        vec = hidden.max(dim=0).values
    else:
        raise ValueError(f"Unknown pool method: {method}")
    return vec.detach().cpu().float().numpy()


def extract_records_from_pt(pt_path: Path, pool: str) -> List[Dict]:
    data = torch.load(pt_path, map_location="cpu")
    rounds = data.get("rounds", [])
    sample_idx = data.get("sample_index", -1)
    query = data.get("query", "")
    rows: List[Dict] = []

    for r in rounds:
        round_id = r.get("round_id", -1)
        correctness = r.get("correctness", None)
        pred_more = r.get("prediction_do_more_retriever", None)
        hidden_states: Dict[str, torch.Tensor] = r.get("hidden_states", {})
        for layer_name, hidden in hidden_states.items():
            try:
                emb = pool_hidden_state(hidden, pool)
            except Exception:
                continue
            rows.append(
                {
                    "pt_file": pt_path.name,
                    "sample_index": sample_idx,
                    "query": query,
                    "round_id": round_id,
                    "layer_name": layer_name,
                    "correctness": correctness,
                    "prediction_do_more_retriever": pred_more,
                    "embedding": emb,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_dir", type=str, required=True)
    parser.add_argument("--failed_index_csv", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--pool", type=str, default="mean", choices=["mean", "last", "max"])
    parser.add_argument("--min_cluster_size", type=int, default=5)
    parser.add_argument("--min_samples", type=int, default=1)
    parser.add_argument("--layer_contains", type=str, default="")
    parser.add_argument("--max_pca_dim", type=int, default=50)
    args = parser.parse_args()

    traj_dir = Path(args.trajectory_dir)
    if not traj_dir.exists():
        raise FileNotFoundError(f"trajectory_dir not found: {traj_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else traj_dir / "cluster_viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(traj_dir.glob("sample_*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No sample_*.pt found in {traj_dir}")

    if args.failed_index_csv:
        failed_df = pd.read_csv(args.failed_index_csv)
        if "sample_index" not in failed_df.columns:
            raise ValueError("failed_index_csv must contain column: sample_index")
        keep_ids = set(int(x) for x in failed_df["sample_index"].dropna().tolist())
        pt_files = [p for p in pt_files if p.stem.startswith("sample_") and int(p.stem.split("_")[1]) in keep_ids]
        if not pt_files:
            raise RuntimeError("No .pt files matched sample_index from failed_index_csv.")

    records: List[Dict] = []
    for pt in pt_files:
        records.extend(extract_records_from_pt(pt, args.pool))

    if not records:
        raise RuntimeError("No hidden states found in .pt trajectories.")

    if args.layer_contains:
        records = [r for r in records if args.layer_contains in r["layer_name"]]
        if not records:
            raise RuntimeError(f"No records after filtering layer_contains={args.layer_contains}")

    X = np.stack([r["embedding"] for r in records], axis=0)
    X = StandardScaler().fit_transform(X)

    pca_dim = min(args.max_pca_dim, X.shape[1], max(2, X.shape[0] - 1))
    X_for_cluster = PCA(n_components=pca_dim, random_state=42).fit_transform(X)
    X_2d = PCA(n_components=2, random_state=42).fit_transform(X_for_cluster)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(X_for_cluster)

    df = pd.DataFrame(
        {
            "pt_file": [r["pt_file"] for r in records],
            "sample_index": [r["sample_index"] for r in records],
            "query": [r["query"] for r in records],
            "round_id": [r["round_id"] for r in records],
            "layer_name": [r["layer_name"] for r in records],
            "correctness": [r["correctness"] for r in records],
            "prediction_do_more_retriever": [r["prediction_do_more_retriever"] for r in records],
            "x": X_2d[:, 0],
            "y": X_2d[:, 1],
            "cluster": labels,
        }
    )

    csv_path = output_dir / "hiddenstate_hdbscan_points.csv"
    df.to_csv(csv_path, index=False)

    # Plot
    plt.figure(figsize=(9, 7))
    unique_labels = sorted(df["cluster"].unique())
    cmap = plt.cm.get_cmap("tab20", max(len(unique_labels), 1))
    for i, label in enumerate(unique_labels):
        sub = df[df["cluster"] == label]
        name = "noise(-1)" if label == -1 else f"cluster_{label}"
        plt.scatter(sub["x"], sub["y"], s=24, alpha=0.8, label=name, color=cmap(i))
    plt.title("HDBSCAN Clusters of Trajectory Hidden States (2D PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    fig_path = output_dir / "hiddenstate_hdbscan_plot.png"
    plt.savefig(fig_path, dpi=180)
    plt.close()

    summary = df["cluster"].value_counts().sort_index()
    summary_path = output_dir / "hiddenstate_hdbscan_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"num_points={len(df)}\n")
        f.write(f"num_clusters(excluding noise)={len([c for c in unique_labels if c != -1])}\n")
        f.write("cluster_counts:\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print(f"Saved: {csv_path}")
    print(f"Saved: {fig_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
