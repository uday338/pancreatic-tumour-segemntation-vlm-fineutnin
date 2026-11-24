"""Analyse radiomics features: standardize, PCA, k-means, size-vs-texture correlation,
and a median split on texture heterogeneity.

    python -m pancvlm.radiomics_analysis --features radiomics/radiomics_features.csv \
        --k 3 --out radiomics/analysis

Outputs: clusters.csv (case_id, cluster, PC1, PC2, texture_label), summary.json, pca_scatter.png
Note: k-means always returns k clusters, even on structureless data. Clusters are descriptive,
not biological subtypes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .common import save_json

SIZE_COL = "shape_volume_mm3"
TEXTURE_COL = "glcm_entropy"


def analyse(df: pd.DataFrame, k: int = 3, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    feats = df.drop(columns=["case_id"])
    X = StandardScaler().fit_transform(feats.values)
    pca = PCA(n_components=min(10, X.shape[1], X.shape[0])).fit(X)
    pcs = pca.transform(X)
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)

    out = pd.DataFrame({"case_id": df["case_id"], "cluster": km.labels_ + 1,
                        "PC1": pcs[:, 0], "PC2": pcs[:, 1]})
    med = df[TEXTURE_COL].median()
    out["texture_label"] = np.where(df[TEXTURE_COL] > med, "heterogeneous", "homogeneous")

    clusters = {}
    for c in sorted(out["cluster"].unique()):
        sel = out["cluster"] == c
        clusters[int(c)] = {
            "n": int(sel.sum()),
            "mean_volume_mm3": float(df.loc[sel, SIZE_COL].mean()),
            "mean_glcm_entropy": float(df.loc[sel, TEXTURE_COL].mean()),
            "mean_sphericity": float(df.loc[sel, "shape_sphericity"].mean()),
        }
    elbow = {}
    sil = {}
    for kk in range(2, min(9, len(df))):
        m = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit(X)
        elbow[kk] = float(m.inertia_)
        sil[kk] = float(silhouette_score(X, m.labels_))
    corr = df.drop(columns=["case_id"]).corr()
    summary = {
        "n_tumors": len(df),
        "k": k,
        "explained_variance_ratio": pca.explained_variance_ratio_.round(4).tolist(),
        "clusters": clusters,
        "size_vs_texture_pearson_r": float(np.corrcoef(df[SIZE_COL], df[TEXTURE_COL])[0, 1]),
        "texture_median_split": out["texture_label"].value_counts().to_dict(),
        "texture_median": float(med),
        "inertia_by_k": elbow,
        "silhouette_by_k": sil,
        "top_correlated_pairs": _top_pairs(corr, 10),
    }
    return out, summary


def _top_pairs(corr: pd.DataFrame, n: int) -> list:
    cols = corr.columns
    pairs = [(cols[i], cols[j], float(corr.iloc[i, j]))
             for i in range(len(cols)) for j in range(i + 1, len(cols))]
    pairs.sort(key=lambda t: -abs(t[2]))
    return [list(p) for p in pairs[:n]]


def plot(out: pd.DataFrame, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    for c, g in out.groupby("cluster"):
        ax.scatter(g["PC1"], g["PC2"], s=14, label=f"cluster {c} (n={len(g)})")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(); ax.set_title("Radiomics PCA")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("radiomics/analysis"))
    a = ap.parse_args()
    df = pd.read_csv(a.features)
    out, summary = analyse(df, a.k, a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out / "clusters.csv", index=False)
    save_json(summary, a.out / "summary.json")
    plot(out, a.out / "pca_scatter.png")
    for c, s in summary["clusters"].items():
        print(f"cluster {c}: n={s['n']}, volume={s['mean_volume_mm3']:.0f} mm3, "
              f"entropy={s['mean_glcm_entropy']:.2f}")
    print(f"size vs texture r = {summary['size_vs_texture_pearson_r']:.3f}")


if __name__ == "__main__":
    main()
