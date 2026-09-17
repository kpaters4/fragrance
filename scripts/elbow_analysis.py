"""Sweeps KMeans cluster counts over the corpus's 2D map layout (the same
(x, y) t-SNE coordinates the /clusters map displays -- not the pre-embedding
note vectors) and plots inertia and silhouette score vs. K, to pick a
data-supported cluster count for the map's "emerged clusters" view.

Clustering the 2D layout directly -- rather than the original 1133-dim note
vectors -- is deliberate: it guarantees an "emerged cluster" is always a
real, contiguous region on the map, since there's no second space (KMeans
optimizing one thing, t-SNE optimizing another) left for them to disagree
about. See README for the fuller rationale.

Run once: python scripts/elbow_analysis.py
Requires pipeline.joblib to already have `clusters.x`/`clusters.y` computed
(reuses them rather than re-running t-SNE). Writes docs/elbow-graph.png and
prints the detected elbow K plus the K with the best (sampled) silhouette
score.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from build import ARTIFACT_PATH
from pipeline_def import NoteFingerprint  # noqa: F401 -- required for joblib unpickling

K_RANGE = list(range(4, 51, 2))
RANDOM_STATE = 42
# Silhouette score needs pairwise distances -- O(n^2) over the full ~37k
# corpus is too slow, so it's estimated from a fixed random sample instead.
SILHOUETTE_SAMPLE_SIZE = 3000
OUT_PATH = "docs/elbow-graph.png"


def detect_elbow(ks, inertias):
    """Kneedle-style elbow detection: the point on the inertia curve with the
    largest perpendicular distance from the straight line connecting its
    first and last points.
    """
    ks = np.array(ks, dtype=float)
    inertias = np.array(inertias, dtype=float)
    # Normalize both axes to [0, 1] so the distance metric isn't dominated
    # by inertia's much larger numeric scale.
    ks_n = (ks - ks.min()) / (ks.max() - ks.min())
    inertias_n = (inertias - inertias.min()) / (inertias.max() - inertias.min())
    p1 = np.array([ks_n[0], inertias_n[0]])
    p2 = np.array([ks_n[-1], inertias_n[-1]])
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    distances = []
    for x, y in zip(ks_n, inertias_n):
        point_vec = np.array([x, y]) - p1
        proj_len = np.dot(point_vec, line_vec_norm)
        proj_point = p1 + proj_len * line_vec_norm
        distances.append(np.linalg.norm(np.array([x, y]) - proj_point))
    return int(ks[int(np.argmax(distances))])


def main():
    bundle = joblib.load(ARTIFACT_PATH)
    coords = np.column_stack([bundle["clusters"]["x"], bundle["clusters"]["y"]]).astype(np.float64)

    inertias = []
    silhouettes = []
    for k in K_RANGE:
        print(f"Fitting KMeans k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(coords)
        inertias.append(kmeans.inertia_)
        sil = silhouette_score(
            coords, labels,
            sample_size=SILHOUETTE_SAMPLE_SIZE, random_state=RANDOM_STATE,
        )
        silhouettes.append(sil)
        print(f"  inertia={kmeans.inertia_:.1f}  silhouette={sil:.4f}")

    elbow_k = detect_elbow(K_RANGE, inertias)
    best_silhouette_k = K_RANGE[int(np.argmax(silhouettes))]
    print(f"\nDetected elbow (inertia) at k={elbow_k}")
    print(f"Best silhouette score at k={best_silhouette_k} ({max(silhouettes):.4f})")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(K_RANGE, inertias, marker="o", color="#6d7d6b", label="Inertia")
    ax1.axvline(elbow_k, color="#a3554c", linestyle="--", label=f"elbow: k={elbow_k}")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia (within-cluster sum of squares)", color="#6d7d6b")
    ax1.tick_params(axis="y", labelcolor="#6d7d6b")

    ax2 = ax1.twinx()
    ax2.plot(K_RANGE, silhouettes, marker="s", color="#8a6fae", label="Silhouette score")
    ax2.axvline(
        best_silhouette_k, color="#8a6fae", linestyle=":",
        label=f"best silhouette: k={best_silhouette_k}",
    )
    ax2.set_ylabel("Silhouette score (sampled)", color="#8a6fae")
    ax2.tick_params(axis="y", labelcolor="#8a6fae")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("KMeans elbow curve + silhouette score -- 2D map layout")
    fig.tight_layout()
    plt.savefig(OUT_PATH, dpi=150)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
