"""Builds the fitted pipeline.joblib artifact for the Fragrance API.

Run once locally: python build.py

Requires SUPABASE_URL and SUPABASE_KEY (see db.py).
"""
import datetime as dt
import json

import joblib
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize

from db import get_client
from pipeline_def import NicheScorer, NoteFingerprint

ARTIFACT_PATH = "pipeline.joblib"
MIN_DF = 2
PAGE_SIZE = 1000
# One cluster per named subfamily on the Michael Edwards Fragrance Wheel (see
# SUBFAMILY_KEYWORDS below) -- the standard reference wheel this taxonomy follows.
N_FINE_CLUSTERS = 14
NICHE_K_NEIGHBORS = 10
RANDOM_STATE = 42

# The Fragrance Wheel's 4 main families, each split into named subfamilies.
# Each cluster's top notes are rank-weighted-voted against these subfamilies
# and assigned to whichever scores highest, so the "type" a cluster gets is
# derived from its actual notes rather than a run-dependent cluster index.
# Gourmand notes (vanilla, chocolate, coffee, ...) have no family of their own
# on the classic wheel -- they're folded into Soft Oriental / Oriental, where
# gourmand fragrances have historically been classified.
SUBFAMILY_KEYWORDS = {
    "Floral": {
        "rose", "jasmine", "floral notes", "tuberose", "freesia", "lily-of-the-valley",
        "peony", "geranium", "magnolia", "gardenia",
    },
    "Soft Floral": {
        "violet", "iris", "heliotrope", "mimosa", "orris", "powdery notes", "aldehydes",
    },
    "Floral Oriental": {
        "orange blossom", "ylang-ylang", "tuberose", "gardenia", "plumeria", "frangipani",
    },
    "Soft Oriental": {
        "amber", "ambrox", "vanilla", "vanille", "tonka bean", "benzoin", "incense",
    },
    "Oriental": {
        "myrrh", "labdanum", "saffron", "cardamom", "cinnamon", "clove", "nutmeg",
        "chocolate", "coffee", "praline", "caramel", "honey", "almond", "coconut",
    },
    "Woody Oriental": {
        "oud", "agarwood (oud)", "patchouli", "sandalwood", "leather", "pink pepper",
    },
    "Woody": {
        "sandalwood", "cedar", "vetiver", "woody notes", "woodsy notes",
    },
    "Mossy Woods": {
        "oakmoss", "moss", "vetiver", "patchouli",
    },
    "Dry Woods": {
        "birch", "tobacco", "leather", "dry woods", "guaiac wood",
    },
    "Citrus": {
        "citrus", "citruses", "lemon", "lime", "bergamot", "grapefruit", "orange",
        "mandarin", "petitgrain", "neroli",
    },
    "Water": {
        "aquatic", "water notes", "marine", "ozonic", "sea notes",
    },
    "Green": {
        "green notes", "green", "galbanum", "violet leaf", "grass",
    },
    "Aromatic": {
        "lavender", "mint", "rosemary", "sage", "basil", "aromatic", "juniper", "thyme",
    },
    "Fruity": {
        "apple", "peach", "pear", "berries", "black currant", "cassis", "mango",
        "pineapple", "strawberry", "raspberry", "melon", "litchi",
    },
}

SUBFAMILY_TO_MAIN = {
    "Floral": "floral", "Soft Floral": "floral", "Floral Oriental": "floral",
    "Soft Oriental": "oriental", "Oriental": "oriental", "Woody Oriental": "oriental",
    "Woody": "woody", "Mossy Woods": "woody", "Dry Woods": "woody",
    "Citrus": "fresh", "Water": "fresh", "Green": "fresh", "Aromatic": "fresh", "Fruity": "fresh",
}
FAMILY_LABELS = {
    "floral": "Floral",
    "oriental": "Oriental",
    "woody": "Woody",
    "fresh": "Fresh",
}


def _score_subfamilies(top_notes):
    scores = {subfamily: 0.0 for subfamily in SUBFAMILY_KEYWORDS}
    for rank, note in enumerate(top_notes):
        weight = 1.0 / (rank + 1)
        for subfamily, keywords in SUBFAMILY_KEYWORDS.items():
            if note in keywords:
                scores[subfamily] += weight
    return scores


def _assign_subfamilies(cluster_top_notes):
    """Greedily assigns each cluster to a distinct Fragrance Wheel subfamily,
    claiming the strongest-scoring (cluster, subfamily) pairs first. With
    N_FINE_CLUSTERS == len(SUBFAMILY_KEYWORDS) this yields a full one-to-one
    mapping, so every named subfamily gets used exactly once instead of a few
    generic ones (like "Floral") independently winning several clusters' votes.
    """
    scores = {cid: _score_subfamilies(notes) for cid, notes in cluster_top_notes.items()}
    candidates = sorted(
        ((scores[cid][sub], cid, sub) for cid in scores for sub in SUBFAMILY_KEYWORDS),
        key=lambda t: t[0],
        reverse=True,
    )
    assigned = {}
    claimed = set()
    for _, cid, sub in candidates:
        if cid in assigned or sub in claimed:
            continue
        assigned[cid] = sub
        claimed.add(sub)
    return assigned


def compute_clusters(pipeline, corpus_matrix):
    """Groups the corpus into fine-grained note clusters, each labeled with its
    best-matching Fragrance Wheel subfamily (e.g. "Dry Woods") and colored by
    that subfamily's main wheel family (Floral/Oriental/Woody/Fresh), plus a 2D
    layout for plotting. Kept separate from fitting the retrieval pipeline
    since it's for the /clusters visualization, not similarity search.
    """
    vocab = pipeline.named_steps["fingerprint"].vocabulary_
    inv_vocab = {idx: note for note, idx in vocab.items()}

    norm_matrix = normalize(corpus_matrix)

    print("Fitting fine clusters...")
    kmeans = KMeans(n_clusters=N_FINE_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    fine_labels = kmeans.fit_predict(norm_matrix)

    cluster_top_notes = {}
    for cluster_id in range(N_FINE_CLUSTERS):
        center = kmeans.cluster_centers_[cluster_id]
        top_idx = np.argsort(center)[::-1][:6]
        cluster_top_notes[cluster_id] = [inv_vocab[i] for i in top_idx if center[i] > 0]

    subfamily_by_cluster = _assign_subfamilies(cluster_top_notes)

    fine_meta = {}
    for cluster_id in range(N_FINE_CLUSTERS):
        subfamily = subfamily_by_cluster[cluster_id]
        fine_meta[cluster_id] = {
            "label": subfamily,
            "family": SUBFAMILY_TO_MAIN[subfamily],
            "count": int(np.sum(fine_labels == cluster_id)),
        }

    print("Reducing dimensionality for layout...")
    reduced = TruncatedSVD(n_components=30, random_state=RANDOM_STATE).fit_transform(norm_matrix)

    print("Computing 2D layout (t-SNE, this takes a few minutes)...")
    coords = TSNE(
        n_components=2, random_state=RANDOM_STATE, init="pca", perplexity=30
    ).fit_transform(reduced)

    return {
        "fine_cluster": fine_labels.astype(np.int16),
        "x": coords[:, 0].astype(np.float32),
        "y": coords[:, 1].astype(np.float32),
        "fine_meta": fine_meta,
        "family_labels": FAMILY_LABELS,
    }


def load_corpus():
    client = get_client()
    documents = []
    start = 0
    while True:
        page = (
            client.table("perfumes")
            .select("brand,perfume,notes")
            .range(start, start + PAGE_SIZE - 1)
            .execute()
            .data
        )
        if not page:
            break
        documents.extend(page)
        if len(page) < PAGE_SIZE:
            break
        start += PAGE_SIZE
    return [d for d in documents if str(d.get("notes") or "").strip()]


def main():
    print(f"scikit-learn version: {sklearn.__version__}")

    corpus_documents = load_corpus()
    print(f"Corpus size after filtering: {len(corpus_documents)}")
    notes_series = [d["notes"] for d in corpus_documents]

    pipeline = Pipeline([
        ("fingerprint", NoteFingerprint(min_df=MIN_DF)),
        ("niche", NicheScorer(n_clusters=N_FINE_CLUSTERS, k_neighbors=NICHE_K_NEIGHBORS,
                               random_state=RANDOM_STATE)),
    ])
    pipeline.fit(notes_series)
    # pipeline.transform() now returns the niche step's output, not the raw
    # fingerprint vector -- go through the named step for the corpus matrix.
    corpus_matrix = pipeline.named_steps["fingerprint"].transform(notes_series)

    neighbor_index = NearestNeighbors(metric="cosine")
    neighbor_index.fit(corpus_matrix)

    clusters = compute_clusters(pipeline, corpus_matrix)

    bundle = {
        "pipeline": pipeline,
        "neighbor_index": neighbor_index,
        "corpus_matrix": corpus_matrix,
        "corpus_documents": corpus_documents,
        "clusters": clusters,
        "metadata": {
            "steps": [name for name, _ in pipeline.steps],
            "built_at": dt.datetime.utcnow().isoformat() + "Z",
            "sklearn_version": sklearn.__version__,
            "corpus_size": len(corpus_documents),
            "vocab_size": pipeline.named_steps["fingerprint"].vocab_size_,
        },
    }

    joblib.dump(bundle, ARTIFACT_PATH)
    print(f"Wrote {ARTIFACT_PATH}")
    print(json.dumps(bundle["metadata"], indent=2))


if __name__ == "__main__":
    main()
