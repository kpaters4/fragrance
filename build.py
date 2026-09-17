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
N_FINE_CLUSTERS = 9
NICHE_K_NEIGHBORS = 10
RANDOM_STATE = 42

# Rank-weighted keyword vote: each cluster's top notes are scored against these
# families and assigned to whichever scores highest, so the "type" a cluster
# gets is derived from its actual notes rather than a run-dependent cluster index.
FAMILY_KEYWORDS = {
    "fresh": {
        "citrus", "citruses", "lemon", "lime", "bergamot", "grapefruit", "orange",
        "green notes", "green", "petitgrain", "neroli",
    },
    "aromatic_aquatic": {
        "lavender", "mint", "aquatic", "water notes", "marine", "ozonic",
        "aromatic", "juniper", "rosemary", "sage", "sea notes",
    },
    "floral": {
        "rose", "jasmine", "floral notes", "tuberose", "freesia", "lily-of-the-valley",
        "peony", "violet", "iris", "orange blossom", "ylang-ylang", "heliotrope",
        "geranium", "magnolia", "mimosa",
    },
    "woody": {
        # "musk" is deliberately excluded: it's the single most common note in
        # the corpus, so it shows up in most clusters' top notes regardless of
        # family and would swamp the vote rather than discriminate.
        "oud", "agarwood (oud)", "sandalwood", "cedar", "vetiver", "leather",
        "tobacco", "woody notes", "woodsy notes", "patchouli", "oakmoss", "birch",
    },
    "oriental_amber": {
        "amber", "ambrox", "incense", "benzoin", "myrrh", "labdanum",
        "saffron", "cardamom", "pink pepper",
    },
    "gourmand": {
        "vanilla", "vanille", "tonka bean", "praline", "caramel", "chocolate",
        "honey", "coffee", "almond", "coconut",
    },
}
FAMILY_LABELS = {
    "fresh": "Fresh & Citrus",
    "aromatic_aquatic": "Aromatic & Aquatic",
    "floral": "Floral",
    "woody": "Woody",
    "oriental_amber": "Oriental & Amber",
    "gourmand": "Gourmand",
}


def _clean_note_name(note):
    if "(" in note:
        return note.split("(", 1)[1].rstrip(")").strip().title()
    return note.replace(" notes", "").strip().title()


def _classify_family(top_notes):
    scores = {family: 0.0 for family in FAMILY_KEYWORDS}
    for rank, note in enumerate(top_notes):
        weight = 1.0 / (rank + 1)
        for family, keywords in FAMILY_KEYWORDS.items():
            if note in keywords:
                scores[family] += weight
    return max(scores, key=scores.get)


def compute_clusters(pipeline, corpus_matrix):
    """Groups the corpus into fine-grained note clusters (for descriptive
    labels) folded into 3 broad families (fresh/floral/woody) for coloring,
    plus a 2D layout for plotting. Kept separate from fitting the retrieval
    pipeline since it's for the /clusters visualization, not similarity search.
    """
    vocab = pipeline.named_steps["fingerprint"].vocabulary_
    inv_vocab = {idx: note for note, idx in vocab.items()}

    norm_matrix = normalize(corpus_matrix)

    print("Fitting fine clusters...")
    kmeans = KMeans(n_clusters=N_FINE_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    fine_labels = kmeans.fit_predict(norm_matrix)

    fine_meta = {}
    for cluster_id in range(N_FINE_CLUSTERS):
        center = kmeans.cluster_centers_[cluster_id]
        top_idx = np.argsort(center)[::-1][:6]
        top_notes = [inv_vocab[i] for i in top_idx if center[i] > 0]
        family = _classify_family(top_notes)
        label = " & ".join(_clean_note_name(n) for n in top_notes[:2]) or FAMILY_LABELS[family]
        fine_meta[cluster_id] = {
            "label": label,
            "family": family,
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
