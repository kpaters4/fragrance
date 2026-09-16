"""Builds the fitted pipeline.joblib artifact for the Fragrance API.

Run once locally: python build.py

Requires SUPABASE_URL and SUPABASE_KEY (see db.py).
"""
import datetime as dt
import json

import joblib
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

from db import get_client
from pipeline_def import NoteFingerprint

ARTIFACT_PATH = "pipeline.joblib"
VOCAB_SIZE = 150
PAGE_SIZE = 1000


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

    pipeline = Pipeline([("fingerprint", NoteFingerprint(vocab_size=VOCAB_SIZE))])
    pipeline.fit(notes_series)
    corpus_matrix = pipeline.transform(notes_series)

    neighbor_index = NearestNeighbors(metric="cosine")
    neighbor_index.fit(corpus_matrix)

    bundle = {
        "pipeline": pipeline,
        "neighbor_index": neighbor_index,
        "corpus_matrix": corpus_matrix,
        "corpus_documents": corpus_documents,
        "metadata": {
            "steps": [name for name, _ in pipeline.steps],
            "built_at": dt.datetime.utcnow().isoformat() + "Z",
            "sklearn_version": sklearn.__version__,
            "corpus_size": len(corpus_documents),
        },
    }

    joblib.dump(bundle, ARTIFACT_PATH)
    print(f"Wrote {ARTIFACT_PATH}")
    print(json.dumps(bundle["metadata"], indent=2))


if __name__ == "__main__":
    main()
