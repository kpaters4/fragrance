"""Builds the fitted pipeline.joblib artifact for the Scent Fingerprint API.

Run once locally: python build.py
"""
import datetime as dt
import json

import joblib
import pandas as pd
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

from pipeline_def import NoteFingerprint

CORPUS_PATH = "data/perfume_database_cleaned.xlsx"
COLLECTION_PATH = "data/collection.csv"
WISHLIST_PATH = "data/wishlist.csv"
ARTIFACT_PATH = "pipeline.joblib"
VOCAB_SIZE = 150
K_NEIGHBORS = 5


def load_corpus():
    df = pd.read_excel(CORPUS_PATH)
    df = df.dropna(subset=["notes"])
    df = df[df["notes"].astype(str).str.strip() != ""]
    df = df.reset_index(drop=True)
    return df


def score_row(pipeline, neighbor_index, corpus_documents, brand, perfume, notes, k=K_NEIGHBORS):
    vector = pipeline.transform([notes])
    stats = pipeline.named_steps["fingerprint"].describe(notes)
    distances, indices = neighbor_index.kneighbors(vector, n_neighbors=k)
    matches = [
        {
            "brand": corpus_documents[idx]["brand"],
            "perfume": corpus_documents[idx]["perfume"],
            "notes": corpus_documents[idx]["notes"],
            "similarity": float(1 - dist),
        }
        for dist, idx in zip(distances[0], indices[0])
    ]
    return {
        "brand": brand,
        "perfume": perfume,
        "notes": notes,
        "stats": stats,
        "matches": matches,
    }


def main():
    print(f"scikit-learn version: {sklearn.__version__}")

    corpus_df = load_corpus()
    print(f"Corpus size after filtering: {len(corpus_df)}")

    pipeline = Pipeline([("fingerprint", NoteFingerprint(vocab_size=VOCAB_SIZE))])
    pipeline.fit(corpus_df["notes"])
    corpus_matrix = pipeline.transform(corpus_df["notes"])

    neighbor_index = NearestNeighbors(metric="cosine")
    neighbor_index.fit(corpus_matrix)

    corpus_documents = corpus_df[["brand", "perfume", "notes"]].to_dict("records")

    def score_csv(path):
        df = pd.read_csv(path)
        df = df.dropna(subset=["notes"])
        df = df[df["notes"].astype(str).str.strip() != ""]
        return [
            score_row(
                pipeline, neighbor_index, corpus_documents,
                row["brand"], row["perfume"], row["notes"],
            )
            for _, row in df.iterrows()
        ]

    collection = score_csv(COLLECTION_PATH)
    wishlist = score_csv(WISHLIST_PATH)
    print(f"Pre-scored {len(collection)} collection rows, {len(wishlist)} wishlist rows")

    bundle = {
        "pipeline": pipeline,
        "neighbor_index": neighbor_index,
        "corpus_matrix": corpus_matrix,
        "corpus_documents": corpus_documents,
        "collection": collection,
        "wishlist": wishlist,
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
