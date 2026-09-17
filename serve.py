"""FastAPI service for the Fragrance API.

Loads the fitted pipeline.joblib bundle once at import time. /collection and
/wishlist query Supabase live on every request and score the results against
the loaded pipeline; the corpus itself stays baked into pipeline.joblib since
refitting NoteFingerprint per-request would be wasteful.
"""
import random
from typing import List

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from db import get_client
from pipeline_def import NicheScorer, NoteFingerprint  # noqa: F401 -- required for joblib unpickling

ARTIFACT_PATH = "pipeline.joblib"

app = FastAPI(title="Fragrance API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_bundle = None
_load_error = None

try:
    _bundle = joblib.load(ARTIFACT_PATH)
except Exception as exc:  # noqa: BLE001
    _load_error = str(exc)


def _artifact_loaded():
    return _bundle is not None


Note = Annotated[str, Field(min_length=1, max_length=50)]


class AnalyzeRequest(BaseModel):
    notes: List[Note] = Field(..., min_length=1, max_length=25)
    k: int = Field(default=5, ge=1, le=20)


def _score(notes_string, k=5):
    pipeline = _bundle["pipeline"]
    neighbor_index = _bundle["neighbor_index"]
    corpus_documents = _bundle["corpus_documents"]

    fingerprint = pipeline.named_steps["fingerprint"]
    niche = pipeline.named_steps["niche"]

    # pipeline.transform() now returns the niche step's output, not the raw
    # fingerprint vector -- go through the named step for the match vector.
    vector = fingerprint.transform([notes_string])
    stats = fingerprint.describe(notes_string)

    niche_row = niche.transform(vector)[0]
    stats["niche_score"] = float(niche_row[0])
    stats["niche_percentile_label"] = f"More niche than {stats['niche_score']:.0f}% of the corpus"
    stats["niche_components"] = {
        "isolation_percentile": float(niche_row[1]),
        "rarity_percentile": float(niche_row[2]),
        "cluster_rarity_percentile": float(niche_row[3]),
    }

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
    return stats, matches


def _score_rows(rows, k=5):
    scored = []
    for row in rows:
        notes = (row.get("notes") or "").strip()
        if not notes:
            continue
        stats, matches = _score(notes, k)
        scored.append(
            {
                "brand": row["brand"],
                "perfume": row["perfume"],
                "notes": notes,
                "stats": stats,
                "matches": matches,
            }
        )
    return scored


def _fetch_table(table_name):
    try:
        return get_client().table(table_name).select("brand,perfume,notes").execute().data
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"failed to reach supabase: {exc}") from exc


@app.get("/health")
def health():
    if not _artifact_loaded():
        raise HTTPException(status_code=503, detail="artifact not loaded")
    return {"status": "ok"}


@app.get("/info")
def info():
    if not _artifact_loaded():
        raise HTTPException(status_code=503, detail="artifact not loaded")
    return _bundle["metadata"]


@app.get("/collection")
def collection():
    if not _artifact_loaded():
        raise HTTPException(status_code=503, detail="artifact not loaded")
    return _score_rows(_fetch_table("collection"))


@app.get("/wishlist")
def wishlist():
    if not _artifact_loaded():
        raise HTTPException(status_code=503, detail="artifact not loaded")
    return _score_rows(_fetch_table("wishlist"))


CLUSTER_SAMPLE_SIZE = 4000
_clusters_cache = None


@app.get("/clusters")
def clusters():
    if not _artifact_loaded():
        raise HTTPException(status_code=503, detail="artifact not loaded")

    global _clusters_cache
    if _clusters_cache is None:
        data = _bundle["clusters"]
        corpus_documents = _bundle["corpus_documents"]
        fine_cluster = data["fine_cluster"]
        xs, ys = data["x"], data["y"]
        fine_meta = data["fine_meta"]
        family_labels = data["family_labels"]

        rng = random.Random(42)
        by_fine = {}
        for idx, fine_id in enumerate(fine_cluster):
            by_fine.setdefault(int(fine_id), []).append(idx)

        total = len(corpus_documents)
        points = []
        for fine_id, indices in by_fine.items():
            quota = max(1, round(len(indices) / total * CLUSTER_SAMPLE_SIZE))
            chosen = indices if len(indices) <= quota else rng.sample(indices, quota)
            family = fine_meta[fine_id]["family"]
            for idx in chosen:
                doc = corpus_documents[idx]
                points.append(
                    {
                        "x": round(float(xs[idx]), 3),
                        "y": round(float(ys[idx]), 3),
                        "fine": fine_id,
                        "family": family,
                        "brand": doc["brand"],
                        "perfume": doc["perfume"],
                        "notes": doc["notes"],
                    }
                )

        present_families = {meta["family"] for meta in fine_meta.values()}
        _clusters_cache = {
            "families": [
                {"id": fam, "label": label}
                for fam, label in family_labels.items()
                if fam in present_families
            ],
            "fine_clusters": [
                {"id": fine_id, "label": meta["label"], "family": meta["family"], "count": meta["count"]}
                for fine_id, meta in sorted(fine_meta.items())
            ],
            "points": points,
        }

    return _clusters_cache


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    if not _artifact_loaded():
        raise HTTPException(status_code=503, detail="artifact not loaded")

    notes_string = ", ".join(n.strip() for n in request.notes if n.strip())
    if not notes_string:
        raise HTTPException(status_code=422, detail="notes must contain at least one non-empty string")

    stats, matches = _score(notes_string, request.k)
    return {"stats": stats, "matches": matches}
