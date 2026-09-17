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


class LocateRequest(BaseModel):
    query: Annotated[str, Field(min_length=1, max_length=100)]


def _point_for_index(idx, similarity=None):
    """Every field the frontend needs to place and label a single fragrance
    on the map -- its position and both groupings (wheel subfamily, emerged
    cluster) -- looked up from the full corpus, not the ~4,000-point /clusters
    sample. Used both by /locate and to enrich /analyze's (and
    /collection's, /wishlist's) matches, so a popup opened from any of those
    lists can always show where that fragrance actually sits, not just ones
    that happened to land in the map's sample.
    """
    data = _bundle["clusters"]
    doc = _bundle["corpus_documents"][idx]
    subfamily = str(data["wheel_label"][idx])
    point = {
        "x": round(float(data["x"][idx]), 3),
        "y": round(float(data["y"][idx]), 3),
        "wheel": subfamily,
        "emerged": int(data["emerged_cluster"][idx]),
        "family": data["wheel_meta"][subfamily]["family"],
        "brand": doc["brand"],
        "perfume": doc["perfume"],
        "notes": doc["notes"],
    }
    if similarity is not None:
        point["similarity"] = similarity
    return point


def _score(notes_string, k=5):
    pipeline = _bundle["pipeline"]
    neighbor_index = _bundle["neighbor_index"]

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
        _point_for_index(int(idx), similarity=float(1 - dist))
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
    """Two independent groupings of the same sampled points, sharing the same
    (x, y) position for each: `wheel` (every fragrance classified directly
    against the 14 named Fragrance Wheel subfamilies) and `emerged` (which of
    the EMERGED_CLUSTERS data-driven KMeans clusters it landed in, with no
    knowledge of the wheel). See build.py's compute_clusters for how each is
    derived. The frontend toggles which one drives dot coloring/labels; both
    are always present on every point so switching views needs no refetch.
    """
    if not _artifact_loaded():
        raise HTTPException(status_code=503, detail="artifact not loaded")

    global _clusters_cache
    if _clusters_cache is None:
        data = _bundle["clusters"]
        corpus_documents = _bundle["corpus_documents"]
        wheel_label = data["wheel_label"]
        emerged_cluster = data["emerged_cluster"]
        xs, ys = data["x"], data["y"]
        wheel_meta = data["wheel_meta"]
        emerged_meta = data["emerged_meta"]
        family_labels = data["family_labels"]

        rng = random.Random(42)
        # Sampled proportionally by wheel subfamily (not emerged cluster) so
        # every named subfamily keeps visible representation on the map even
        # though the two groupings carve up the corpus differently.
        by_wheel = {}
        for idx, subfamily in enumerate(wheel_label):
            by_wheel.setdefault(str(subfamily), []).append(idx)

        total = len(corpus_documents)
        points = []
        for subfamily, indices in by_wheel.items():
            quota = max(1, round(len(indices) / total * CLUSTER_SAMPLE_SIZE))
            chosen = indices if len(indices) <= quota else rng.sample(indices, quota)
            family = wheel_meta[subfamily]["family"]
            for idx in chosen:
                doc = corpus_documents[idx]
                points.append(
                    {
                        "x": round(float(xs[idx]), 3),
                        "y": round(float(ys[idx]), 3),
                        "wheel": subfamily,
                        "emerged": int(emerged_cluster[idx]),
                        "family": family,
                        "brand": doc["brand"],
                        "perfume": doc["perfume"],
                        "notes": doc["notes"],
                    }
                )

        present_families = {meta["family"] for meta in wheel_meta.values()}
        _clusters_cache = {
            "families": [
                {"id": fam, "label": label}
                for fam, label in family_labels.items()
                if fam in present_families
            ],
            "wheel_subfamilies": [
                {"id": subfamily, "label": meta["label"], "family": meta["family"], "count": meta["count"]}
                for subfamily, meta in wheel_meta.items()
            ],
            "emerged_clusters": [
                {"id": cid, "label": meta["label"], "count": meta["count"]}
                for cid, meta in sorted(emerged_meta.items())
            ],
            "contingency": data["contingency"],
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


def _rank_name_match(doc, q_lower):
    perfume = doc["perfume"].lower()
    brand = doc["brand"].lower()
    if perfume == q_lower:
        return 0
    if perfume.startswith(q_lower):
        return 1
    if brand.startswith(q_lower):
        return 2
    if q_lower in perfume:
        return 3
    return 4  # only the brand contains it, substring mid-word


@app.post("/locate")
def locate(request: LocateRequest):
    """Finds a specific fragrance to show on the map, by name or by notes.
    Name match (brand/perfume) is tried first over the *full* corpus, not
    just the /clusters sample, so any fragrance that exists can be found.
    If nothing matches by name, the query is treated as a notes list instead
    and resolved to the nearest real fragrances by note similarity -- the
    same matching /analyze uses, just returning full corpus rows.
    """
    if not _artifact_loaded():
        raise HTTPException(status_code=503, detail="artifact not loaded")

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be empty")

    corpus_documents = _bundle["corpus_documents"]
    q_lower = query.lower()
    name_hits = [
        idx for idx, doc in enumerate(corpus_documents)
        if q_lower in f"{doc['brand']} {doc['perfume']}".lower()
    ]
    if name_hits:
        name_hits.sort(key=lambda idx: (_rank_name_match(corpus_documents[idx], q_lower), corpus_documents[idx]["perfume"]))
        return {"mode": "name", "results": [_point_for_index(idx) for idx in name_hits[:8]]}

    notes_string = ", ".join(n.strip() for n in query.split(",") if n.strip())
    if not notes_string:
        return {"mode": "name", "results": []}
    stats, matches = _score(notes_string, k=5)
    # A nearest-neighbor search always returns its k closest points, even
    # when the query matched nothing real -- there's no "no match" case in
    # kneighbors() itself. If none of the typed words are notes the corpus
    # actually knows (a typo, a nonexistent fragrance name, gibberish),
    # those "nearest" results are meaningless noise, not a real match, so
    # report no results instead of quietly recommending unrelated fragrances.
    if stats["known_note_count"] == 0:
        return {"mode": "notes", "results": []}
    return {"mode": "notes", "results": matches}
