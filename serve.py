"""FastAPI service for the Fragrance API.

Loads the fitted pipeline.joblib bundle once at import time.
"""
from typing import List

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from pipeline_def import NoteFingerprint  # noqa: F401 -- required for joblib unpickling

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
    return _bundle["collection"]


@app.get("/wishlist")
def wishlist():
    if not _artifact_loaded():
        raise HTTPException(status_code=503, detail="artifact not loaded")
    return _bundle["wishlist"]


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    if not _artifact_loaded():
        raise HTTPException(status_code=503, detail="artifact not loaded")

    notes_string = ", ".join(n.strip() for n in request.notes if n.strip())
    if not notes_string:
        raise HTTPException(status_code=422, detail="notes must contain at least one non-empty string")

    pipeline = _bundle["pipeline"]
    neighbor_index = _bundle["neighbor_index"]
    corpus_documents = _bundle["corpus_documents"]

    vector = pipeline.transform([notes_string])
    stats = pipeline.named_steps["fingerprint"].describe(notes_string)

    distances, indices = neighbor_index.kneighbors(vector, n_neighbors=request.k)
    matches = [
        {
            "brand": corpus_documents[idx]["brand"],
            "perfume": corpus_documents[idx]["perfume"],
            "notes": corpus_documents[idx]["notes"],
            "similarity": float(1 - dist),
        }
        for dist, idx in zip(distances[0], indices[0])
    ]

    return {"stats": stats, "matches": matches}
