# Scent Fingerprint

Finds perfumes with similar scent profiles by comparing their listed notes. Notes are converted into TF-IDF-style vectors and matched against a corpus via cosine nearest-neighbors, so "sandalwood, cardamom, iris" finds perfumes that share the same rare/common note mix rather than just an exact-string match.

## How it works

- `pipeline_def.py` — `NoteFingerprint`, a custom scikit-learn transformer. Splits a comma-separated notes string, builds a vocabulary of the top N most common notes, and weights each by inverse document frequency (rarer notes count more).
- `build.py` — fits `NoteFingerprint` on `data/perfume_database_cleaned.xlsx`, builds a `NearestNeighbors` (cosine) index over the corpus, pre-scores `data/collection.csv` and `data/wishlist.csv` against it, and dumps everything to `pipeline.joblib`.
- `serve.py` — FastAPI app that loads `pipeline.joblib` once at startup and serves it.
- `modal_serve.py` — wraps `serve.py` for deployment on [Modal](https://modal.com).
- `frontend/index.html` — single-page UI that calls the deployed API.

`pipeline_def.py` must be importable identically at build time and serve time — it's how `joblib` unpickles the fitted transformer.

## Setup

```bash
python -m venv venv
venv\Scripts\activate      # or `source venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

## Build the model artifact

```bash
python build.py
```

Reads `data/perfume_database_cleaned.xlsx`, `data/collection.csv`, and `data/wishlist.csv`, and writes `pipeline.joblib`.

## Run the API locally

```bash
uvicorn serve:app --reload
```

### Endpoints

- `GET /health` — liveness check
- `GET /info` — build metadata (corpus size, sklearn version, build timestamp)
- `GET /collection` — pre-scored `data/collection.csv` rows
- `GET /wishlist` — pre-scored `data/wishlist.csv` rows
- `POST /analyze` — score an arbitrary note list against the corpus

  ```json
  { "notes": ["sandalwood", "cardamom", "iris"], "k": 5 }
  ```

A Postman collection for these endpoints is in `postman/scent_fingerprint.postman_collection.json`.

## Deploy

```bash
modal deploy modal_serve.py
```

`SKLEARN_VERSION` in `modal_serve.py` must match `metadata.sklearn_version` in the built `pipeline.joblib` (printed by `build.py`) to avoid unpickling errors.

The frontend (`frontend/index.html`, deployed via Vercel) points at the deployed API through the `API_BASE` constant near the top of the file — update it after redeploying to a new Modal URL.

## Data

- `data/perfume_database_cleaned.xlsx` — corpus of perfumes with `brand`, `perfume`, `notes` columns
- `data/collection.csv` / `data/wishlist.csv` — same shape, your own perfumes to pre-score against the corpus
