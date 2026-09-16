# Fragrance

Live site: https://fragrance-azure.vercel.app/

Finds perfumes with similar scent profiles by comparing their listed notes. Notes are converted into TF-IDF-style vectors and matched against a corpus via cosine nearest-neighbors, so "sandalwood, cardamom, iris" finds perfumes that share the same rare/common note mix rather than just an exact-string match.

## How it works

- `pipeline_def.py` — `NoteFingerprint`, a custom scikit-learn transformer. Splits a comma-separated notes string, builds a vocabulary of the top N most common notes, and weights each by inverse document frequency (rarer notes count more).
- `db.py` — Supabase client factory, used by `build.py` and `serve.py`.
- `build.py` — fits `NoteFingerprint` on the `perfumes` table in Supabase, builds a `NearestNeighbors` (cosine) index over the corpus, and dumps it all to `pipeline.joblib`.
- `serve.py` — FastAPI app that loads `pipeline.joblib` once at startup. `/collection` and `/wishlist` query the `collection`/`wishlist` tables in Supabase live on every request and score the rows against the loaded pipeline.
- `modal_serve.py` — wraps `serve.py` for deployment on [Modal](https://modal.com).
- `frontend/index.html` — single-page UI that calls the deployed API.

`pipeline_def.py` must be importable identically at build time and serve time — it's how `joblib` unpickles the fitted transformer.

## Setup

```bash
python -m venv venv
venv\Scripts\activate      # or `source venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

Set `SUPABASE_URL` and `SUPABASE_KEY` (a secret key, not the publishable key — `build.py` and `serve.py` run server-side only) in the environment before running `build.py` or `serve.py` locally.

## Build the model artifact

```bash
python build.py
```

Fits the corpus pipeline from the `perfumes` table in Supabase and writes `pipeline.joblib`. Re-run this after the corpus changes; `collection`/`wishlist` don't need a rebuild since those are queried live.

## Run the API locally

```bash
uvicorn serve:app --reload
```

### Endpoints

- `GET /health` — liveness check
- `GET /info` — build metadata (corpus size, sklearn version, build timestamp)
- `GET /collection` — rows from the Supabase `collection` table, scored live against the corpus
- `GET /wishlist` — rows from the Supabase `wishlist` table, scored live against the corpus
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

`modal_serve.py` reads Supabase credentials from a Modal secret named `fragrance-supabase`, containing `SUPABASE_URL` and `SUPABASE_KEY`:

```bash
modal secret create fragrance-supabase SUPABASE_URL=... SUPABASE_KEY=...
```

The frontend (`frontend/index.html`, deployed via Vercel) points at the deployed API through the `API_BASE` constant near the top of the file — update it after redeploying to a new Modal URL.

## Data

Data lives in Supabase, in three tables (`brand`, `perfume`, `notes` columns): `perfumes` (the corpus), `collection`, and `wishlist` (your own perfumes).

The `data/` directory holds the original CSV/XLSX files used to seed those tables — they aren't read at runtime. To (re)load Supabase from them:

```bash
python scripts/seed_supabase.py            # all three tables
python scripts/seed_supabase.py perfumes   # just one
```
