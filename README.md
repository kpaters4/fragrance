# Fragrance

Live site: https://fragrance-azure.vercel.app/

Finds perfumes with similar scent profiles by comparing their listed notes. Notes are converted into TF-IDF-style vectors and matched against a corpus via cosine nearest-neighbors, so "sandalwood, cardamom, iris" finds perfumes that share the same rare/common note mix rather than just an exact-string match.

## The model

The model lives in `pipeline_def.py`, in a custom scikit-learn transformer called `NoteFingerprint`. It turns a perfume's freeform notes string (e.g. `"sandalwood, cardamom, iris, violet, leather, ambrox, musk"`) into a sparse numeric vector, so ordinary vector-similarity search can be used to compare perfumes.

**Fitting (`fit`)** — run once, over the whole corpus (~37,000 perfumes):

1. Every notes string is split on commas, lowercased, and trimmed into a list of individual notes (`"Sandalwood, Iris"` → `["sandalwood", "iris"]`).
2. The transformer counts how often each note appears across the corpus and keeps only the **top 150 most common notes** as its vocabulary (`vocab_size=150`) — rare one-off notes that appear in a single perfume aren't worth a dimension.
3. Each vocabulary note gets an **inverse-document-frequency (IDF) weight**: `log(n_documents / (1 + doc_freq[note]))`. Common notes like "musk" or "vanilla" (appear in thousands of perfumes) get a low weight; distinctive notes like "ambrox" or "oud" (appear in relatively few) get a high weight.

**Transforming (`transform`)** — run per perfume, corpus or query alike:

Each perfume becomes a sparse vector of length 150. For every note in its list that's in the vocabulary, the vector's entry at that note's position is set to the note's IDF weight (everything else is 0). This is deliberately *not* term-frequency weighted — a perfume's note list has no meaningful repeat counts, so presence × rarity is what matters, not frequency.

**Matching** — `build.py` fits `NoteFingerprint` on the corpus, transforms every perfume into its vector, and indexes all of them in a scikit-learn `NearestNeighbors` (cosine metric). A query (an arbitrary note list, or a row from `collection`/`wishlist`) is transformed with the same fitted vocabulary/weights and matched against that index with cosine similarity. Because rare notes dominate the vector's magnitude, two perfumes that share a handful of unusual notes score higher than two that share only ubiquitous ones — which is closer to how someone would describe two fragrances as "similar" than plain note-overlap counting would be.

**Stats** — alongside matches, every scored perfume also gets a `describe()` breakdown:

- `note_count` — how many notes were listed
- `known_note_count` — how many of those are in the fitted vocabulary (i.e. common enough in the corpus to matter)
- `avg_rarity` — mean IDF weight of the known notes (higher = more distinctive note list)
- `diversity` — `known_note_count / note_count` (how much of the listed notes the vocabulary actually "understands")

## The API

`serve.py` is a FastAPI app. On startup it loads `pipeline.joblib` — a bundle (built by `build.py`) containing the fitted `NoteFingerprint`, the fitted `NearestNeighbors` index, and the corpus documents (`brand`/`perfume`/`notes`) the index points into. That bundle is expensive to build (it requires the whole 37k-row corpus) but cheap to load, so it's built once offline and just deserialized at request time — no per-request refitting.

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check; 503 if `pipeline.joblib` failed to load |
| `/info` | GET | Build metadata: corpus size, scikit-learn version, build timestamp |
| `/collection` | GET | Rows from the Supabase `collection` table, scored live against the corpus |
| `/wishlist` | GET | Rows from the Supabase `wishlist` table, scored live against the corpus |
| `/analyze` | POST | Score an arbitrary note list against the corpus |
| `/clusters` | GET | A sampled, 2D-projected view of the corpus grouped into note-based fragrance types, for the clusters visualization |

`/collection` and `/wishlist` query Supabase **live, on every request** — unlike the corpus, they're small (a handful of rows) and change often, so it's cheaper to fetch-then-score them per request than to keep a stale pre-scored copy in `pipeline.joblib`. Both are scored through the same `_score()` helper `/analyze` uses, so the response shape is consistent everywhere: a `stats` object (see above) plus a `matches` array of the top-`k` nearest corpus perfumes, each with `brand`, `perfume`, `notes`, and `similarity` (cosine similarity, 0–1).

`POST /analyze` request body:

```json
{ "notes": ["sandalwood", "cardamom", "iris"], "k": 5 }
```

`notes` must be 1–25 non-empty strings (each ≤ 50 chars); `k` (neighbors to return) defaults to 5 and is capped at 20. Invalid input returns `422`.

A Postman collection covering all five endpoints (including the invalid-input cases) is in `postman/scent_fingerprint.postman_collection.json`; run it with [Newman](https://github.com/postmanlabs/newman) via `newman run postman/scent_fingerprint.postman_collection.json`.

## Code layout

- `pipeline_def.py` — the `NoteFingerprint` model (see above).
- `db.py` — Supabase client factory, used by `build.py` and `serve.py`.
- `build.py` — fits `NoteFingerprint` on the `perfumes` table in Supabase, builds the `NearestNeighbors` index, and dumps it all to `pipeline.joblib`.
- `serve.py` — the FastAPI app (see above).
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
