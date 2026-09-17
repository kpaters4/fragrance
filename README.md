# Fragrance

Live site: https://fragrance-azure.vercel.app/

Finds perfumes with similar scent profiles by comparing their listed notes, groups the whole corpus into families from the Michael Edwards Fragrance Wheel for an interactive cluster map, and scores how "niche" any note list is relative to everything else in the corpus. Notes are converted into TF-IDF-style vectors and matched against a corpus via cosine nearest-neighbors, so "sandalwood, cardamom, iris" finds perfumes that share the same rare/common note mix rather than just an exact-string match.

## The model

The pipeline (`pipeline_def.py`) has two fitted, hand-written scikit-learn steps: `NoteFingerprint` turns a notes string into a vector, and `NicheScorer` scores how niche that vector is relative to the fitted corpus.

### `NoteFingerprint`

Turns a perfume's freeform notes string (e.g. `"sandalwood, cardamom, iris, violet, leather, ambrox, musk"`) into a sparse numeric vector, so ordinary vector-similarity search can be used to compare perfumes.

**Fitting (`fit`)** — run once, over the whole corpus (~37,000 perfumes):

1. Every notes string is split on commas, lowercased, and trimmed into a list of individual notes (`"Sandalwood, Iris"` → `["sandalwood", "iris"]`).
2. The transformer counts how often each note appears across the corpus and keeps every note that appears in **at least `min_df` perfumes** (default `min_df=2`) as its vocabulary — currently ~1,100 notes. This is a document-frequency threshold, not a fixed top-K cutoff, so long-tail real notes (e.g. "hay", "petrichor") aren't silently dropped just for being uncommon — that uncommonness is exactly the signal the niche score below needs. Only true one-offs/typos (appearing in a single perfume) are excluded.
3. Each vocabulary note gets an **inverse-document-frequency (IDF) weight**: `log(n_documents / (1 + doc_freq[note]))`. Common notes like "musk" or "vanilla" (appear in thousands of perfumes) get a low weight; distinctive notes like "ambrox" or "oud" (appear in relatively few) get a high weight.

**Transforming (`transform`)** — run per perfume, corpus or query alike:

Each perfume becomes a sparse vector of length 150. For every note in its list that's in the vocabulary, the vector's entry at that note's position is set to the note's IDF weight (everything else is 0). This is deliberately *not* term-frequency weighted — a perfume's note list has no meaningful repeat counts, so presence × rarity is what matters, not frequency.

**Matching** — `build.py` fits `NoteFingerprint` on the corpus, transforms every perfume into its vector, and indexes all of them in a scikit-learn `NearestNeighbors` (cosine metric). A query (an arbitrary note list, or a row from `collection`/`wishlist`) is transformed with the same fitted vocabulary/weights and matched against that index with cosine similarity. Because rare notes dominate the vector's magnitude, two perfumes that share a handful of unusual notes score higher than two that share only ubiquitous ones — which is closer to how someone would describe two fragrances as "similar" than plain note-overlap counting would be.

**Stats** — alongside matches, every scored perfume also gets a `describe()` breakdown:

- `note_count` — how many notes were listed
- `known_note_count` — how many of those are in the fitted vocabulary (i.e. common enough in the corpus to matter)
- `avg_rarity` — mean IDF weight of the known notes (higher = more distinctive note list)
- `diversity` — `known_note_count / note_count` (how much of the listed notes the vocabulary actually "understands")
- `niche_score`, `niche_percentile_label`, `niche_components` — see `NicheScorer` below

### `NicheScorer`

Composed after `NoteFingerprint` (`Pipeline([("fingerprint", NoteFingerprint()), ("niche", NicheScorer())])`), so it consumes the fingerprint vector directly. Scores how "niche" a fragrance is relative to the fitted corpus by combining three signals, each computed relative to the training corpus and expressed as a percentile so the numbers stay comparable regardless of raw scale:

- **Isolation** — mean cosine distance to its `k` nearest neighbors in scent-space (its own internal fitted `NearestNeighbors`). A note combination that doesn't closely resemble anything already in the database — including one you just typed in that doesn't exist yet — scores high here; a near-duplicate of an existing perfume scores low.
- **Rarity** — mean IDF weight of its matched notes (read directly off the fingerprint vector's nonzero entries, no separate lookup needed).
- **Cluster rarity** — how small its assigned cluster is, from `NicheScorer`'s own internal fitted `KMeans` (`1 - cluster_size / corpus_size`).

`fit()` learns a reference distribution for each of the three signals across the whole corpus; `transform()` converts a query's raw signals into percentiles against those references and combines them into `niche_score` (0–100, e.g. "more niche than 82% of the corpus"). It works identically for corpus rows or a brand-new query vector, since both `NearestNeighbors.kneighbors()` and `KMeans.predict()` handle held-out points natively.

### Clusters (the map)

Built by `compute_clusters()` in `build.py`, separately from the retrieval pipeline above — this is only for the `/clusters` visualization, not similarity search.

1. Fit a `KMeans` with `N_FINE_CLUSTERS=14` on the normalized corpus fingerprint vectors. Fourteen is chosen to match the number of named subfamilies on the **Michael Edwards Fragrance Wheel**, the standard reference used across the perfume industry: 4 main families — Floral, Oriental, Woody, Fresh — each split into 3–5 named subfamilies (`SUBFAMILY_KEYWORDS` in `build.py`), e.g. Floral splits into Floral / Soft Floral / Floral Oriental, Woody splits into Woody / Mossy Woods / Dry Woods.
2. For each of the 14 KMeans clusters, take its top 6 notes by centroid weight and score them (rank-weighted, so the strongest note counts most) against every subfamily's keyword set — e.g. a cluster whose top notes are oud, patchouli, and sandalwood scores highest for **"Woody Oriental"**. Rather than letting each cluster independently pick its own top-scoring subfamily (which let a few generic categories like "Floral" win several clusters' votes and left others unused), `_assign_subfamilies()` claims the single strongest-scoring (cluster, subfamily) pair across the whole 14×14 score grid first, then the next-strongest unclaimed pair, and so on. Since there are exactly 14 clusters and 14 subfamilies, this greedy matching always lands on a full one-to-one assignment — every named subfamily appears exactly once, never zero or duplicated. The main family shown in the legend and used for the dot color (Floral/Oriental/Woody/Fresh) is just a fixed lookup from the assigned subfamily.
3. Separately, `TruncatedSVD` then `t-SNE` reduce the same fingerprint vectors to 2D coordinates purely for plotting — this layout has no bearing on the cluster labels or families, it just lays similar perfumes near each other on the map.

The bundle stores per-perfume `(x, y, fine_cluster_id)` plus each fine cluster's `{label, family, count}`; `GET /clusters` samples this down to ~4,000 points (proportionally per cluster) for the frontend to render. On the map, hovering a family in the legend dims every other family's dots, hovering a subfamily label dims everything outside that one cluster, and clicking a dot (or a result in "Nearest") opens that fragrance's details.

## The API

`serve.py` is a FastAPI app. On startup it loads `pipeline.joblib` — a bundle (built by `build.py`) containing the fitted two-step pipeline (`NoteFingerprint` + `NicheScorer`), a separate fitted `NearestNeighbors` index for similarity search, precomputed cluster/family data for the `/clusters` map, and the corpus documents (`brand`/`perfume`/`notes`) the indexes point into. That bundle is expensive to build (it requires the whole 37k-row corpus, KMeans, and a t-SNE layout) but cheap to load, so it's built once offline and just deserialized at request time — no per-request refitting.

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check; 503 if `pipeline.joblib` failed to load |
| `/info` | GET | Build metadata: corpus size, vocab size, scikit-learn version, build timestamp |
| `/collection` | GET | Rows from the Supabase `collection` table, scored live against the corpus |
| `/wishlist` | GET | Rows from the Supabase `wishlist` table, scored live against the corpus |
| `/analyze` | POST | Score an arbitrary note list against the corpus (similarity matches + niche score) |
| `/clusters` | GET | A sampled, 2D-projected view of the corpus grouped into 14 note-based clusters, each labeled with a named subfamily from the Michael Edwards Fragrance Wheel (e.g. "Dry Woods") and colored by that subfamily's main wheel family (Floral, Oriental, Woody, Fresh), for the clusters visualization |

`/collection` and `/wishlist` query Supabase **live, on every request** — unlike the corpus, they're small (a handful of rows) and change often, so it's cheaper to fetch-then-score them per request than to keep a stale pre-scored copy in `pipeline.joblib`. Both are scored through the same `_score()` helper `/analyze` uses, so the response shape is consistent everywhere: a `stats` object (see above) plus a `matches` array of the top-`k` nearest corpus perfumes, each with `brand`, `perfume`, `notes`, and `similarity` (cosine similarity, 0–1).

`POST /analyze` request body:

```json
{ "notes": ["sandalwood", "cardamom", "iris"], "k": 5 }
```

`notes` must be 1–25 non-empty strings (each ≤ 50 chars); `k` (neighbors to return) defaults to 5 and is capped at 20. Invalid input returns `422`.

A Postman collection covering all six endpoints (including the invalid-input cases) is in `postman/scent_fingerprint.postman_collection.json`; run it with [Newman](https://github.com/postmanlabs/newman) via `newman run postman/scent_fingerprint.postman_collection.json`.

Run against the deployed Modal URL:

| `POST /analyze` — valid (200) | `POST /analyze` — invalid, empty notes (422) |
|---|---|
| ![valid /analyze response](postman/screenshots/analyze_valid_200.jpg) | ![invalid /analyze response](postman/screenshots/analyze_invalid_422.jpg) |

## Code layout

- `pipeline_def.py` — the `NoteFingerprint` and `NicheScorer` models (see above).
- `db.py` — Supabase client factory, used by `build.py` and `serve.py`.
- `build.py` — fits the `NoteFingerprint` → `NicheScorer` pipeline on the `perfumes` table in Supabase, builds the `NearestNeighbors` index and cluster/family layout, and dumps it all to `pipeline.joblib`.
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
