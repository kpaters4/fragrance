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
# Clusters used only for NicheScorer's "cluster rarity" signal (how small a
# fragrance's neighborhood is) -- unrelated to the /clusters map below, kept
# as its own constant so retuning the map's resolution never silently
# changes niche scoring.
NICHE_CLUSTERS = 14
# The map's fine-grained clustering. Deliberately more than the 14 named
# Fragrance Wheel subfamilies (see SUBFAMILY_KEYWORDS below) -- some
# subfamilies (aquatic, fruity, dry-wood notes) are rarely any single
# cluster's dominant character at coarser resolution, so a strict 14-cluster
# / 14-subfamily bijection forced a few clusters into names that didn't
# match their actual notes at all. At this resolution every subfamily has a
# real, distinctly-matching cluster; see _assign_subfamilies for how a
# popular subfamily (e.g. Floral) ends up covering several clusters while a
# rarer one (e.g. Water) still gets exactly the one it best matches.
MAP_FINE_CLUSTERS = 40
# How many of a cluster's top centroid-weighted notes get scored against the
# subfamily keyword sets when labeling it.
TOP_NOTES_FOR_LABELING = 10
NICHE_K_NEIGHBORS = 10
RANDOM_STATE = 42

# The Fragrance Wheel's 4 main families, each split into named subfamilies.
# Each cluster's top notes are rank-weighted-voted against these subfamilies
# and assigned to whichever scores highest, so the "type" a cluster gets is
# derived from its actual notes rather than a run-dependent cluster index.
# This follows the 2010 revision of the wheel, where Edwards renamed the
# "Oriental" family to "Amber" (Floral Oriental/Soft Oriental/Oriental/Woody
# Oriental became Floral Amber/Soft Amber/Amber/Woody Amber) and "Aquatic"
# became "Water". Gourmand notes (vanilla, chocolate, coffee, ...) have no
# family of their own on the wheel -- they're folded into Soft Amber / Amber,
# where gourmand fragrances have historically been classified.
SUBFAMILY_KEYWORDS = {
    "Floral": {
        "rose", "bulgarian rose", "damask rose", "turkish rose", "jasmine", "jasmine sambac",
        "tuberose", "freesia", "lily-of-the-valley", "lily", "peony", "geranium", "magnolia",
        "gardenia", "carnation", "lilac", "hyacinth", "narcissus", "wisteria", "camellia",
        "sweet pea", "honeysuckle", "floral notes",
    },
    "Soft Floral": {
        "violet", "iris", "iris flower", "iris petals", "orris", "orris root", "heliotrope",
        "mimosa", "aldehydes", "powdery notes", "cyclamen",
    },
    "Floral Amber": {
        "orange blossom", "tunisian orange blossom", "french orange flower",
        "african orange flower", "valencia orange flower", "ylang-ylang",
        "madagascar ylang-ylang", "tuberose", "egyptian tuberose", "indian tuberose",
        "gardenia", "tahitian gardenia", "frangipani", "tiare flower", "champaca",
        "yellow champaca",
    },
    "Soft Amber": {
        "amber", "ambroxan", "vanilla", "vanille", "bourbon vanilla", "madagascar vanilla",
        "tahitian vanilla", "mexican vanilla", "natural vanilla", "tonka bean", "benzoin",
        "siam benzoin", "incense", "opoponax",
    },
    "Amber": {
        "amber", "myrrh", "labdanum", "french labdanum", "spanish labdanum", "saffron",
        "indian saffron", "cardamom", "black cardamom", "white cardamom",
        "guatemalan cardamom", "cinnamon", "ceylon cinnamon", "clove", "cloves", "nutmeg",
        "indonesian nutmeg", "chocolate", "dark chocolate", "mexican chocolate", "coffee",
        "roasted coffee beans", "praline", "caramel", "honey", "almond", "coconut", "resin",
        "resins", "elemi", "styrax", "olibanum",
    },
    "Woody Amber": {
        "agarwood (oud)", "cambodian oud", "indian oud", "laotian oud", "thailand oud",
        "white oud", "patchouli", "indian patchouli", "indonesian patchouli leaf",
        "singapore patchouli", "sandalwood", "madagascar sandalwood", "leather",
        "russian leather", "suede", "white suede", "pink pepper", "castoreum",
    },
    "Woods": {
        "sandalwood", "australian sandalwood", "madagascar sandalwood", "red sandalwood",
        "white sandalwood", "cedar", "atlas cedar", "virginia cedar", "texas cedar",
        "himalayan cedar", "chinese cedar", "moroccan cedar", "red cedar", "vetiver",
        "haitian vetiver", "bourbon vetiver", "madagascar vetiver", "java vetiver oil",
        "tahitian vetiver", "woody notes", "woodsy notes", "cashmeran",
    },
    "Mossy Woods": {
        "oakmoss", "oak moss", "serbian oakmoss", "moss", "cedarmoss", "patchouli", "vetiver",
    },
    "Dry Woods": {
        "birch", "birch leaf", "tobacco", "tobacco leaf", "tobacco blossom",
        "bulgarian light tobacco", "white tobacco", "leather", "guaiac wood", "smoke",
        "gunpowder", "ash",
    },
    "Citrus": {
        "citruses", "sicilian citrus", "sicilian citruses", "japanese citruses", "lemon",
        "sicilian lemon", "italian lemon", "amalfi lemon", "argentinian lemon",
        "californian lemon", "moroccan lemon", "lemon zest", "lemon peel", "lime", "bergamot",
        "calabrian bergamot", "sicilian bergamot", "white bergamot", "grapefruit",
        "pink grapefruit", "blood grapefruit", "florida grapefruit", "white grapefruit",
        "mandarin orange", "petitgrain", "petitgrain paraguay", "neroli", "neroli essence",
        "yuzu", "tangerine", "clementine", "pomelo", "bitter orange",
    },
    "Water": {
        "water notes", "watery notes", "sea notes", "sea water", "sea salt", "seagrass",
        "seashells", "seaweed", "ozonic notes", "rain notes", "calone", "salt", "cucumber",
        "watercress", "solar notes",
    },
    "Green": {
        "green notes", "green accord", "green leaves", "grass", "green grass", "galbanum",
        "violet leaf", "violet leaves", "fig leaf", "tomato leaf", "ivy", "nettle",
        "bamboo leaf",
    },
    "Aromatic": {
        "lavender", "wild lavender", "blue lavender", "lavender extract", "mint",
        "water mint", "spicy mint", "rosemary", "sage", "clary sage", "blue sage",
        "egyptian sage", "basil", "basil leaf", "israeli basil", "black basil", "thyme",
        "red thyme", "tulsi", "marjoram", "oregano", "juniper", "juniper berries",
        "juniper berry", "aromatic spices", "artemisia", "tarragon",
    },
    "Fruity": {
        "apple", "green apple", "red apple", "granny smith apple", "pear", "pear blossom",
        "peach", "white peach", "red peach", "peach blossom", "apricot", "white apricot",
        "apricot blossom", "plum", "mirabelle plum", "damask plum", "yellow plum",
        "chinese plum", "japanese plum", "cherry", "sour cherry", "white cherry",
        "maraschino cherry", "red berries", "wild berries", "forest fruits", "berry fruits",
        "black currant", "cassis", "red currant", "white currant", "mango", "pineapple",
        "strawberry", "wild strawberry", "raspberry", "litchi", "chinese litchi",
        "pink litchi", "red litchi", "melon", "italian melon", "frosted melon", "watermelon",
        "guava", "papaya", "passionfruit", "pomegranate", "quince", "banana", "kiwi",
        "dried fruits", "tropical fruit", "tropical fruits", "exotic fruits", "gooseberry",
        "cranberry", "blueberry", "blackberry",
    },
}

SUBFAMILY_TO_MAIN = {
    "Floral": "floral", "Soft Floral": "floral", "Floral Amber": "floral",
    "Soft Amber": "amber", "Amber": "amber", "Woody Amber": "amber",
    "Woods": "woody", "Mossy Woods": "woody", "Dry Woods": "woody",
    "Citrus": "fresh", "Water": "fresh", "Green": "fresh", "Aromatic": "fresh", "Fruity": "fresh",
}
FAMILY_LABELS = {
    "floral": "Floral",
    "amber": "Amber",
    "woody": "Woody",
    "fresh": "Fresh",
}


def _score_subfamilies(top_notes):
    scores = {subfamily: 0.0 for subfamily in SUBFAMILY_KEYWORDS}
    for rank, note in enumerate(top_notes):
        weight = 1.0 / (rank + 1)
        for subfamily, keywords in SUBFAMILY_KEYWORDS.items():
            if note in keywords:
                scores[subfamily] += weight
    return scores


def _assign_subfamilies(cluster_top_notes):
    """Assigns each cluster to its best-matching Fragrance Wheel subfamily, in
    two passes:

    1. Coverage pass -- same greedy claiming as a strict one-to-one mapping
       (strongest-scoring (cluster, subfamily) pairs claimed first), but only
       until every subfamily has claimed exactly one cluster. This guarantees
       every named subfamily is used at least once, by its single best-fitting
       cluster out of all of them -- even a rare subfamily like Water still
       gets a real match rather than being forced onto whatever's left over.
    2. Free pass -- every remaining, unclaimed cluster independently picks
       whichever subfamily scores highest for it, with no uniqueness
       constraint. A broad subfamily like Floral or Amber naturally ends up
       covering several clusters this way, since the corpus has many
       distinct flavors of it; a narrow one just doesn't gain any more.
    """
    scores = {cid: _score_subfamilies(notes) for cid, notes in cluster_top_notes.items()}
    candidates = sorted(
        ((scores[cid][sub], cid, sub) for cid in scores for sub in SUBFAMILY_KEYWORDS),
        key=lambda t: t[0],
        reverse=True,
    )
    assigned = {}
    claimed = set()
    for _, cid, sub in candidates:
        if len(claimed) == len(SUBFAMILY_KEYWORDS):
            break
        if cid in assigned or sub in claimed:
            continue
        assigned[cid] = sub
        claimed.add(sub)
    for cid, subfamily_scores in scores.items():
        if cid not in assigned:
            assigned[cid] = max(subfamily_scores, key=subfamily_scores.get)
    return assigned


def compute_clusters(pipeline, corpus_matrix):
    """Groups the corpus into fine-grained note clusters, each labeled with its
    best-matching Fragrance Wheel subfamily (e.g. "Dry Woods") and colored by
    that subfamily's main wheel family (Floral/Amber/Woody/Fresh), plus a 2D
    layout for plotting. Kept separate from fitting the retrieval pipeline
    since it's for the /clusters visualization, not similarity search.
    """
    vocab = pipeline.named_steps["fingerprint"].vocabulary_
    inv_vocab = {idx: note for note, idx in vocab.items()}

    norm_matrix = normalize(corpus_matrix)

    print("Fitting fine clusters...")
    kmeans = KMeans(n_clusters=MAP_FINE_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    fine_labels = kmeans.fit_predict(norm_matrix)

    cluster_top_notes = {}
    for cluster_id in range(MAP_FINE_CLUSTERS):
        center = kmeans.cluster_centers_[cluster_id]
        top_idx = np.argsort(center)[::-1][:TOP_NOTES_FOR_LABELING]
        cluster_top_notes[cluster_id] = [inv_vocab[i] for i in top_idx if center[i] > 0]

    subfamily_by_cluster = _assign_subfamilies(cluster_top_notes)

    fine_meta = {}
    for cluster_id in range(MAP_FINE_CLUSTERS):
        subfamily = subfamily_by_cluster[cluster_id]
        fine_meta[cluster_id] = {
            "label": subfamily,
            "family": SUBFAMILY_TO_MAIN[subfamily],
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
        ("niche", NicheScorer(n_clusters=NICHE_CLUSTERS, k_neighbors=NICHE_K_NEIGHBORS,
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
