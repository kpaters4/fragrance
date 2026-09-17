"""Builds the fitted pipeline.joblib artifact for the Fragrance API.

Run once locally: python build.py

Requires SUPABASE_URL and SUPABASE_KEY (see db.py).
"""
import datetime as dt
import json

import joblib
import numpy as np
import scipy.sparse as sp
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
# The map's second, data-driven view: how the corpus's note vectors group on
# their own, independent of the Fragrance Wheel entirely. Picked via
# scripts/elbow_analysis.py (see README) rather than hand-fixed -- KMeans
# inertia and silhouette score both plateau smoothly with no sharp knee
# (perfume notes blend continuously; this corpus doesn't split into a small
# number of naturally separable archetypes), so k=22 was chosen as the
# elbow point: close enough to the wheel's 14 subfamilies that the two views
# stay comparable side by side on the map.
EMERGED_CLUSTERS = 22
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


def _classify_wheel_subfamilies(corpus_matrix, vocab):
    """Classifies every fragrance directly against its own note vector,
    independent of any clustering. Each fragrance is scored against all 14
    subfamily keyword sets -- summing the IDF weight of every one of its own
    notes that's a keyword for that subfamily -- and assigned to whichever
    scores highest.

    This replaced an earlier design that clustered first (KMeans) and then
    tried to name each cluster after the fact by matching its centroid's top
    notes. That indirection broke in two ways: some subfamilies (aquatic,
    dry-wood notes) were almost never any single cluster's dominant
    character, so they got forced onto whatever cluster was left over with
    zero real match; and giving KMeans enough clusters to fix that meant the
    2D map layout could no longer visually resolve them. Classifying each
    fragrance directly against its own notes sidesteps both: it doesn't rely
    on an intermediate unsupervised grouping lining up with a taxonomy it
    was never trying to reproduce, so it's implemented as a single sparse
    matrix multiply (corpus vectors x a keyword indicator matrix) rather
    than a per-cluster loop.
    """
    subfamilies = list(SUBFAMILY_KEYWORDS)
    n_vocab = corpus_matrix.shape[1]
    rows, cols = [], []
    for col, subfamily in enumerate(subfamilies):
        for keyword in SUBFAMILY_KEYWORDS[subfamily]:
            if keyword in vocab:
                rows.append(vocab[keyword])
                cols.append(col)
    keyword_mask = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n_vocab, len(subfamilies))
    )
    scores = np.asarray((corpus_matrix @ keyword_mask).todense())
    best = scores.argmax(axis=1)
    return np.array(subfamilies)[best]


def compute_clusters(pipeline, corpus_matrix):
    """Builds the /clusters map's two independent views of the same corpus:

    - The classic Fragrance Wheel view -- every fragrance classified
      directly against the 14 named subfamilies by its own notes (see
      _classify_wheel_subfamilies).
    - The emerged view -- unsupervised KMeans on the same note vectors, with
      no knowledge of the wheel at all; each cluster is auto-labeled by its
      own top 2 notes (e.g. "Oud & Patchouli") rather than matched to a
      wheel name.

    Both views share the same 2D t-SNE layout, so a fragrance's position on
    the map never changes between them -- only how it's grouped/labeled
    does, which is what makes the two comparable side by side.
    """
    vocab = pipeline.named_steps["fingerprint"].vocabulary_
    inv_vocab = {idx: note for note, idx in vocab.items()}

    norm_matrix = normalize(corpus_matrix)

    print("Classifying fragrances against the Fragrance Wheel...")
    wheel_labels = _classify_wheel_subfamilies(corpus_matrix, vocab)
    wheel_meta = {
        subfamily: {
            "label": subfamily,
            "family": SUBFAMILY_TO_MAIN[subfamily],
            "count": int(np.sum(wheel_labels == subfamily)),
        }
        for subfamily in SUBFAMILY_KEYWORDS
    }

    print("Reducing dimensionality for layout...")
    reduced = TruncatedSVD(n_components=30, random_state=RANDOM_STATE).fit_transform(norm_matrix)

    print("Computing 2D layout (t-SNE, this takes a few minutes)...")
    coords = TSNE(
        n_components=2, random_state=RANDOM_STATE, init="pca", perplexity=30
    ).fit_transform(reduced)

    print(f"Fitting {EMERGED_CLUSTERS} emerged clusters on the 2D layout...")
    # Clustered on the map's own (x, y) coordinates, not the pre-embedding
    # note vectors. KMeans and t-SNE optimize different things (closest
    # centroid vs. preserving local neighbors), so clustering the
    # 1133-dimension vectors let the two disagree about what's "nearby" --
    # a cluster could be perfectly reasonable in note-vector space and still
    # scatter across the whole map, since t-SNE was never told about it.
    # Clustering the 2D output directly makes "emerged cluster" and "a
    # region you can see on the map" the same thing by construction, at the
    # cost of using less information than the full note vectors held (see
    # README). Each cluster's label still comes from real notes: after
    # KMeans assigns 2D memberships, average those members' original
    # fingerprint vectors to get a pseudo-centroid and take its top notes.
    kmeans = KMeans(n_clusters=EMERGED_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    emerged_labels = kmeans.fit_predict(coords)
    emerged_meta = {}
    for cluster_id in range(EMERGED_CLUSTERS):
        member_idx = np.where(emerged_labels == cluster_id)[0]
        pseudo_centroid = np.asarray(corpus_matrix[member_idx].mean(axis=0)).ravel()
        top_idx = np.argsort(pseudo_centroid)[::-1][:2]
        top_notes = [inv_vocab[i].title() for i in top_idx if pseudo_centroid[i] > 0]
        emerged_meta[cluster_id] = {
            "label": " & ".join(top_notes) if top_notes else f"Cluster {cluster_id}",
            "count": int(len(member_idx)),
        }

    # Exact wheel x emerged-cluster crosstab over the full corpus (not the
    # ~4,000-point map sample) -- how many fragrances of each wheel
    # subfamily landed in each emerged cluster. This is what the wheel vs.
    # emerged comparison diagram is built from: a subfamily whose members
    # concentrate into one or two emerged clusters is a real point of
    # agreement between the classic wheel and the data; one spread thin
    # across most of the EMERGED_CLUSTERS is a point of disagreement.
    contingency = {}
    for subfamily, cluster_id in zip(wheel_labels, emerged_labels):
        key = (str(subfamily), int(cluster_id))
        contingency[key] = contingency.get(key, 0) + 1
    contingency_rows = [
        {"wheel": subfamily, "emerged": cluster_id, "count": count}
        for (subfamily, cluster_id), count in contingency.items()
    ]

    return {
        "wheel_label": wheel_labels,
        "emerged_cluster": emerged_labels.astype(np.int16),
        "x": coords[:, 0].astype(np.float32),
        "y": coords[:, 1].astype(np.float32),
        "wheel_meta": wheel_meta,
        "emerged_meta": emerged_meta,
        "family_labels": FAMILY_LABELS,
        "contingency": contingency_rows,
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
