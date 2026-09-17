"""Custom sklearn transformer for the Fragrance pipeline.

Must be importable identically at build time (build.py) and serve time
(serve.py / modal_serve.py) so joblib can unpickle fitted instances.
"""
import math
from collections import Counter

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def _parse_notes(notes_string):
    if notes_string is None:
        return []
    return [n.strip().lower() for n in str(notes_string).split(",") if n.strip()]


class NoteFingerprint(BaseEstimator, TransformerMixin):
    """IDF-weighted note-presence encoder.

    Vocabulary is every note meeting `min_df` (not a fixed top-K cutoff) so
    long-tail real notes (e.g. "hay", "petrichor") aren't silently dropped
    just for being uncommon -- that uncommonness is exactly the signal a
    niche-fragrance feature needs. Only true one-offs/typos (doc_freq < min_df)
    are excluded.
    """

    def __init__(self, min_df=2):
        self.min_df = min_df

    def fit(self, X, y=None):
        parsed_docs = [_parse_notes(x) for x in X]
        n_documents = len(parsed_docs)

        term_counts = Counter()
        doc_freq = Counter()
        for notes in parsed_docs:
            term_counts.update(notes)
            doc_freq.update(set(notes))

        vocab_notes = [note for note, freq in doc_freq.items() if freq >= self.min_df]
        vocab_notes.sort(key=lambda note: term_counts[note], reverse=True)

        self.vocabulary_ = {note: idx for idx, note in enumerate(vocab_notes)}
        self.idf_ = {
            note: math.log(n_documents / (1 + doc_freq[note]))
            for note in vocab_notes
        }
        self.vocab_size_ = len(vocab_notes)
        return self

    def transform(self, X):
        rows, cols, data = [], [], []
        for row_idx, x in enumerate(X):
            for note in _parse_notes(x):
                col_idx = self.vocabulary_.get(note)
                if col_idx is not None:
                    rows.append(row_idx)
                    cols.append(col_idx)
                    data.append(self.idf_[note])

        n_rows = len(X) if hasattr(X, "__len__") else len(list(X))
        matrix = sp.csr_matrix(
            (data, (rows, cols)), shape=(n_rows, self.vocab_size_), dtype=np.float64
        )
        return matrix

    def describe(self, notes_string):
        notes = _parse_notes(notes_string)
        note_count = len(notes)
        matched_idfs = [self.idf_[n] for n in notes if n in self.vocabulary_]
        known_note_count = len(matched_idfs)
        avg_rarity = float(np.mean(matched_idfs)) if matched_idfs else 0.0
        diversity = (known_note_count / note_count) if note_count else 0.0
        return {
            "note_count": note_count,
            "known_note_count": known_note_count,
            "avg_rarity": avg_rarity,
            "diversity": diversity,
        }


def _mean_nonzero_rows(X):
    """Mean of nonzero entries per row; 0.0 for all-zero rows. For a
    NoteFingerprint-transformed row, nonzero entries ARE the per-note idf
    weights, so this reproduces avg_rarity without a separate idf_ lookup.
    """
    X = sp.csr_matrix(X)
    sums = np.asarray(X.sum(axis=1)).ravel()
    counts = np.diff(X.indptr)
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)


def _percentile_rank(sorted_ref, values):
    """% of sorted_ref <= each value. sorted_ref must be pre-sorted ascending."""
    sorted_ref = np.asarray(sorted_ref)
    if sorted_ref.size == 0:
        return np.zeros(len(values), dtype=float)
    ranks = np.searchsorted(sorted_ref, values, side="right")
    return (ranks / sorted_ref.size) * 100.0


class NicheScorer(BaseEstimator, TransformerMixin):
    """Scores how "niche" a fragrance is relative to the fitted corpus, as a
    single 0-100 percentile combining three signals (each also reported as
    its own percentile): distance to nearest neighbors in scent-space,
    rarity of its individual notes, and how small its assigned cluster is.

    Consumes a NoteFingerprint matrix directly (the natural pipeline input)
    and needs no external cluster labels -- it fits its own internal KMeans
    rather than depending on a separately-fit clustering step, so it composes
    cleanly as Pipeline([("fingerprint", NoteFingerprint()), ("niche", NicheScorer())]).
    """

    def __init__(self, n_clusters=9, k_neighbors=10, random_state=42, weights=(0.4, 0.3, 0.3)):
        self.n_clusters = n_clusters
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.weights = weights

    def fit(self, X, y=None):
        X = sp.csr_matrix(X)
        n_samples = X.shape[0]
        self.n_train_samples_ = n_samples

        self.neighbor_index_ = NearestNeighbors(metric="cosine")
        self.neighbor_index_.fit(X)
        k = min(self.k_neighbors, n_samples - 1)
        distances, _ = self.neighbor_index_.kneighbors(X, n_neighbors=k + 1)
        isolation_raw = distances[:, 1:].mean(axis=1)  # col 0 is always self (dist 0)

        rarity_raw = _mean_nonzero_rows(X)

        norm_X = normalize(X)
        n_clusters = min(self.n_clusters, n_samples)
        self.kmeans_ = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = self.kmeans_.fit_predict(norm_X)
        self.cluster_sizes_ = np.bincount(labels, minlength=n_clusters)
        cluster_rarity_raw = 1.0 - (self.cluster_sizes_[labels] / n_samples)

        self.isolation_ref_ = np.sort(isolation_raw)
        self.rarity_ref_ = np.sort(rarity_raw)
        self.cluster_rarity_ref_ = np.sort(cluster_rarity_raw)
        return self

    def transform(self, X):
        """Works identically for re-scoring training rows or a brand-new
        query row: kneighbors() and kmeans_.predict() both handle held-out
        points natively. Returns a dense (n_rows, 4) array:
        [niche_percentile, isolation_percentile, rarity_percentile, cluster_percentile]
        """
        X = sp.csr_matrix(X)
        k = min(self.k_neighbors, self.n_train_samples_)
        distances, _ = self.neighbor_index_.kneighbors(X, n_neighbors=k)
        isolation_raw = distances.mean(axis=1)

        rarity_raw = _mean_nonzero_rows(X)

        cluster_ids = self.kmeans_.predict(normalize(X))
        cluster_rarity_raw = 1.0 - (self.cluster_sizes_[cluster_ids] / self.n_train_samples_)

        isolation_pct = _percentile_rank(self.isolation_ref_, isolation_raw)
        rarity_pct = _percentile_rank(self.rarity_ref_, rarity_raw)
        cluster_pct = _percentile_rank(self.cluster_rarity_ref_, cluster_rarity_raw)

        w_iso, w_rarity, w_cluster = self.weights
        niche_pct = w_iso * isolation_pct + w_rarity * rarity_pct + w_cluster * cluster_pct

        return np.column_stack([niche_pct, isolation_pct, rarity_pct, cluster_pct])
