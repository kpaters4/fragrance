"""Custom sklearn transformer for the Scent Fingerprint pipeline.

Must be importable identically at build time (build.py) and serve time
(serve.py / modal_serve.py) so joblib can unpickle fitted instances.
"""
import math
from collections import Counter

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin


def _parse_notes(notes_string):
    if notes_string is None:
        return []
    return [n.strip().lower() for n in str(notes_string).split(",") if n.strip()]


class NoteFingerprint(BaseEstimator, TransformerMixin):
    def __init__(self, vocab_size=150):
        self.vocab_size = vocab_size

    def fit(self, X, y=None):
        parsed_docs = [_parse_notes(x) for x in X]
        n_documents = len(parsed_docs)

        term_counts = Counter()
        doc_freq = Counter()
        for notes in parsed_docs:
            term_counts.update(notes)
            doc_freq.update(set(notes))

        top_notes = [note for note, _ in term_counts.most_common(self.vocab_size)]
        self.vocabulary_ = {note: idx for idx, note in enumerate(top_notes)}
        self.idf_ = {
            note: math.log(n_documents / (1 + doc_freq[note]))
            for note in top_notes
        }
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
            (data, (rows, cols)), shape=(n_rows, self.vocab_size), dtype=np.float64
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
