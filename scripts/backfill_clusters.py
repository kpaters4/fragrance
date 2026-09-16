"""One-off: adds cluster data to an existing pipeline.joblib that predates the
/clusters feature, without needing to refetch the corpus from Supabase (the
fitted pipeline and corpus matrix are already baked into the artifact).

Run once: python scripts/backfill_clusters.py
Future rebuilds via build.py compute clusters automatically.
"""
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build import ARTIFACT_PATH, compute_clusters  # noqa: E402
from pipeline_def import NoteFingerprint  # noqa: E402,F401 -- required for joblib unpickling


def main():
    bundle = joblib.load(ARTIFACT_PATH)
    bundle["clusters"] = compute_clusters(bundle["pipeline"], bundle["corpus_matrix"])
    joblib.dump(bundle, ARTIFACT_PATH)
    print(f"Wrote clusters into {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
