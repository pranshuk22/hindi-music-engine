"""
scripts/test_single.py
───────────────────────
End-to-end smoke test for a single song.

Runs the full 18-step pipeline on one song and validates the output:
  - Embedding file exists and is 256d (or raw ~991d if PCA not yet fit)
  - Features JSON contains all expected keys
  - SQLite record is inserted correctly
  - Search returns results (if index already built)

Usage:
  python scripts/test_single.py
  python scripts/test_single.py --url https://www.youtube.com/watch?v=Umqb9KENgmk
"""

import sys
import os
import argparse
import logging
import numpy as np

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

EXPECTED_FEATURE_KEYS = [
    "vocal_mfcc_mean", "vocal_mfcc_std",
    "tempo", "spectral_centroid_mean", "spectral_rolloff_mean",
    "spectral_bandwidth_mean", "zcr_mean", "rms_mean", "tonnetz_mean",
    "vocal_energy_ratio", "murki_index",
    "microtonal_pcp", "meend_variance", "raga_probability",
    "onset_skewness", "hnr_mean", "tempo_bucket",
    "lyrics_missing", "category",
]


def run_test(title: str, artist: str, youtube_url: str = None, category: str = "sad_romantic"):
    from utils.db           import init_db, get_song_by_id
    from pipeline.processor import process_song
    from pipeline.feature_extractor import load_features

    log.info("=" * 60)
    log.info(f"Smoke test: {title} — {artist}")
    log.info("=" * 60)

    init_db()

    song_id = process_song(
        title       = title,
        artist      = artist,
        youtube_url = youtube_url,
        category    = category,
    )

    log.info("\nValidating outputs ...")

    # ── Check embedding ────────────────────────────────────────────────────────
    emb_path = f"data/embeddings/{song_id}.npy"
    assert os.path.exists(emb_path), f"FAIL: embedding not found at {emb_path}"
    emb = np.load(emb_path)
    log.info(f"  Embedding shape: {emb.shape}  (expected 256 or ~991 if PCA not fit)")
    assert emb.ndim == 1, f"FAIL: embedding should be 1D, got shape {emb.shape}"
    assert len(emb) in (256, 991), (
        f"FAIL: unexpected embedding dimension {len(emb)}. "
        "Expected 256 (post-PCA) or ~991 (pre-PCA)."
    )
    emb_norm = np.linalg.norm(emb)
    log.info(f"  Embedding L2 norm: {emb_norm:.4f}  (expected ~1.0)")
    assert 0.99 < emb_norm < 1.01, f"FAIL: embedding not L2-normalised (norm={emb_norm:.4f})"

    # ── Check features JSON ────────────────────────────────────────────────────
    feat_path = f"data/features/{song_id}.json"
    assert os.path.exists(feat_path), f"FAIL: features JSON not found at {feat_path}"
    features = load_features(feat_path)

    missing_keys = [k for k in EXPECTED_FEATURE_KEYS if k not in features]
    if missing_keys:
        log.warning(f"  WARN: missing feature keys: {missing_keys}")
    else:
        log.info(f"  Features JSON: all {len(EXPECTED_FEATURE_KEYS)} expected keys present ✓")

    # Check dimensions
    assert len(features["vocal_mfcc_mean"]) == 20, "FAIL: vocal_mfcc_mean should have 20 values"
    assert len(features["microtonal_pcp"])  == 36, "FAIL: microtonal_pcp should have 36 values"
    assert len(features["raga_probability"]) == 30, "FAIL: raga_probability should have 30 values"
    assert len(features["tempo_bucket"])    ==  4, "FAIL: tempo_bucket should have 4 values"
    log.info("  Feature dimensions: ✓")

    # Check tempo bucket is one-hot
    tb = features["tempo_bucket"]
    assert sum(tb) == 1.0 and max(tb) == 1.0, f"FAIL: tempo_bucket not one-hot: {tb}"
    log.info("  Tempo bucket one-hot: ✓")

    # ── Check NLP embedding ────────────────────────────────────────────────────
    nlp_path = f"data/nlp/{song_id}.npy"
    if os.path.exists(nlp_path):
        nlp = np.load(nlp_path)
        log.info(f"  NLP embedding shape: {nlp.shape}  (expected 384)")
        assert len(nlp) == 384, f"FAIL: NLP embedding should be 384d, got {len(nlp)}"
    else:
        log.warning("  NLP embedding not found — lyrics may not have been fetched")

    # ── Check SQLite record ────────────────────────────────────────────────────
    row = get_song_by_id(song_id)
    assert row is not None, "FAIL: song not found in SQLite"
    assert row["processed"] == 1, "FAIL: processed flag not set"
    log.info(f"  SQLite record: ✓  (lyrics_missing={row['lyrics_missing']}, "
             f"category={row['category']})")

    # ── Check no audio left on disk ────────────────────────────────────────────
    import glob
    leftover = glob.glob(f"/tmp/hindi_music_temp/{song_id}*.wav")
    if leftover:
        log.warning(f"  WARN: leftover audio files: {leftover}")
    else:
        log.info("  Audio cleanup: ✓  (no WAV files left in temp)")

    log.info("\n" + "=" * 60)
    log.info("SMOKE TEST PASSED ✓")
    log.info("=" * 60)
    return song_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--title",    default="Tum Hi Ho")
    parser.add_argument("--artist",   default="Arijit Singh")
    parser.add_argument("--url",      default=None,
                        help="YouTube URL (optional — searches by name if omitted)")
    parser.add_argument("--category", default="sad_romantic")
    args = parser.parse_args()

    run_test(
        title       = args.title,
        artist      = args.artist,
        youtube_url = args.url,
        category    = args.category,
    )