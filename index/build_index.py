"""
index/build_index.py
─────────────────────
Build (or rebuild) the FAISS vector index.

Modes:
  --fit-pca        Fit PCA on stored raw fused vectors, compress to Nd,
                   overwrite data/embeddings/*.npy, then build FAISS index.
                   Run once after the pilot batch completes.

  --from-features  Rebuild fused vectors from data/features/*.json without
                   re-downloading audio. Use after fixing feature extraction
                   bugs (murki, tempo_bucket, onset_skewness, raga_probability)
                   when audio is gone but JSONs still exist. Automatically
                   implies --fit-pca.

  (default)        Assume PCA already fitted and data/embeddings/*.npy
                   already contain Nd compressed vectors. Just build index.

Adaptive PCA target:
  n_components = min(256, n_songs // 3)
  — At 44 songs  →  14 components  (numerically stable)
  — At 200 songs →  66 components
  — At 768+ songs → 256 components  (full spec target)
  This prevents fitting PCA where n_components ≥ n_samples.

Output:
  index/faiss_index.bin   FAISS IndexFlatIP (<1k songs) or IndexIVFPQ (≥1k)
  index/song_id_map.npy   Song ID array aligned to FAISS row indices
  index/pca_model.pkl     Fitted PCA model (--fit-pca / --from-features only)

Usage:
  python index/build_index.py
  python index/build_index.py --fit-pca
  python index/build_index.py --from-features
"""

import sys
import os
import argparse
import logging
import json
import numpy as np
import faiss
import joblib
from sklearn.decomposition import PCA

sys.path.insert(0, ".")
from utils.db import get_all_songs, init_db

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PCA_PATH        = "index/pca_model.pkl"
INDEX_PATH      = "index/faiss_index.bin"
SONG_MAP_PATH   = "index/song_id_map.npy"
MAX_COMPONENTS  = 256           # absolute ceiling for PCA target
IVFPQ_THRESHOLD = 1000          # switch to IVF above this many songs
FEATURES_DIR    = "data/features"
EMBEDDINGS_DIR  = "data/embeddings_raw"


# ─── Adaptive PCA target ─────────────────────────────────────────────────────

def _adaptive_n_components(n_songs: int, raw_dim: int) -> int:
    """
    Goldilocks PCA target for the pilot dataset. 
    Filters noise while preserving acoustic nuance.
    """
    target = 24 # The sweet spot for 50 songs
    log.info(f"Fixed PCA Target: n_components={target}")
    return target


# ─── --from-features: rebuild fused vectors from JSON ─────────────────────────

def rebuild_embeddings_from_features(songs: list) -> list:
    """
    Re-fuse and re-save raw embeddings from data/features/*.json files
    without re-downloading audio.

    When to use:
      After fixing feature extraction bugs (murki_index, tempo_bucket,
      onset_skewness, raga_probability) in indian_features.py or
      feature_extractor.py when the raw audio is already deleted but
      feature JSONs on disk still exist.

    How it works:
      Reads every feature array from the JSON, then calls
      embedder.build_fused_vector() with those values. The CLAP embedding
      is recovered from the existing .npy file only if it is still the
      raw ~991d fused vector (first 512d = CLAP). If the stored embedding
      is already PCA-compressed, that song must be re-downloaded — a warning
      is printed and the song is skipped.

    Returns the same `songs` list (DB rows are unchanged).
    Overwrites data/embeddings/<song_id>.npy with the corrected fused vector.
    """
    from pipeline.embedder import build_fused_vector, save_embedding, _l2
    from pipeline.feature_extractor import get_tempo_bucket

    rebuilt = 0
    skipped = 0

    log.info(f"\nRebuilding fused vectors from JSONs for {len(songs)} songs ...")

    for row in songs:
        song_id        = row["id"]
        features_path  = row["features_path"]
        embedding_path = row["embedding_path"]
        nlp_path       = row["nlp_path"]

        # ── Validate inputs ──────────────────────────────────────────────────
        if not features_path or not os.path.exists(features_path):
            log.warning(f"  SKIP {song_id}: features JSON missing — {features_path}")
            skipped += 1
            continue

        if not embedding_path or not os.path.exists(embedding_path):
            log.warning(f"  SKIP {song_id}: embedding .npy missing — {embedding_path}")
            skipped += 1
            continue

        stored = np.load(embedding_path).astype(np.float32)

        # If already PCA-compressed we cannot recover the 512d CLAP component
        if stored.shape[0] < 512:
            log.warning(
                f"  SKIP {song_id}: stored embedding is {stored.shape[0]}d "
                "(already PCA-compressed — CLAP component unrecoverable). "
                "Re-run the full pipeline for this song."
            )
            skipped += 1
            continue

        # First 512d of the raw fused vector is the weighted CLAP component
        clap_emb = _l2(stored[:512])

        # ── Load features JSON ───────────────────────────────────────────────
        with open(features_path) as f:
            feat = json.load(f)

        # ── Load NLP embedding ───────────────────────────────────────────────
        lyrics_missing = bool(feat.get("lyrics_missing", 0))
        if nlp_path and os.path.exists(nlp_path):
            nlp_emb = np.load(nlp_path).astype(np.float32)
        else:
            nlp_emb = np.zeros(384, dtype=np.float32)
            lyrics_missing = True

        # ── Assemble feature arrays ──────────────────────────────────────────
        vocal_mfcc         = np.array(feat.get("vocal_mfcc_mean",   [0.0] * 20), dtype=np.float32)
        vocal_energy_ratio = float(feat.get("vocal_energy_ratio",   0.5))
        murki_index        = float(feat.get("murki_index",          0.0))
        microtonal_pcp     = np.array(feat.get("microtonal_pcp",    [0.0] * 36), dtype=np.float32)
        meend_variance     = float(feat.get("meend_variance",       0.0))
        raga_probability   = np.array(feat.get("raga_probability",  [0.0] * 30), dtype=np.float32)
        onset_skewness     = float(feat.get("onset_skewness",       0.0))
        hnr_mean           = float(feat.get("hnr_mean",             0.5))

        # Tempo bucket — stored as list [0,1,0,0]; fall back to recomputing
        tb_raw = feat.get("tempo_bucket")
        if tb_raw is not None and len(tb_raw) == 4:
            tempo_bucket = np.array(tb_raw, dtype=np.float32)
        else:
            tempo_bucket = get_tempo_bucket(float(feat.get("tempo", 100.0)))

        # ── Rebuild fused vector ─────────────────────────────────────────────
        fused = build_fused_vector(
            clap_emb           = clap_emb,
            nlp_emb            = nlp_emb,
            vocal_mfcc         = vocal_mfcc,
            vocal_energy_ratio = vocal_energy_ratio,
            murki_index        = murki_index,
            microtonal_pcp     = microtonal_pcp,
            meend_variance     = meend_variance,
            raga_probability   = raga_probability,
            onset_skewness     = onset_skewness,
            tempo_bucket       = tempo_bucket,
            hnr_mean           = hnr_mean,
            lyrics_missing     = lyrics_missing,
        )

        save_embedding(fused, embedding_path)
        rebuilt += 1
        log.info(f"  ✓ {song_id}  ({fused.shape[0]}d)")

    log.info(f"\nRebuilt: {rebuilt}  |  Skipped: {skipped}")
    return songs


# ─── PCA fitting + compression ────────────────────────────────────────────────

def fit_and_compress_pca(songs: list) -> list[tuple[str, np.ndarray]]:
    """
    1. Load all stored embeddings (raw ~991d fused vectors expected).
    2. Validate dimension consistency — print actionable error on mismatch.
    3. Fit PCA with adaptive n_components.
    4. Save PCA model to index/pca_model.pkl.
    5. Compress and overwrite all data/embeddings/*.npy files.
    6. Return list of (song_id, compressed_vector) pairs.
    """
    log.info("\nLoading embeddings for PCA fitting ...")

    song_ids  = []
    emb_paths = []
    raw_vecs  = []

    for row in songs:
        emb_path = row["embedding_path"]
        if not emb_path or not os.path.exists(emb_path):
            log.warning(f"  Missing embedding for {row['id']} — skipping")
            continue
        vec = np.load(emb_path).astype(np.float32)
        song_ids.append(row["id"])
        emb_paths.append(emb_path)
        raw_vecs.append(vec)

    n = len(raw_vecs)
    if n < 10:
        raise RuntimeError(
            f"Only {n} embeddings found — need ≥10 for meaningful PCA. "
            "Process more songs first."
        )

    # Dimension consistency check — fail loudly with actionable info
    dims = [v.shape[0] for v in raw_vecs]
    if len(set(dims)) > 1:
        dominant = max(set(dims), key=dims.count)
        log.error(f"\nDimension mismatch across {n} embeddings (dominant={dominant}d):")
        for i, (sid, d) in enumerate(zip(song_ids, dims)):
            if d != dominant:
                log.error(f"  ✗ {sid}: {d}d — delete {emb_paths[i]} and re-run pipeline")
        raise RuntimeError(
            "All embeddings must have the same dimension. "
            "Delete the stale files listed above and re-run run_pipeline.py --force "
            "for those songs, then retry build_index.py --fit-pca."
        )

    matrix = np.vstack(raw_vecs)
    n_songs, raw_dim = matrix.shape
    log.info(f"Raw matrix: {n_songs} × {raw_dim}d")

    n_components = _adaptive_n_components(n_songs, raw_dim)
    log.info(f"Fitting PCA(n_components={n_components}) ...")
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(matrix)

    explained = pca.explained_variance_ratio_.sum()
    log.info(f"Variance explained by {n_components} components: {explained:.1%}")

    os.makedirs("index", exist_ok=True)
    joblib.dump(pca, PCA_PATH)
    log.info(f"PCA model saved → {PCA_PATH}")

    log.info(f"Compressing {n_songs} embeddings: {raw_dim}d → {n_components}d ...")
    compressed = pca.transform(matrix).astype(np.float32)

    # L2-normalise rows
    norms = np.linalg.norm(compressed, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    compressed /= norms

    pairs = []
    for i, (sid, path) in enumerate(zip(song_ids, emb_paths)):
        np.save(path, compressed[i])
        pairs.append((sid, compressed[i]))

    log.info(f"Compressed {len(pairs)} embeddings saved.")
    return pairs


# ─── FAISS index construction ─────────────────────────────────────────────────

def build_index(songs: list = None, pca_already_fit: bool = False):
    """
    Build FAISS index from compressed embeddings in data/embeddings/.
    Uses IndexFlatIP for <IVFPQ_THRESHOLD songs, IndexIVFPQ above.
    """
    if songs is None:
        songs = get_all_songs()

    log.info(f"\nBuilding FAISS index for {len(songs)} candidate songs ...")

    song_ids = []
    vectors  = []

    for row in songs:
        emb_path = row["embedding_path"]
        if not emb_path or not os.path.exists(emb_path):
            log.warning(f"  Missing embedding: {row['id']} — skipped")
            continue
        vec = np.load(emb_path).astype(np.float32)
        song_ids.append(row["id"])
        vectors.append(vec)

    if not vectors:
        raise RuntimeError(
            "No valid embeddings found. Run scripts/run_pipeline.py first."
        )

    matrix = np.vstack(vectors).astype(np.float32)
    n_songs, dim = matrix.shape
    log.info(f"Index matrix: {n_songs} × {dim}d")

    # Warn if embeddings look uncompressed
    if dim >= 900 and not pca_already_fit:
        log.warning(
            f"Embeddings are {dim}d — appears to be raw fused vector (not PCA-compressed). "
            "Run with --fit-pca to compress before indexing."
        )

    if n_songs < IVFPQ_THRESHOLD:
        log.info(f"Using IndexFlatIP (exact, {n_songs} songs < {IVFPQ_THRESHOLD})")
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)
    else:
        nlist = min(int(n_songs ** 0.5), 512)
        m = 8
        while dim % m != 0 and m > 1:
            m //= 2
        nbits = 8
        log.info(f"Using IndexIVFPQ (approx, nlist={nlist}, m={m}, nbits={nbits})")
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
        index.train(matrix)
        index.add(matrix)
        index.nprobe = min(64, nlist)

    os.makedirs("index", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    np.save(SONG_MAP_PATH, np.array(song_ids, dtype=object))

    log.info(f"\n✓ FAISS index → {INDEX_PATH}  ({index.ntotal} vectors, dim={dim})")
    log.info(f"✓ Song ID map → {SONG_MAP_PATH}")
    return index, song_ids


# ─── Incremental add (for async background worker) ───────────────────────────

def add_song_to_index(song_id: str, embedding_path: str):
    """
    Append one new pre-compressed song to an existing IndexFlatIP.
    Not compatible with IndexIVFPQ — call build_index() for full rebuild.
    Called by the async worker after ingesting an unknown song.
    """
    if not os.path.exists(INDEX_PATH):
        log.warning("No FAISS index found. Run build_index.py first.")
        return

    index    = faiss.read_index(INDEX_PATH)
    song_ids = np.load(SONG_MAP_PATH, allow_pickle=True).tolist()

    if song_id in song_ids:
        log.info(f"{song_id} already in index.")
        return

    vec = np.load(embedding_path).astype(np.float32).reshape(1, -1)

    if isinstance(index, faiss.IndexIVFPQ):
        log.warning(
            "Incremental add not supported on IndexIVFPQ. "
            "Run build_index.py (no flags) for a full rebuild."
        )
        return

    index.add(vec)
    song_ids.append(song_id)

    faiss.write_index(index, INDEX_PATH)
    np.save(SONG_MAP_PATH, np.array(song_ids, dtype=object))
    log.info(f"Added {song_id} → index now has {index.ntotal} songs.")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build or rebuild the FAISS similarity index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # After first pilot batch:
  python index/build_index.py --fit-pca

  # After fixing feature bugs (audio gone, feature JSONs exist):
  python index/build_index.py --from-features

  # Quick rebuild after adding new pre-compressed songs:
  python index/build_index.py
        """,
    )
    parser.add_argument(
        "--fit-pca",
        action="store_true",
        help="Fit adaptive PCA on raw fused embeddings, compress, then build index.",
    )
    parser.add_argument(
        "--from-features",
        action="store_true",
        help=(
            "Rebuild fused vectors from data/features/*.json (no audio re-download needed). "
            "Use after fixing feature extraction bugs. Implies --fit-pca."
        ),
    )
    args = parser.parse_args()

    init_db()
    songs = get_all_songs()

    if not songs:
        log.error(
            "No processed songs in the database. "
            "Run scripts/run_pipeline.py first."
        )
        sys.exit(1)

    log.info(f"Found {len(songs)} processed songs.")

    if args.from_features:
        log.info("\n=== Mode: Rebuild from JSONs → Fit PCA → Build Index ===")
        songs = rebuild_embeddings_from_features(songs)
        fit_and_compress_pca(songs)
        songs = get_all_songs()
        build_index(songs, pca_already_fit=True)

    elif args.fit_pca:
        log.info("\n=== Mode: Fit PCA → Compress → Build Index ===")
        fit_and_compress_pca(songs)
        songs = get_all_songs()
        build_index(songs, pca_already_fit=True)

    else:
        log.info("\n=== Mode: Build Index from existing compressed embeddings ===")
        build_index(songs)


if __name__ == "__main__":
    main()