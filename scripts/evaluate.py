"""
scripts/evaluate.py
────────────────────
Precision@K evaluation for the recommendation engine.

For each song in the database, retrieves its top-K recommendations
and measures what fraction share the same category label.

Reports:
  - Per-category Precision@K
  - Overall macro-average Precision@K
  - Confusion matrix (which categories bleed into which)

Target benchmarks (from FEATURES.md):
  ghazal       ≥ 0.80
  sufi         ≥ 0.75
  sad_romantic ≥ 0.70
  party        ≥ 0.85
  upbeat       ≥ 0.65

Usage:
  python scripts/evaluate.py
  python scripts/evaluate.py --k 5
  python scripts/evaluate.py --k 10 --category ghazal
"""

import sys
import os
import argparse
import logging
from collections import defaultdict

sys.path.insert(0, ".")
from utils.db    import init_db, get_all_songs
from index.search import find_similar_by_id, load_index

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Target benchmarks per FEATURES.md
TARGETS = {
    "ghazal":       0.80,
    "sufi":         0.75,
    "sad_romantic": 0.70,
    "party":        0.85,
    "upbeat":       0.65,
}


def evaluate(k: int = 5, category_filter: str = None):
    init_db()
    songs = get_all_songs()

    if not songs:
        log.error("No songs in the database. Run the pipeline first.")
        return

    # Load index once
    try:
        load_index()
    except FileNotFoundError as e:
        log.error(str(e))
        return

    # Filter by category if requested
    if category_filter:
        songs = [s for s in songs if s["category"] == category_filter]
        if not songs:
            log.error(f"No songs with category='{category_filter}'")
            return

    # Accumulate precision scores per category
    cat_scores  = defaultdict(list)   # category → list of precision values
    cat_matrix  = defaultdict(lambda: defaultdict(int))  # true → predicted counts

    total = len(songs)
    log.info(f"\nEvaluating Precision@{k} on {total} songs ...\n")

    for i, song in enumerate(songs):
        song_id  = song["id"]
        true_cat = song["category"]

        emb_path = song["embedding_path"]
        if not emb_path or not os.path.exists(emb_path):
            log.warning(f"  SKIP {song_id}: missing embedding")
            continue

        try:
            results = find_similar_by_id(song_id, top_k=k)
        except Exception as exc:
            log.warning(f"  SKIP {song_id}: {exc}")
            continue

        if not results:
            cat_scores[true_cat].append(0.0)
            continue

        # Count how many of top-K share the true category
        matches = sum(1 for r in results if r["category"] == true_cat)
        precision = matches / len(results)
        cat_scores[true_cat].append(precision)

        # Confusion matrix
        for r in results:
            cat_matrix[true_cat][r["category"]] += 1

        if (i + 1) % 10 == 0:
            log.info(f"  Evaluated {i+1}/{total} songs ...")

    # ── Report ────────────────────────────────────────────────────────────────

    print(f"\n{'═'*60}")
    print(f"  Precision@{k} Results")
    print(f"{'═'*60}")

    macro_scores = []
    categories = sorted(cat_scores.keys())

    print(f"\n{'Category':<20} {'Precision@K':>12} {'Songs':>8} {'Target':>10} {'Status':>8}")
    print(f"{'─'*60}")

    for cat in categories:
        scores = cat_scores[cat]
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        macro_scores.append(avg)
        target = TARGETS.get(cat, 0.60)
        status = "✓ PASS" if avg >= target else "✗ FAIL"
        print(f"  {cat:<18} {avg:>12.3f} {len(scores):>8}  {target:>8.2f}  {status:>8}")

    if macro_scores:
        macro_avg = sum(macro_scores) / len(macro_scores)
        print(f"{'─'*60}")
        print(f"  {'Macro average':<18} {macro_avg:>12.3f}")

    # ── Confusion matrix ──────────────────────────────────────────────────────

    if len(categories) > 1:
        print(f"\n{'─'*60}")
        print(f"  Confusion matrix (true category → top-{k} predicted categories)")
        print(f"{'─'*60}")
        col_w = 14
        header = f"{'True \\ Pred':<20}" + "".join(f"{c[:col_w-1]:>{col_w}}" for c in categories)
        print(f"  {header}")
        for true_cat in categories:
            row_str = f"  {true_cat:<20}"
            for pred_cat in categories:
                count = cat_matrix[true_cat].get(pred_cat, 0)
                row_str += f"{count:>{col_w}}"
            print(row_str)

    # ── Recommendations ───────────────────────────────────────────────────────

    print(f"\n{'─'*60}")
    print("  Improvement recommendations:")
    for cat in categories:
        scores = cat_scores.get(cat, [])
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        target = TARGETS.get(cat, 0.60)
        if avg < target:
            if cat == "ghazal":
                print(f"  → {cat}: Increase Vocal weight (vocal_energy_ratio is the key separator)")
            elif cat == "party":
                print(f"  → {cat}: Increase Rhythmic weight (onset_skewness + tempo_bucket)")
            elif cat == "sufi":
                print(f"  → {cat}: Increase Melodic weight (raga_probability helps here)")
            else:
                print(f"  → {cat}: Inspect UMAP cluster and review feature weights")

    print(f"{'═'*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Precision@K of the recommendation engine.")
    parser.add_argument("--k",        type=int, default=5, help="K for Precision@K (default: 5)")
    parser.add_argument("--category", type=str, default=None,
                        help="Evaluate only this category (default: all)")
    args = parser.parse_args()
    evaluate(k=args.k, category_filter=args.category)