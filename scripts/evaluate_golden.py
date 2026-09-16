"""
scripts/evaluate_golden.py
────────────────────────────
Evaluate the recommendation engine against hand-judged relevance, NOT
against the self-assigned `category` column.

Why this exists (see data/eval/golden_relevance.json for details):
  scripts/evaluate.py measures whether top-K results share the anchor's
  coarse category label. That's a weak, partly circular proxy — it was
  computed on the same 50 songs the PCA model was fit on, and category
  boundaries between e.g. "upbeat" and "party" are fuzzy human calls.

  This script instead scores against a small curated set of graded
  relevance judgments (0/1/2) per anchor song, and reports nDCG@K —
  which respects ranking order and partial relevance — alongside a
  simpler Precision@K (grade >= 1 counts as relevant).

Usage:
  python scripts/evaluate_golden.py
  python scripts/evaluate_golden.py --k 10
  python scripts/evaluate_golden.py --golden data/eval/golden_relevance.json --verbose
"""

import sys
import os
import json
import argparse
import math

sys.path.insert(0, ".")
from utils.db import init_db, get_song_by_id
from index.search import find_similar_by_id, load_index
from index import rerank

DEFAULT_GOLDEN = "data/eval/golden_relevance.json"


def _dcg(grades: list[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(ranked_grades: list[int], ideal_grades: list[int], k: int) -> float:
    """ranked_grades: grades of the actually-returned results in rank order.
    ideal_grades: all judged grades for this anchor, sorted descending."""
    dcg = _dcg(ranked_grades[:k])
    idcg = _dcg(sorted(ideal_grades, reverse=True)[:k])
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(ranked_grades: list[int], k: int) -> float:
    top = ranked_grades[:k]
    if not top:
        return 0.0
    return sum(1 for g in top if g >= 1) / len(top)


def evaluate(golden_path: str, k: int = 5, verbose: bool = False, rerank_weight_overrides: dict = None):
    init_db()
    try:
        load_index()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    with open(golden_path) as f:
        golden = json.load(f)

    weights = dict(rerank.DEFAULT_WEIGHTS)
    if rerank_weight_overrides:
        weights.update(rerank_weight_overrides)

    anchors = golden["anchors"]
    ndcg_scores = []
    prec_scores = []

    print(f"\n{'═'*72}")
    print(f"  Golden-set evaluation  (nDCG@{k} / Precision@{k}, k_fetch={max(k,10)})")
    print(f"{'═'*72}")

    for anchor in anchors:
        anchor_id = anchor["anchor_id"]
        judgments = {j["song_id"]: j["grade"] for j in anchor["judgments"]}

        if get_song_by_id(anchor_id) is None:
            print(f"\n  [SKIP] anchor not in DB: {anchor_id}")
            continue

        try:
            results = find_similar_by_id(anchor_id, top_k=max(k, 10), rerank_weights=weights)
        except Exception as exc:
            print(f"\n  [SKIP] search failed for {anchor_id}: {exc}")
            continue

        ranked_grades = [judgments.get(r["song_id"], 0) for r in results]
        ideal_grades = list(judgments.values())

        ndcg = ndcg_at_k(ranked_grades, ideal_grades, k)
        prec = precision_at_k(ranked_grades, k)
        ndcg_scores.append(ndcg)
        prec_scores.append(prec)

        row = get_song_by_id(anchor_id)
        print(f"\n  Anchor: {row['title']} — {row['artist']}  [{anchor_id}]")
        print(f"    nDCG@{k} = {ndcg:.3f}   Precision@{k} = {prec:.3f}   "
              f"(judged pool: {len(judgments)} songs, "
              f"{sum(1 for g in judgments.values() if g == 2)} excellent / "
              f"{sum(1 for g in judgments.values() if g == 1)} decent)")

        if verbose:
            print(f"    {'Rank':<5}{'Grade':<7}{'Song':<45}{'Score':>8}")
            for i, r in enumerate(results[:k], 1):
                grade = judgments.get(r["song_id"], 0)
                marker = "✓✓" if grade == 2 else ("✓ " if grade == 1 else "  ")
                print(f"    {i:<5}{marker+' '+str(grade):<7}"
                      f"{(r['title']+' — '+r['artist'])[:43]:<45}{r['score']:>8.4f}")

    print(f"\n{'─'*72}")
    if ndcg_scores:
        print(f"  Mean nDCG@{k}:      {sum(ndcg_scores)/len(ndcg_scores):.3f}  "
              f"across {len(ndcg_scores)} anchors")
        print(f"  Mean Precision@{k}: {sum(prec_scores)/len(prec_scores):.3f}")
    else:
        print("  No anchors were evaluable — check that anchor songs are in the DB.")
    print(f"{'═'*72}\n")

    provenance = golden.get("_readme", "")
    if provenance:
        print(f"  Provenance ({golden_path}):")
        print(f"  {provenance}\n")
    else:
        print(f"  Reminder: {golden_path} has no _readme describing its provenance —")
        print("  don't trust its judgments without knowing where they came from.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate against hand-judged relevance (nDCG).")
    parser.add_argument("--golden", type=str, default=DEFAULT_GOLDEN)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--verbose", action="store_true", help="Show ranked results per anchor")
    parser.add_argument(
        "--rerank-weight", action="append", default=[], metavar="NAME=VALUE",
        help=(
            "Override one Stage 2 rerank signal's weight (index/rerank.py SCORERS: "
            "composer, vocal_energy, tempo, tonnetz, lyric, mood). Repeatable. "
            "E.g. --rerank-weight tempo=0.1 --rerank-weight composer=0 "
            "(the latter disables composer entirely for this run)."
        ),
    )
    args = parser.parse_args()

    overrides = {}
    for item in args.rerank_weight:
        if "=" not in item:
            parser.error(f"--rerank-weight expects NAME=VALUE, got: {item}")
        name, value = item.split("=", 1)
        if name not in rerank.SCORERS:
            parser.error(f"Unknown rerank signal '{name}'. Valid: {list(rerank.SCORERS)}")
        overrides[name] = float(value)

    evaluate(args.golden, k=args.k, verbose=args.verbose, rerank_weight_overrides=overrides)
