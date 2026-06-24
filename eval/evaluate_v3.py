import os
"""
evaluate.py
===========
Grades the whole pool with the independent gold_scorer, then computes the
challenge's official composite for one or more submission CSVs:

    composite = 0.50*NDCG@10 + 0.30*NDCG@50 + 0.15*MAP + 0.05*P@10

Against a PROXY ground truth, so treat absolute numbers as indicative, not exact.
The useful output is the RELATIVE comparison between submissions and the
distribution of tiers in each top-100.
"""
from __future__ import annotations
import csv
import math
import sys

import orjson

from gold_v3_independent import grade

PATH = os.environ.get("CANDIDATES_PATH", "data/candidates.jsonl")


def load_gold():
    gold = {}
    with open(PATH, "rb") as f:
        for line in f:
            if line.strip():
                c = orjson.loads(line)
                gold[c["candidate_id"]] = grade(c)
    return gold


def dcg(rels):
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked_rels, all_rels, k):
    actual = dcg(ranked_rels[:k])
    ideal = dcg(sorted(all_rels, reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


def average_precision(ranked_rels, n_relevant, rel_threshold=3):
    """AP treating tier>=threshold as relevant (binary), over the ranked list."""
    if n_relevant == 0:
        return 0.0
    hits = 0
    s = 0.0
    for i, r in enumerate(ranked_rels):
        if r >= rel_threshold:
            hits += 1
            s += hits / (i + 1)
    return s / min(n_relevant, len(ranked_rels)) if n_relevant else 0.0


def evaluate(csv_path, gold):
    with open(csv_path, newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["rank"]))
    ranked_ids = [r["candidate_id"] for r in rows]
    ranked_rels = [gold.get(cid, 0) for cid in ranked_ids]

    all_rels = list(gold.values())
    n_relevant = sum(1 for r in all_rels if r >= 3)

    ndcg10 = ndcg_at_k(ranked_rels, all_rels, 10)
    ndcg50 = ndcg_at_k(ranked_rels, all_rels, 50)
    p10 = sum(1 for r in ranked_rels[:10] if r >= 3) / 10.0
    ap = average_precision(ranked_rels, n_relevant)

    composite = 0.50 * ndcg10 + 0.30 * ndcg50 + 0.15 * ap + 0.05 * p10

    from collections import Counter
    tier_dist = Counter(ranked_rels)
    return dict(ndcg10=ndcg10, ndcg50=ndcg50, map=ap, p10=p10,
                composite=composite, tier_dist=dict(sorted(tier_dist.items())),
                top10_tiers=ranked_rels[:10])


if __name__ == "__main__":
    gold = load_gold()
    from collections import Counter
    gd = Counter(gold.values())
    total_rel = sum(v for k, v in gd.items() if k >= 3)
    print("=== PROXY GOLD distribution over 100k ===")
    for t in range(6):
        print(f"  tier {t}: {gd.get(t,0):6d}")
    print(f"  tier>=3 (relevant): {total_rel}")
    print()

    for path in sys.argv[1:]:
        r = evaluate(path, gold)
        print(f"=== {path} ===")
        print(f"  NDCG@10   : {r['ndcg10']:.4f}")
        print(f"  NDCG@50   : {r['ndcg50']:.4f}")
        print(f"  MAP       : {r['map']:.4f}")
        print(f"  P@10      : {r['p10']:.4f}")
        print(f"  COMPOSITE : {r['composite']:.4f}")
        print(f"  top-100 tier dist: {r['tier_dist']}")
        print(f"  top-10 tiers     : {r['top10_tiers']}")
        print()
