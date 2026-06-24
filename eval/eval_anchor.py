"""Evaluate ranking quality against the 49-candidate HUMAN-ANCHORED set.
Because the anchor is a sample (not the full pool), we report:
  - Spearman/pairwise agreement between a ranker's ordering of the anchor
    candidates and their human tiers (the trustworthy, non-circular signal).
This is the tiebreaker grader: non-circular with both the ranker and regex golds.
"""
import json, sys, csv
from itertools import combinations

def load_anchor(path):
    return {o['id']: o['anchor'] for o in json.load(open(path))}

def pairwise_agreement(order_ids, tiers):
    """Of all anchor pairs with different tiers, fraction the ranker orders correctly."""
    pos = {cid: i for i, cid in enumerate(order_ids)}
    present = [cid for cid in order_ids if cid in tiers]
    good = total = 0
    for a, b in combinations(present, 2):
        if tiers[a] == tiers[b]:
            continue
        total += 1
        higher = a if tiers[a] > tiers[b] else b
        lower = b if higher == a else a
        if pos[higher] < pos[lower]:
            good += 1
    return good / total if total else 0.0, total

if __name__ == "__main__":
    anchor = load_anchor(sys.argv[1])
    # subsequent args: ranker CSVs that include ALL anchor ids in ranked order.
    # For a top-100 CSV we can only judge anchor ids that appear; report coverage.
    for csv_path in sys.argv[2:]:
        rows = sorted(csv.DictReader(open(csv_path)), key=lambda r: int(r['rank']))
        order = [r['candidate_id'] for r in rows]
        in_top = [c for c in anchor if c in set(order)]
        agr, npairs = pairwise_agreement(order, {c: anchor[c] for c in in_top})
        print(f"{csv_path}: {len(in_top)}/{len(anchor)} anchor ids in list; "
              f"pairwise order agreement={agr:.3f} over {npairs} tiered pairs")
