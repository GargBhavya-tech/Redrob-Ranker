# Redrob — Intelligent Candidate Discovery & Ranking

Reads a job description, understands what the role actually needs (not just
keywords), looks at the full candidate picture — career history, skills,
behavioral/platform signals — and produces a **top-100 shortlist a recruiter can
trust**, with a grounded one-line justification per candidate.

CPU-only, no GPU, no network at ranking time, ~45s on the full 100K pool, well
inside the 5-minute / 16 GB budget.

---

## Quick start

```bash
# 1. install
pip install -r requirements.txt

# 2. put the dataset in place (NOT committed to git — see below)
#    copy the organizer's candidates.jsonl into ./data/
cp /path/to/candidates.jsonl data/candidates.jsonl

# 3. run
python -m src.run --candidates data/candidates.jsonl --out submission.csv

# 4. validate the output format
python validate_submission.py submission.csv
```

Output `submission.csv` columns: `candidate_id, rank, score, reasoning`.

---

## How it works (architecture)

A **two-stage cascade** — the only design that fits "no labeled data + 5-minute
CPU budget" (supervised LambdaMART/XGBoost have no labels to train on).

1. **Feature extraction** (`src/features.py`) — streams the pool with `msgspec`
   and computes per-candidate structured signals: title match, skills overlap,
   **build-vs-listed evidence** (reads career-description prose, not just skill
   tags), years-of-experience fit (asymmetric Gaussian), location, seniority,
   and a behavioral composite (activity recency, recruiter-response rate,
   open-to-work, notice-period). Plus two trap flags:
   - `is_honeypot` — **structural impossibilities** (total tenure ≫ career
     length; a single role longer than the whole career; ≥2 expert/advanced
     skills with 0 months used; summary-vs-profile YoE contradiction). Matches
     the spec's own honeypot examples; calibrated to the planted ~80.
   - `is_keyword_stuffer` — AI keywords on a non-technical profile with no
     matching project evidence.

2. **Stage-1 shortlist** (`src/stage1_shortlist.py`) — fast structured score
   over all 100K, drops honeypots, keeps the top ~1500 for the expensive stage.

3. **Stage-2 semantic** (`src/stage2_semantic.py`) — recent-role-weighted
   TF-IDF cosine vs the JD query. The candidate document **front-loads the most
   recent role** (the ground truth of what they built) and discounts the
   summary/title (which oversell in this dataset). Fully offline, deterministic.

4. **RRF fusion** (`src/fusion.py`) — Reciprocal Rank Fusion (k=60) over
   structured + semantic + behavioral signals. Rank-based fusion avoids the
   score-domination problem of linear weighting; keyword-stuffers get a large
   rank penalty.

5. **Reasoning** (`src/reasoning.py`) — grounded, employer-named, one-line
   justifications built only from verified profile fields (no hallucination),
   with honest concerns surfaced (notice period, location, seniority).

6. *(optional)* **Cross-encoder re-rank** (`src/stage3_cross_encoder.py`) — off
   by default; runs only if the model is present, blends with the fused score,
   and falls back to a no-op otherwise. See `CROSS_ENCODER_RUNBOOK.md`. Most of
   its benefit is already captured by the recent-role document weighting.

---

## Why these choices (design defense)

- **Cascade, not end-to-end LTR** — no labels, so no supervised ranker to train.
- **RRF over linear weighting** — cosine, structured, and behavioral scores live
  on incompatible scales; fusing *ranks* is stable and needs no scale tuning.
- **Read recent-role prose, distrust title/skills** — the dataset deliberately
  makes titles and summaries oversell. Two "Computer Vision Engineer"-titled
  candidates correctly land at tier 5 vs tier 2 based on what their recent role
  actually built. This is the central insight; the whole pipeline encodes it.
- **Behavioral as a bounded multiplier, not a primary signal** — an inactive,
  unresponsive candidate is down-weighted, never erased.

---

## Evaluation (the rigor story — see `EVALUATION.md` + `DECISION_MAP.md`)

There is **no labeled ground truth**, so we built our own and stress-tested it:

- a regex proxy gold, a second *independent-feature* grader, and a
  **49-candidate human-anchored set** graded by reading recent-role prose.
- Key finding: the regex graders agreed with the human anchor only **25/49
  exactly** — proof they were partly circular. We re-tuned weights against the
  **human anchor**, not the regex gold.
- All evaluators live in `eval/`. Run, e.g.:
  ```bash
  CANDIDATES_PATH=data/candidates.jsonl python eval/evaluate.py submission.csv
  python eval/eval_anchor.py eval/anchor_labels.json submission.csv
  ```

Measured on the final submission: **0 honeypots and 0 keyword-stuffers in the
top 100**, top picks are Senior/Lead retrieval engineers (Zomato, Razorpay,
Paytm, Google) in the 6–8 year band with concerns surfaced honestly.

---

## What to push to GitHub

Push the **files**, not a zip, and not the dataset. The included `.gitignore`
already excludes the dataset, model caches, and scratch outputs.

```bash
bash init_git.sh                 # makes an honest, staged commit history
git remote add origin https://github.com/<you>/redrob-ranker.git
git branch -M main
git push -u origin main
```

`init_git.sh` commits in build order (schema → features → stages → fusion →
reasoning → evaluation → docs) so the history reflects real development rather
than one flat "initial commit" (a Stage-4 elimination signal).

**Do NOT commit:** `data/candidates.jsonl` (the 465 MB dataset), `__pycache__/`,
model weights, or scratch CSVs — all already in `.gitignore`.

---

## The three deliverables

1. **This repo** (public GitHub URL).
2. **Approach deck → PDF** (separate upload).
3. **Ranked output CSV** — the `submission.csv` this produces (separate upload,
   in the format above).

---

## Repo layout

```
src/                ranker pipeline (ingest → features → stages → fusion → reasoning)
  stage3_cross_encoder.py   optional, off-by-default upside layer
eval/               evaluation harness: regex golds, independent grader, human anchor
data/               sample_candidates.json + example_submission.csv (dataset NOT included)
README.md           this file
DECISION_MAP.md     the optimization investigation, ticket by ticket
EVALUATION.md       honest multi-grader results + caveats
CROSS_ENCODER_RUNBOOK.md   how to benchmark the optional cross-encoder
init_git.sh         staged-commit script for honest git history
requirements.txt
validate_submission.py
```
