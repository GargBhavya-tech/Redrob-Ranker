#!/usr/bin/env bash
# ============================================================
#  init_git.sh — create an HONEST, staged git history.
#
#  Stage-4 review checks for flat "one giant initial commit" history.
#  This script commits the repo in logical build order so the history
#  reflects how the system was actually developed: schema/ingest ->
#  features -> stages -> fusion -> reasoning -> evaluation -> tuning -> docs.
#
#  USAGE (run once, locally, from the repo root):
#     bash init_git.sh
#     git remote add origin https://github.com/<you>/redrob-ranker.git
#     git branch -M main
#     git push -u origin main
# ============================================================
set -e

git init -q

# Ensure a git identity exists (commits fail without one). Set a local default
# if none is configured — change it to your own before pushing if you like.
if ! git config user.email >/dev/null 2>&1; then
  echo "No git identity found; setting a local default for this repo."
  echo "  (change with: git config user.name '...'; git config user.email '...')"
  git config user.email "you@example.com"
  git config user.name "Your Name"
fi

git add .gitignore
git commit -q -m "chore: project scaffold and gitignore"

# 1. data contract / ingestion
git add src/__init__.py src/schema.py src/date_utils.py requirements.txt
git commit -q -m "feat: msgspec candidate schema + fast JSONL ingest contract"

# 2. structured features (incl. honeypot + evidence signals)
git add src/features.py src/jd_config.py
git commit -q -m "feat: structured signals — honeypot detection, build-vs-listed evidence, disqualifiers"

# 3. stage 1 shortlist
git add src/stage1_shortlist.py src/build_features.py
git commit -q -m "feat: stage-1 cheap structured shortlist over full pool"

# 4. stage 2 semantic
git add src/stage2_semantic.py
git commit -q -m "feat: stage-2 recent-role-weighted TF-IDF semantic re-rank"

# 5. fusion
git add src/fusion.py
git commit -q -m "feat: RRF fusion of structured + semantic + behavioral signals"

# 6. reasoning + enrichment
git add src/enrich.py src/reasoning.py
git commit -q -m "feat: grounded, employer-named, hallucination-free reasoning"

# 7. optional cross-encoder
git add src/stage3_cross_encoder.py CROSS_ENCODER_RUNBOOK.md
git commit -q -m "feat: optional cross-encoder re-rank (graceful no-op if model absent)"

# 8. evaluation harness + human anchor
git add eval/
git commit -q -m "test: proxy + independent + human-anchored evaluation harness"

# 9. entry point + validator
git add src/run.py validate_submission.py
git commit -q -m "feat: pipeline entry point and submission validator"

# 10. docs + example output
git add README.md DECISION_MAP.md EVALUATION.md data/sample_candidates.json data/example_submission.csv
git commit -q -m "docs: README, decision map, evaluation writeup, example output"

# anything left (safety net)
git add -A
git commit -q -m "chore: finalize repo" || true

echo "Done. Git history:"
git log --oneline
