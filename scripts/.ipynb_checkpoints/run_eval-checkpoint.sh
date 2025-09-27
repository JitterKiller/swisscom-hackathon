#!/usr/bin/env bash
set -e
python main.py eval \
  --csv data/sample_edges.csv \
  --checkpoint artifacts/best.pt \
  --batch_seconds 60 \
  --inject_pct 0.02 \
  --tau_add 0.5 \
  --tau_rem 0.5
