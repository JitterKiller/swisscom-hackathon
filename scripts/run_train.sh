#!/usr/bin/env bash
set -e
python main.py train \
  --csv data/sample_edges.csv \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --epochs 10 \
  --batch_seconds 60 \
  --neg_per_pos 3 \
  --out_dir artifacts
