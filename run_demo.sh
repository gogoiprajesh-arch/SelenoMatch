#!/bin/bash

echo "[SelenoMatch] Starting Quick Evaluation Demo..."
echo "[SelenoMatch] Bypassing spatial ingestion to run core matching engine on pre-processed CLAHE pairs..."

mkdir -p demo_output

python demo_only.py \
    --ref-img processed/pair_2/reference_processed.png \
    --src-img processed/pair_2/source_processed.png \
    --out-dir demo_output

echo "[SelenoMatch] Demo complete! Check 'demo_output' for tie-line visualizations and metrics."