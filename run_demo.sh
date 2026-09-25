#!/bin/bash

echo "[SelenoMatch] Starting Quick Evaluation Demo..."
echo "[SelenoMatch] Bypassing spatial ingestion to run core matching engine on pre-processed CLAHE pairs..."

mkdir -p demo_output

python demo_only.py \
    --ref-img sample_dataset/reference_clahe.png \
    --src-img sample_dataset/source_clahe.png \
    --out-dir demo_output

echo "[SelenoMatch] Demo complete! Check 'demo_output' for tie-line visualizations and metrics."
