# SelenoMatch Sample Dataset

This directory contains pre-processed, CLAHE-enhanced image pairs intended for quick evaluation of the SelenoMatch core registration engine.

## Why use this sample?
Raw lunar orbital data (e.g., ISRO PDS3/XML and NASA GeoTIFFs) can be gigabytes in size and require heavy memory usage for footprint intersection and metric grid resampling (Stages 1-4 of our pipeline). 

To allow judges to test our **LoFTR + RIFT-2 dual-branch matching algorithm** instantly without downloading massive raw datasets, these images have already been:
1. Geographically cropped to their overlapping region.
2. Resampled to a shared spatial resolution.
3. Processed with robust 8-bit normalization and masked CLAHE.

## Contents
* **Reference Image:** M192036964CC (NASA LRO WAC)
* **Source Image:** ch2_iir_nci_20221226T1010382905_d_img_d32 (ISRO Chandrayaan-2 IIRS)

## Usage
Do not run the standard `run_pipeline.py` on these `.png` files, as they intentionally bypass the geospatial metadata parsers. 

To test the matching pipeline on this dataset, return to the repository root and run the demo script:
```bash
bash run_demo.sh