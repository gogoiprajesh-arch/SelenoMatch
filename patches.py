"""
Stage 5: Patch Extractor
--------------------------
Slices the two co-registered, preprocessed arrays into overlapping
"macro-patches" and drops low-information ones (too much no-data, too
flat/low-contrast) before they ever reach the feature extractors.

Macro/Micro-patching: this stage hands out 1024x1024 patches by default,
not 512x512. RIFT-2 consumes a macro-patch whole (phase congruency needs
the wider spatial context for stable orientation histograms); LoFTR slices
each macro-patch into four 512x512 micro-patch quadrants internally
(see features.LoFTRRunner) to stay within GPU memory. This module only
produces the 1024x1024 macro-patches -- the quadrant split is a LoFTR-only
concern and lives entirely in features.py so patches.py doesn't need to
know which branch will consume its output.

Kept entirely in memory by default -- pass `save_dir` only if you want the
patches written to disk for debugging/inspection.
"""
import os
import cv2
import numpy as np


def extract_and_filter_patches(img_ref, img_src, patch_size=1024, stride=1024,
                                max_black_fraction=0.30, min_std=15.0,
                                save_dir=None):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    h, w = img_ref.shape
    max_allowed_black = (patch_size * patch_size) * max_black_fraction
    
    valid_count = 0
    total_count = ((h - patch_size) // stride + 1) * ((w - patch_size) // stride + 1)

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch_ref = img_ref[y:y + patch_size, x:x + patch_size]
            patch_src = img_src[y:y + patch_size, x:x + patch_size]

            black_ref = np.count_nonzero(patch_ref < 25)
            black_src = np.count_nonzero(patch_src < 25)
            if black_ref > max_allowed_black or black_src > max_allowed_black:
                continue

            if np.std(patch_ref) < min_std or np.std(patch_src) < min_std:
                continue

            if save_dir:
                cv2.imwrite(f"{save_dir}/reference_{y}_{x}.png", patch_ref)
                cv2.imwrite(f"{save_dir}/source_{y}_{x}.png", patch_src)

            valid_count += 1
            yield {"y_off": y, "x_off": x, "ref": patch_ref, "src": patch_src}

    print(f"[patches] Processed {valid_count} valid high-quality tile pairs out of {total_count} candidates.")