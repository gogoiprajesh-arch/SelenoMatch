"""
Master Pipeline
------------------
Raw Optical Datasets -> Metadata Parser -> Intersection Calculator ->
Geo-Sampler -> Optical Preprocessor -> Patch Extractor ->
[LoFTR || RIFT-2] -> Match Fusion -> RANSAC Filter -> Sub-Pixel Alignment
-> Match Visualization

Everything is kept in memory between stages -- the only things written to
disk are the final deliverables, all into one flat `out_dir` (no
data/raw, data/processed, etc. sub-tree).

Usage:
    python run_pipeline.py \
        --ref Raw/M1195313832LE_stereo.tif \
        --src Raw/ch2_ohr_ncp_..._d_img_d18.xml \
        --out-dir output\
        --is--iirs
"""
import argparse
import json
import time
from pathlib import Path
import torch
import gc

import cv2
import numpy as np
import rasterio
from tqdm import tqdm

from metadata import parse_metadata
from geometry import compute_overlap_windows
from geosampler import sample_overlap
from preprocess import preprocess
from patches import extract_and_filter_patches
from features import LoFTRRunner, RIFT2Runner
from fusion import fuse_tile_matches , enforce_uniform_distribution
from ransac_filter import evaluate_global, evaluate_piecewise
from subpixel import refine_to_subpixel
from visualize import save_tie_line_visualization


def _build_meta(path):
    corners, resolution, product_type = parse_metadata(path)
    with rasterio.open(str(path)) as src:
        shape = (src.height, src.width)
        transform = src.transform if product_type == "tif" else None
    return {
        "corners": corners, "resolution": resolution, "product_type": product_type,
        "shape": shape, "transform": transform,
    }


def run_pipeline(ref_path, src_path, loftr_repo_dir, loftr_weights, out_dir="output",
                  patch_size=1024, stride=512, solar_incidence_deg=65.0,
                  use_piecewise=True, rift2_config=None, loftr_conf_thresh=0.3,
                  grid_cap_per_cell=15, is_iirs=False):
    t_start = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Metadata Parser 
    print("[1/10] Parsing metadata...")
    meta_ref = _build_meta(ref_path)
    meta_src = _build_meta(src_path)

    # 2. Intersection Calculator
    print("[2/10] Computing overlap geometry...")
    overlap = compute_overlap_windows(meta_ref, meta_src)
    if overlap is None:
        raise ValueError("Reference and source footprints do not overlap.")
    print(f"       target resolution: {overlap['target_res']:.2f} m/px, "
          f"shared grid: {overlap['universal_out_shape']}")

    # 3. Geo-Sampler 
    print("[3/10] Sampling overlap window from each raw file...")
    out_shape = overlap["universal_out_shape"]
    
    is_ref_xml = str(ref_path).endswith(".xml")
    is_src_xml = str(src_path).endswith(".xml")
    if is_ref_xml and is_src_xml:
        print("       [XML vs XML Mode] Native extraction with memory-safe scaling...")
  
        raw_ref = sample_overlap(ref_path, overlap["bbox1"], out_shape=None)
        raw_src = sample_overlap(src_path, overlap["bbox2"], out_shape=None)
         
        if raw_ref.size > 0 and raw_src.size > 0:
            # 1. Align with max(res1, res2) by finding the smaller array's native bounds
            target_h = max(raw_ref.shape[0], raw_src.shape[0])
            target_w = min(max(raw_ref.shape[1], raw_src.shape[1]),5000)
            
            # 2. Resize safely down to that exact physical intersection using INTER_AREA
            raw_ref = cv2.resize(raw_ref, (target_w, target_h), interpolation=cv2.INTER_AREA)
            raw_src = cv2.resize(raw_src, (target_w, target_h), interpolation=cv2.INTER_AREA)

            print(f"       DEBUG Final Scaled Shape: {raw_ref.shape}")
            
    else:
        # Standard processing for LRO/TIF using the universal out_shape grid
        raw_ref = sample_overlap(ref_path, overlap["bbox1"], out_shape=out_shape)
        
        if is_iirs:
            from geosampler import sample_iirs_aligned
            print("       [IIRS Mode] Averaging Bands 40-60 and orthorectifying to WAC grid...")
            raw_src = sample_iirs_aligned(
                src_path, 
                meta_src['corners'], 
                meta_ref['transform'], 
                overlap["bbox1"], 
                out_shape
            )
        else:
            raw_src = sample_overlap(src_path, overlap["bbox2"], out_shape=out_shape)

   #4. Optical Preprocessor
    print(f"[4/10] Preprocessing (IIRS mode: {is_iirs})...")
    img_ref = preprocess(raw_ref)
    img_src = preprocess(raw_src)

    is_ref_tif = str(ref_path).endswith(".tif")
    is_src_xml = str(src_path).endswith(".xml")
    
    if is_ref_tif and is_src_xml:
        h, w = img_ref.shape
        if h < patch_size or w < patch_size:
            print(f"       [TIF vs XML] Padding images from {w}x{h} to meet the {patch_size}x{patch_size} minimum...")
            pad_h = max(0, patch_size - h)
            pad_w = max(0, patch_size - w)
            img_ref = cv2.copyMakeBorder(img_ref, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            img_src = cv2.copyMakeBorder(img_src, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    cv2.imwrite(str(out_dir / "reference_clahe.png"), img_ref)
    cv2.imwrite(str(out_dir / "source_clahe.png"), img_src)

    # 5. Patch Extractor 
    print(f"[5/10] Extracting QC-filtered {patch_size}x{patch_size} macro-patch pairs...")
    tiles = extract_and_filter_patches(
        img_ref, 
        img_src, 
        patch_size=patch_size, 
        stride=stride,
        max_black_fraction=0.80  # Overrides strict 20% limit for diagonal strips
    )
    if not tiles:
        raise ValueError("No valid patch pairs survived QC filtering.")

    # 6+7. Feature Extractors + Match Fusion (per macro-patch
   
    print("[6-7/10] Running LoFTR + RIFT-2 per macro-patch and fusing candidates...")
    loftr = LoFTRRunner(loftr_repo_dir, loftr_weights, conf_thresh=loftr_conf_thresh)
    rift2 = RIFT2Runner(config_file=rift2_config)

    global_pts1, global_pts2, global_conf = [], [], []
    tile_matches = []  # for the piecewise RANSAC strategy

    for i, tile in enumerate(tqdm(tiles, desc="tiles")):
        y_off, x_off = tile["y_off"], tile["x_off"]

        pts1_r, pts2_r, conf_r = rift2.match(tile["ref"], tile["src"])
        pts1_l, pts2_l, conf_l = loftr.match(tile["ref"], tile["src"])

   
        pts1_g, pts2_g, conf_g, source_g = fuse_tile_matches(
            pts1_r, pts2_r, conf_r, pts1_l, pts2_l, conf_l,
            y_off, x_off, solar_incidence_deg=solar_incidence_deg,
        )

        if len(pts1_g) == 0:
            continue

        rift_idx = np.where(source_g == "rift")[0]
        loftr_idx = np.where(source_g == "loftr")[0]

        if len(loftr_idx) > 80:
            best_loftr_idx = loftr_idx[np.argsort(-conf_g[loftr_idx])[:80]]
            keep_idx = np.concatenate([rift_idx, best_loftr_idx])
            
            pts1_g, pts2_g = pts1_g[keep_idx], pts2_g[keep_idx]
            conf_g, source_g = conf_g[keep_idx], source_g[keep_idx]
    

        global_pts1.append(pts1_g)
        global_pts2.append(pts2_g)
        global_conf.append(conf_g)
        
        tile_matches.append({"y_off": y_off, "x_off": x_off,
                              "pts_ref": pts1_g, "pts_src": pts2_g, 
                              "conf": conf_g, "source": source_g})
        
        if i % 15 == 0:
            gc.collect()
            
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not global_pts1:
        raise ValueError("No matches survived fusion across any tile.")

    pts1 = np.vstack(global_pts1).astype(np.float32)
    pts2 = np.vstack(global_pts2).astype(np.float32)
    conf = np.concatenate(global_conf).astype(np.float32)
    print(f"       {len(pts1)} fused candidate matches before density capping")

    pts1, pts2, conf = enforce_uniform_distribution(
        pts1, pts2, conf, img_ref.shape, grid_size=10, max_per_cell=grid_cap_per_cell
    )
    print(f"       {len(pts1)} candidate matches after density capping")

    np.savez_compressed(out_dir / "fused_matches.npz", pts1=pts1, pts2=pts2, conf=conf)

    # ---- 8. RANSAC Filter ----------------------------------------------
    print("[8/10] RANSAC geometric verification...")
    global_result = evaluate_global(pts1, pts2, conf, img_ref, img_src,
                                     out_prefix=str(out_dir / "global"))

    piecewise_result = None
    if use_piecewise:
        piecewise_result = evaluate_piecewise(
            tile_matches, img_ref, img_src, patch_size=patch_size,
            min_inlier_ratio=0.02,  # <--- ADD THIS LINE
            out_prefix=str(out_dir / "piecewise"),
        )

    # ---- 9. Sub-Pixel Alignment ----------------------------------------
    print("[9/10] Refining inlier keypoints to sub-pixel precision...")
    refined_ref, refined_src = None, None
    
    if piecewise_result is not None and len(piecewise_result["tiles"]) > 0:
        # Extract all piecewise points and masks
        pw_pts_ref = np.vstack([t["pts_ref"] for t in piecewise_result["tiles"]])
        pw_pts_src = np.vstack([t["pts_src"] for t in piecewise_result["tiles"]])
        pw_mask = np.concatenate([t["mask"] for t in piecewise_result["tiles"]]).ravel().astype(bool)
        
        # Filter down to only the green inliers
        p_ref_in = pw_pts_ref[pw_mask]
        p_src_in = pw_pts_src[pw_mask]
        
        refined_ref, refined_src = refine_to_subpixel(img_ref, img_src, p_ref_in, p_src_in)
        np.savez_compressed(out_dir / "subpixel_inliers.npz",
                             pts_ref=refined_ref, pts_src=refined_src)
        print(f"       refined {len(refined_ref)} Piecewise inlier keypoints")

    # ---- 10. Match Visualization ----------------------------------------
    print("[10/10] Rendering tie-line visualization...")
    
    if piecewise_result is not None and len(piecewise_result["tiles"]) > 0:
        # Aggregate all points and masks from the successful piecewise tiles
        pw_pts_ref = np.vstack([t["pts_ref"] for t in piecewise_result["tiles"]])
        pw_pts_src = np.vstack([t["pts_src"] for t in piecewise_result["tiles"]])
        pw_mask = np.concatenate([t["mask"] for t in piecewise_result["tiles"]])
        
        save_tie_line_visualization(
            img_ref, img_src, pw_pts_ref, pw_pts_src, pw_mask,
            str(out_dir / "piecewise_visualization.png"),
            title="Piecewise MAGSAC — Lunar Keypoint Registration",
        )
    elif global_result is not None:
        # Fallback to global if piecewise was turned off
        save_tie_line_visualization(
            img_ref, img_src, global_result["pts1_valid"], global_result["pts2_valid"],
            global_result["mask"], str(out_dir / "global_visualization.png"),
            title="Global RANSAC — Lunar Keypoint Registration",
        )

    summary = {
        "elapsed_sec": round(time.time() - t_start, 1),
        #"tiles_processed": len(tiles),
        "candidate_matches_after_capping": int(len(pts1)),
        "global": {k: v for k, v in (global_result or {}).items()
                   if k in ("inlier_count", "total_tested", "inlier_ratio", "rmse")},
        "piecewise": {k: v for k, v in (piecewise_result or {}).items()
                      if k in ("tiles_used", "avg_tile_inlier_ratio",
                                "median_tile_inlier_ratio", "avg_tile_rmse", "median_tile_rmse")},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone in {summary['elapsed_sec']}s. Outputs in {out_dir}/")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end lunar cross-sensor registration pipeline.")
    
    # Only keep the arguments that actually change between test runs
    parser.add_argument("--ref", required=True, help="Reference raw file (.xml or .tif)")
    parser.add_argument("--src", required=True, help="Source raw file (.xml or .tif)")
    parser.add_argument("--out-dir", default="output", help="Output directory name")
    parser.add_argument("--is-iirs", action="store_true", help="Trigger IIRS hyperspectral Band 50 RAM-load")
    
    args = parser.parse_args()

    run_pipeline(
        ref_path=args.ref, 
        src_path=args.src,
        loftr_repo_dir="./LoFTR",                           # Fixed path
        loftr_weights="./LoFTR/weights/outdoor_ds.ckpt",    # Fixed path
        out_dir=args.out_dir,                               # Fixed to use the CLI argument
        patch_size=1024,                                    # Locked to RIFT-2/LoFTR macro-patch spec
        stride=512,                                         # Standard overlap stride
        solar_incidence_deg=69.0,                           # Default grazing angle
        use_piecewise=True,                                 # Always run Piecewise MAGSAC
        rift2_config=None,                                  # Use default RIFT-2 internal config
        is_iirs=args.is_iirs                                # Converted hyphen to underscore
    )

