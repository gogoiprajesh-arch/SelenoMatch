import cv2
import numpy as np
import torch
import gc
from pathlib import Path
from tqdm import tqdm

from patches import extract_and_filter_patches
from features import LoFTRRunner, RIFT2Runner
from fusion import fuse_tile_matches, enforce_uniform_distribution
from ransac_filter import evaluate_piecewise
from subpixel import refine_to_subpixel
from visualize import save_tie_line_visualization
from ransac_filter import evaluate_global, evaluate_piecewise

def test_processed_pair(ref_png_path, src_png_path, out_dir="test_output"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("[1/5] Loading pre-processed images to RAM...")
    img_ref = cv2.imread(ref_png_path, cv2.IMREAD_GRAYSCALE)
    img_src = cv2.imread(src_png_path, cv2.IMREAD_GRAYSCALE)
    
    if img_ref is None or img_src is None:
        raise ValueError("Could not load images. Check your file paths!")

    print(f"      Reference Shape: {img_ref.shape} | Source Shape: {img_src.shape}")

    patch_size = 1024
    stride = 512
    print(f"[2/5] Extracting QC-filtered {patch_size}x{patch_size} macro-patch pairs...")
    tiles = extract_and_filter_patches(
        img_ref, 
        img_src, 
        patch_size=patch_size, 
        stride=stride,
        max_black_fraction=0.80  # Keeps diagonal strips alive
    )
    if not tiles:
        raise ValueError("No valid patch pairs survived QC filtering. Check black space limits.")

    
    print("[3/5] Running LoFTR + RIFT-2 per macro-patch...")
    loftr = LoFTRRunner("./LoFTR", "./LoFTR/weights/outdoor_ds.ckpt", conf_thresh=0.3)
    rift2 = RIFT2Runner(config_file=None)

    tile_matches = []
    
    for i, tile in enumerate(tqdm(tiles, desc="Matching Tiles")):
        pts1_r, pts2_r, conf_r = rift2.match(tile["ref"], tile["src"])
        pts1_l, pts2_l, conf_l = loftr.match(tile["ref"], tile["src"])

        pts1_g, pts2_g, conf_g, source_g = fuse_tile_matches(
            pts1_r, pts2_r, conf_r, pts1_l, pts2_l, conf_l,
            tile["y_off"], tile["x_off"], solar_incidence_deg=69.0,
        )

        if len(pts1_g) > 0:
           
            rift_idx = np.where(source_g == "rift")[0]
            loftr_idx = np.where(source_g == "loftr")[0]

            if len(loftr_idx) > 80:
                best_loftr_idx = loftr_idx[np.argsort(-conf_g[loftr_idx])[:80]]
                keep_idx = np.concatenate([rift_idx, best_loftr_idx])
                pts1_g, pts2_g, conf_g, source_g = pts1_g[keep_idx], pts2_g[keep_idx], conf_g[keep_idx], source_g[keep_idx]

            tile_matches.append({
                "y_off": tile["y_off"], "x_off": tile["x_off"],
                "pts_ref": pts1_g, "pts_src": pts2_g, 
                "conf": conf_g, "source": source_g,
                "ref_patch": tile["ref"], "src_patch": tile["src"]
            })
      
        if i % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    print("[4/5] RANSAC geometric verification...")
    
    if tile_matches:
        
        all_pts_ref = np.vstack([t["pts_ref"] for t in tile_matches])
        all_pts_src = np.vstack([t["pts_src"] for t in tile_matches])
        all_conf = np.concatenate([t["conf"] for t in tile_matches])
        all_source = np.concatenate([t["source"] for t in tile_matches])
        
        all_pts_ref, all_pts_src, all_conf = enforce_uniform_distribution(
            all_pts_ref, all_pts_src, all_conf, img_ref.shape, grid_size=10, max_per_cell=15
        )
        
        evaluate_global(
            all_pts_ref, all_pts_src, all_conf, img_ref, img_src,
            out_prefix=str(out_dir / "global")
        )

    piecewise_result = evaluate_piecewise(
        tile_matches, img_ref, img_src, patch_size=patch_size,
        min_inlier_ratio=0.02,
        out_prefix=str(out_dir / "piecewise")
    )

   
    pw_warped_path = out_dir / "piecewise_warped.png"
    if pw_warped_path.exists():
        warped_src = cv2.imread(str(pw_warped_path), cv2.IMREAD_GRAYSCALE)
        overlay = cv2.merge([np.zeros_like(img_ref), img_ref, warped_src])
        cv2.imwrite(str(out_dir / "piecewise_overlay.png"), overlay)
        print(f"      Saved diagnostic overlay to {out_dir}/piecewise_overlay.png")
        
   
    print("[5/5] Rendering tie-line visualization...")
    if piecewise_result is not None and len(piecewise_result["tiles"]) > 0:
        pw_pts_ref = np.vstack([t["pts_ref"] for t in piecewise_result["tiles"]])
        pw_pts_src = np.vstack([t["pts_src"] for t in piecewise_result["tiles"]])
        pw_mask = np.concatenate([t["mask"] for t in piecewise_result["tiles"]])
        
        
        print("      Refining piecewise inliers to sub-pixel precision...")
        
        
        inlier_idx = np.where(pw_mask.flatten())[0]
        inliers_ref = pw_pts_ref[inlier_idx]
        inliers_src = pw_pts_src[inlier_idx]
        
        
        refined_ref, refined_src = refine_to_subpixel(img_ref, img_src, inliers_ref, inliers_src)
        
        np.savez(
            str(out_dir / "subpixel_inliers.npz"), 
            pts_ref=refined_ref, 
            pts_src=refined_src
        )
        print(f"      Saved sub-pixel refined inliers to {out_dir}/subpixel_inliers.npz")
        
        save_tie_line_visualization(
            img_ref, img_src, pw_pts_ref, pw_pts_src, pw_mask,
            str(out_dir / "piecewise_visualization.png"),
            title="Piecewise MAGSAC — Lunar Keypoint Registration"
        )
        print(f"Success! Found {np.sum(pw_mask)} inliers. Check {out_dir}/piecewise_visualization.png")
    else:
        print("Matching failed to find valid geometric consensus.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run SelenoMatch core engine on CLAHE images.")
    parser.add_argument("--ref-img", required=True, help="Path to reference CLAHE PNG")
    parser.add_argument("--src-img", required=True, help="Path to source CLAHE PNG")
    parser.add_argument("--out-dir", default="demo_output", help="Output directory")
    args = parser.parse_args()
    
    test_processed_pair(args.ref_img, args.src_img, args.out_dir)
