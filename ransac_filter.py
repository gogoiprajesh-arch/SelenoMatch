"""
Stage 8: RANSAC Filter
--------------------------
Geometric verification of the fused match list. Two strategies, both kept
(this mirrors Ransac.ipynb's "run both, compare" approach rather than
picking one blind):

  * global      -- one homography across the whole overlap region.
  * piecewise   -- one homography per tile, blended together. More
                    forgiving of local terrain relief / parallax that a
                    single global homography can't represent.

Works entirely on in-memory arrays -- no .npz/.jsonl round-trip required,
though `evaluate_global_from_npz` / `load_tile_homographies_from_jsonl` are
kept as thin compatibility wrappers in case you still want to run this
stage standalone against files written by an earlier run.
"""
import json
import cv2
import numpy as np


# ----------------------------------------------------------------------
# Shared utilities
# ----------------------------------------------------------------------

def compute_rmse(pts1, pts2, matrix, mask):
    """RMSE of reprojection error, computed only over the accepted inliers."""
    if mask is None or np.sum(mask) == 0:
        return None
    inlier_mask = mask.ravel().astype(bool)
    src = pts2[inlier_mask].reshape(-1, 1, 2).astype(np.float32)
    dst = pts1[inlier_mask].reshape(-1, 1, 2).astype(np.float32)
    projected = cv2.perspectiveTransform(src, matrix).reshape(-1, 2)
    errors = np.linalg.norm(projected - dst.reshape(-1, 2), axis=1)
    return float(np.sqrt(np.mean(errors ** 2)))


def save_overlay(ref_img, aligned_img, out_path):
    """Green = reference, Red = aligned source. Misalignment shows as red/green fringing."""
    overlay = np.zeros((*ref_img.shape, 3), dtype=np.uint8)
    overlay[..., 1] = ref_img
    overlay[..., 2] = aligned_img
    cv2.imwrite(out_path, overlay)


# ----------------------------------------------------------------------
# Global approach
# ----------------------------------------------------------------------

def ransac_global(pts1, pts2, conf, conf_threshold=0.30, proj_threshold=5.0):
    valid_mask = conf > conf_threshold
    pts1, pts2 = pts1[valid_mask], pts2[valid_mask]
    matrix, mask = cv2.findHomography(
        pts2, pts1, cv2.USAC_MAGSAC, proj_threshold, maxIters=50000, confidence=0.999
    )
    return matrix, mask, valid_mask


def evaluate_global(pts1, pts2, conf, ref_img, src_img,
                     conf_threshold=0.20, proj_threshold=3.3, out_prefix=None):
    print(f"Evaluating (GLOBAL): {len(pts1)} candidate matches")
    try:
        matrix, mask, valid_mask = ransac_global(pts1, pts2, conf, conf_threshold, proj_threshold)
    except cv2.error as e:
        print(f"  -> RANSAC failed: {e}")
        return None

    pts1_v, pts2_v = pts1[valid_mask], pts2[valid_mask]
    total_tested = len(pts1_v)
    inliers = int(np.sum(mask)) if mask is not None else 0
    rmse = compute_rmse(pts1_v, pts2_v, matrix, mask) if matrix is not None else None
    inlier_ratio = float(inliers / total_tested) if total_tested else 0.0

    print(f"  -> GLOBAL inliers: {inliers}/{total_tested} | ratio: {inlier_ratio:.3f}"
          + (f" | RMSE: {rmse:.2f}px" if rmse is not None else ""))

    aligned = None
    if matrix is not None and ref_img is not None and src_img is not None:
        h, w = ref_img.shape[:2]
        aligned = cv2.warpPerspective(src_img, matrix, (w, h))
        if out_prefix:
            cv2.imwrite(f"{out_prefix}_warped.png", aligned)
            save_overlay(ref_img, aligned, f"{out_prefix}_overlay.png")

    return {
        "matrix": matrix, "mask": mask, "inlier_count": inliers,
        "total_tested": total_tested, "inlier_ratio": inlier_ratio,
        "rmse": rmse, "aligned": aligned,
        "pts1_valid": pts1_v, "pts2_valid": pts2_v,
    }


def evaluate_global_from_npz(npz_path, ref_img_path, src_img_path, **kwargs):
    """Compatibility wrapper matching the original notebook's file-based entry point."""
    data = np.load(npz_path)
    ref_img = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
    src_img = cv2.imread(src_img_path, cv2.IMREAD_GRAYSCALE)
    return evaluate_global(data['pts1'], data['pts2'], data['conf'], ref_img, src_img, **kwargs)


# ----------------------------------------------------------------------
# Piecewise approach
# ----------------------------------------------------------------------

def _fit_tile_homography(pts_ref, pts_src, proj_threshold=5.0):
    if len(pts_ref) < 10:
        return None, None

    pts_ref_arr = np.array(pts_ref, dtype=np.float32)
    pts_src_arr = np.array(pts_src, dtype=np.float32)

    H, mask = cv2.findHomography(pts_src_arr, pts_ref_arr, cv2.USAC_MAGSAC, proj_threshold)
    if H is None:
        return None, None

    det = np.linalg.det(H[0:2, 0:2])
    if det < 0.1 or det > 10.0:
        return None, None

    return H, mask


def build_tile_homographies(tile_matches, conf_thresh=0.15, min_inlier_ratio=0.05):
    tiles = []
    for t in tile_matches:
        conf = np.asarray(t["conf"])
        keep = conf >= conf_thresh
        p_ref = np.asarray(t["pts_ref"])[keep]
        p_src = np.asarray(t["pts_src"])[keep]
        
        # Grab the source labels and apply the same confidence filter
        p_source = np.asarray(t.get("source", []))
        if len(p_source) > 0:
            p_source = p_source[keep]

        H, mask = _fit_tile_homography(p_ref, p_src)
        n_tested = len(p_ref)
        
        if H is None or n_tested == 0:
            print(f"  Tile (y={t['y_off']}, x={t['x_off']}): insufficient matches, skipped")
            continue

        n_inliers = int(np.sum(mask))
        t_rmse = compute_rmse(p_ref, p_src, H, mask)
        ratio = n_inliers / n_tested

        # Calculate RIFT vs LoFTR inlier breakdown
        rift_count, loftr_count = 0, 0
        if len(p_source) == len(mask):
            inlier_sources = p_source[mask.ravel().astype(bool)]
            rift_count = np.sum(inlier_sources == "rift")
            loftr_count = np.sum(inlier_sources == "loftr")
        
        source_log = f" [Inliers -> RIFT2: {rift_count} | LoFTR: {loftr_count}]" if len(p_source) > 0 else ""

        if ratio >= min_inlier_ratio:
            print(f"  Tile (y={t['y_off']}, x={t['x_off']}): {n_inliers}/{n_tested} inliers "
                  f"({ratio:.3f}){source_log} | RMSE: {t_rmse:.2f}px")
            tiles.append({
                "H": H, "y_off": t["y_off"], "x_off": t["x_off"],
                "inlier_ratio": ratio, "n_inliers": n_inliers, "mask": mask,
                "pts_ref": p_ref, "pts_src": p_src, "rmse": t_rmse,
            })
        else:
            print(f"  Tile (y={t['y_off']}, x={t['x_off']}): {n_inliers}/{n_tested} inliers "
                  f"({ratio:.3f}){source_log} [SKIPPED]")
    return tiles


def load_tile_homographies_from_jsonl(jsonl_path, conf_thresh, min_inlier_ratio=0.1):
    """Compatibility wrapper for the original jsonl-on-disk format."""
    tile_matches = []
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            y_off, x_off = data["patch_y_offset"], data["patch_x_offset"]
            pts_ref, pts_src, conf = [], [], []
            for m in data.get("matches", []):
                pts_ref.append([m["ref_pt"][1] + x_off, m["ref_pt"][0] + y_off])
                pts_src.append([m["src_pt"][1] + x_off, m["src_pt"][0] + y_off])
                conf.append(m.get("confidence", 1.0))
            tile_matches.append({"y_off": y_off, "x_off": x_off,
                                  "pts_ref": pts_ref, "pts_src": pts_src, "conf": conf})
    return build_tile_homographies(tile_matches, conf_thresh, min_inlier_ratio)


def piecewise_warp_blended(ref_img, src_img, tiles, patch_size=1024, feather=48):
    h, w = ref_img.shape[:2]
    accum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    ramp = np.ones(patch_size, dtype=np.float32)
    if feather > 0:
        edge = np.linspace(0, 1, feather, dtype=np.float32)
        ramp[:feather] = edge
        ramp[-feather:] = edge[::-1]
    tile_weight = np.outer(ramp, ramp)

    for t in tiles:
        H, y_off, x_off = t["H"], t["y_off"], t["x_off"]
        warped_full = cv2.warpPerspective(src_img, H, (w, h))

        y2 = min(h, y_off + patch_size)
        x2 = min(w, x_off + patch_size)
        th, tw = y2 - y_off, x2 - x_off
        if th <= 0 or tw <= 0:
            continue

        w_crop = tile_weight[:th, :tw]
        accum[y_off:y2, x_off:x2] += warped_full[y_off:y2, x_off:x2].astype(np.float32) * w_crop
        weight_sum[y_off:y2, x_off:x2] += w_crop

    result = np.zeros((h, w), dtype=np.uint8)
    valid = weight_sum > 1e-6
    result[valid] = np.clip(accum[valid] / weight_sum[valid], 0, 255).astype(np.uint8)
    return result, valid


def evaluate_piecewise(tile_matches, ref_img, src_img, patch_size=1024,
                        conf_threshold=0.05, min_inlier_ratio=0.1, out_prefix=None):
    print(f"\nEvaluating (PIECEWISE): {len(tile_matches)} tiles")
    tiles = build_tile_homographies(tile_matches, conf_threshold, min_inlier_ratio)
    print(f"  Usable tiles: {len(tiles)}")
    if not tiles:
        print("  -> No usable tiles, cannot proceed.")
        return None

    blended, valid_mask = piecewise_warp_blended(ref_img, src_img, tiles, patch_size=patch_size)
    if out_prefix:
        cv2.imwrite(f"{out_prefix}_warped.png", blended)

    rmses = [t["rmse"] for t in tiles if t["rmse"] is not None]
    med_rmse = float(np.median(rmses)) if rmses else 0.0
    avg_rmse = float(np.mean(rmses)) if rmses else 0.0
    med_ratio = float(np.median([t["inlier_ratio"] for t in tiles]))
    avg_ratio = float(np.mean([t["inlier_ratio"] for t in tiles]))

    print(f"  median inlier ratio: {med_ratio:.3f} | median RMSE: {med_rmse:.2f}px")
    print(f"  avg inlier ratio: {avg_ratio:.3f} | avg RMSE: {avg_rmse:.2f}px")

    return {
        "tiles": tiles, "blended": blended, "valid_mask": valid_mask,
        "tiles_used": len(tiles), "avg_tile_inlier_ratio": avg_ratio,
        "median_tile_inlier_ratio": med_ratio, "avg_tile_rmse": avg_rmse,
        "median_tile_rmse": med_rmse,
    }