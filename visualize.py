"""
Stage 10: Match Visualization
---------------------------------
Side-by-side reference/source images connected by tie-lines: green for
RANSAC inliers, red for rejected outliers. Works for either the global
match set or the concatenated piecewise tile matches.
"""
import cv2
import numpy as np


def save_tie_line_visualization(ref_img, src_img, pts_ref, pts_src, mask,
                                 out_path, scale=0.10, max_lines=8000,
                                 title="Lunar Keypoint Registration"):
    h, w = ref_img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)

    # Convert grayscale to BGR for color drawing
    ref_bgr = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
    src_bgr = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)

    # Scale the background images DOWN to match the points
    ref_bgr = cv2.resize(ref_bgr, (new_w, new_h))
    src_bgr = cv2.resize(src_bgr, (new_w, new_h))

    # Create the black separator using the NEW scaled height
    gap_width = 50 
    separator = np.zeros((new_h, gap_width, 3), dtype=np.uint8)

    # Now concatenate
    vis = cv2.hconcat([ref_bgr, separator, src_bgr])
    pts_ref = np.asarray(pts_ref)
    pts_src = np.asarray(pts_src)
    mask = np.asarray(mask).ravel().astype(bool) if mask is not None else np.zeros(len(pts_ref), dtype=bool)

    total_pts = len(pts_ref)
    inliers_count = int(np.sum(mask))
    outliers_count = total_pts - inliers_count
    ratio = (inliers_count / total_pts * 100) if total_pts else 0

    if total_pts > max_lines:
        idx = np.random.choice(total_pts, max_lines, replace=False)
        plot_ref, plot_src, plot_mask = pts_ref[idx], pts_src[idx], mask[idx]
    else:
        plot_ref, plot_src, plot_mask = pts_ref, pts_src, mask

    for i in range(len(plot_ref)):
        # --- NEW: Skip drawing the red outlier lines completely ---
        if not plot_mask[i]:
            continue 
        # ----------------------------------------------------------

        pt1 = (int(plot_ref[i][0] * scale), int(plot_ref[i][1] * scale))
        pt2 = (int(plot_src[i][0] * scale) + new_w + gap_width, int(plot_src[i][1] * scale))
        
        # Only green lines will reach this point
        cv2.line(vis, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(vis, pt1, 2, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, pt2, 2, (0, 255, 255), -1, cv2.LINE_AA)

    banner_h = 80
    banner = np.ones((banner_h, vis.shape[1], 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX

    def put_centered_text(img, text, y_pos, font_scale, thickness):
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        cv2.putText(img, text, (text_x, y_pos), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    put_centered_text(banner, title, 25, 0.7, 2)
    put_centered_text(banner, f"Total Points: {total_pts} | Inliers (Green): {inliers_count} | "
                               f"Outliers (Red): {outliers_count} | Ratio: {ratio:.2f}%", 55, 0.5, 1)

    final_out = cv2.vconcat([banner, vis])
    cv2.imwrite(out_path, final_out)
    print(f"[visualize] Saved {out_path} ({inliers_count}/{total_pts} inliers, {ratio:.2f}%)")

# debug block
if __name__ == "__main__":
    from pathlib import Path
    from ransac_filter import ransac_global

    print("Loading cached 25-minute feature data from disk...")
    out_dir = Path("output")
    
    # Load the preprocessed images
    ref_img = cv2.imread(str(out_dir / "reference_clahe.png"), cv2.IMREAD_GRAYSCALE)
    src_img = cv2.imread(str(out_dir / "source_clahe.png"), cv2.IMREAD_GRAYSCALE)
    
    # Load the fused RIFT-2 and LoFTR points
    data = np.load(out_dir / "fused_matches.npz")
    pts1, pts2, conf = data["pts1"], data["pts2"], data["conf"]

    print("Running rapid RANSAC geometry...")
    # This uses the ransac_global function (make sure you swapped pts1/pts2 in that file!)
    matrix, mask, valid_mask = ransac_global(pts1, pts2, conf)

    print("Drawing visualization...")
    out_path = str(out_dir / "debug_visualization.png")
    save_tie_line_visualization(
        ref_img, src_img, pts1[valid_mask], pts2[valid_mask], mask,
        out_path, title="Standalone Debug Visualization"
    )