"""
Stage 9: Sub-Pixel Alignment
--------------------------------
NOTE: master_pipeline.ipynb calls `%run subPixelAllignment.ipynb` for this
stage, but that notebook wasn't among the uploaded files, so this is a
clean reimplementation of the same idea rather than a port: refine each
inlier keypoint's integer-pixel location using the local intensity
gradient covariance around it (`cv2.cornerSubPix`), independently in each
image. If you do have the original notebook, drop its function in here in
place of `refine_to_subpixel` and the rest of the pipeline won't need to
change.
"""
import cv2
import numpy as np


def refine_to_subpixel(img_ref, img_src, pts_ref, pts_src,
                        win_size=(5, 5), zero_zone=(-1, -1), max_iter=40, eps=0.001):
    """
    pts_ref, pts_src: Nx2 float arrays of (x, y) matched keypoints, refined
    independently in their own image using that image's local gradients.
    """
    if len(pts_ref) == 0:
        return pts_ref, pts_src

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)

    pts_ref_in = pts_ref.astype(np.float32).reshape(-1, 1, 2).copy()
    pts_src_in = pts_src.astype(np.float32).reshape(-1, 1, 2).copy()

    refined_ref = cv2.cornerSubPix(img_ref, pts_ref_in, win_size, zero_zone, criteria)
    refined_src = cv2.cornerSubPix(img_src, pts_src_in, win_size, zero_zone, criteria)

    return refined_ref.reshape(-1, 2), refined_src.reshape(-1, 2)
