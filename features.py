"""
Stage 6: Feature Extractors [Parallel Matching Architectures]
-----------------------------------------------------------------
Runs LoFTR (dense transformer matching) and RIFT-2 (phase-congruency,
radiation-invariant) on the *same* 1024x1024 macro-patch pair and hands
back two raw candidate match sets, both already expressed in the shared
1024x1024 macro-patch coordinate frame. Fusion happens one stage later
(fusion.py) -- this module's only job is "given a macro-patch pair, get
candidates from each architecture, in the same local coordinate space."

Macro/Micro-patching:
  * RIFT2Runner takes the 1024x1024 macro-patch whole. Phase congruency
    needs that wider spatial context for stable orientation histograms.
  * LoFTRRunner takes the same 1024x1024 macro-patch but internally slices
    it into four 512x512 micro-patch quadrants (TL, TR, BL, BR) to stay
    within GPU memory on smaller cards. Each quadrant is matched
    independently, and quadrant-local coordinates are translated back into
    the 1024x1024 parent frame (e.g. + (512, 0) for the TR quadrant)
    before the four results are concatenated. Callers of `.match()` never
    see the quadrant split -- they get back one unified array of keypoints
    per patch, already living in the 1024x1024 frame, exactly like
    RIFT2Runner's output. fusion.py can treat both branches identically.

Patches arriving here are assumed to already be CLAHE-enhanced uint8
grayscale (i.e. they came out of preprocess.py + patches.py), so neither
branch re-applies its own contrast enhancement.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from RIFT2 import RIFT2
from matcher_functions import match_keypoints_nn


class LoFTRRunner:
    """
    Loads LoFTR weights once; call .match(macro_patch_ref, macro_patch_src)
    per 1024x1024 macro-patch pair. Internally splits each macro-patch into
    four 512x512 quadrants to bound GPU memory use, and returns results
    re-stitched into the 1024x1024 macro-patch frame.
    """

    QUADRANT_OFFSETS = [(0, 0), (0, 512), (512, 0), (512, 512)]  # (y_off, x_off) of TL, TR, BL, BR

    def __init__(self, loftr_repo_dir, weights_path, conf_thresh=0.85,
                 micro_patch_size=512, quadrant_black_fraction=0.1):
        repo_dir = Path(loftr_repo_dir).expanduser().resolve()
        loftr_src = repo_dir / "src"
        loftr_pkg = loftr_src / "loftr"
        weights_path = Path(weights_path).expanduser().resolve()

        if not loftr_pkg.exists():
            raise FileNotFoundError(
                f"Couldn't find the LoFTR package at '{loftr_pkg}'.\n"
                f"  --loftr-repo should point at the *root* of the cloned "
                f"zju3dv/LoFTR repo (the folder containing 'src/loftr', "
                f"'weights/', 'environment.yaml', etc.) -- got '{repo_dir}'.\n"
                f"  If you're running from the lunar_dataset workspace root "
                f"where LoFTR/ lives right next to run_pipeline.py, "
                f"'./LoFTR' (the default) is correct."
            )
        if not weights_path.exists():
            raise FileNotFoundError(
                f"LoFTR weights not found at '{weights_path}'. Download a "
                f"checkpoint (e.g. outdoor_ds.ckpt) into "
                f"'{repo_dir / 'weights'}' and point --loftr-weights at it."
            )

        if str(loftr_src) not in sys.path:
            sys.path.insert(0, str(loftr_src))
        from loftr import LoFTR, default_cfg  # noqa: import deferred until path is set

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_thresh = conf_thresh
        self.micro_patch_size = micro_patch_size
        self.quadrant_black_fraction = quadrant_black_fraction

        self.matcher = LoFTR(config=default_cfg)
        checkpoint = torch.load(str(weights_path), map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.matcher.load_state_dict(state_dict, strict=False)
        self.matcher.eval().to(self.device)

    def _to_tensor(self, patch):
        h, w = patch.shape
        # LoFTR requires dims divisible by 8; 512x512 patches already are,
        # but guard against odd input sizes anyway.
        new_w, new_h = w - (w % 8), h - (h % 8)
        if new_w != w or new_h != h:
            patch = cv2.resize(patch, (new_w, new_h))
        return torch.from_numpy(patch)[None][None].float().to(self.device) / 255.0

    def _match_quadrant(self, quad_ref, quad_src):
        """Raw single-quadrant LoFTR inference, in that quadrant's own local frame."""
        t0 = self._to_tensor(quad_ref)
        t1 = self._to_tensor(quad_src)
        with torch.inference_mode():
            batch = {"image0": t0, "image1": t1}
            self.matcher(batch)
        pts_ref = batch["mkpts0_f"].cpu().numpy()
        pts_src = batch["mkpts1_f"].cpu().numpy()
        conf = batch["mconf"].cpu().numpy()
        mask = conf > self.conf_thresh
        return pts_ref[mask], pts_src[mask], conf[mask]

    def match(self, macro_patch_ref, macro_patch_src):
        """
        Takes a 1024x1024 macro-patch pair, internally matches it as four
        512x512 quadrants, and returns one unified (pts_ref, pts_src, conf)
        triple with all coordinates already translated into the
        1024x1024 macro-patch frame -- same shape/contract as
        RIFT2Runner.match(), so fusion.py doesn't need to know a quadrant
        split ever happened.
        """
        s = self.micro_patch_size
        all_pts_ref, all_pts_src, all_conf = [], [], []

        for qy, qx in self.QUADRANT_OFFSETS:
            quad_ref = macro_patch_ref[qy:qy + s, qx:qx + s]
            quad_src = macro_patch_src[qy:qy + s, qx:qx + s]

            # Skip quadrants that fall off the edge of a smaller-than-1024
            # macro-patch, or that are almost entirely no-data.
            if quad_ref.shape != (s, s) or quad_src.shape != (s, s):
                continue
            if cv2.countNonZero(quad_ref) < (s * s * self.quadrant_black_fraction):
                continue

            pts_ref_q, pts_src_q, conf_q = self._match_quadrant(quad_ref, quad_src)
            if len(pts_ref_q) == 0:
                continue

            offset = np.array([qx, qy], dtype=np.float32)  # points are (x, y)
            all_pts_ref.append(pts_ref_q + offset)
            all_pts_src.append(pts_src_q + offset)
            all_conf.append(conf_q)

        if not all_pts_ref:
            print("LoFTR total candidates: 0")
            empty = np.zeros((0, 2), dtype=np.float32)
            return empty, empty, np.zeros((0,), dtype=np.float32)

        final_pts_ref = np.vstack(all_pts_ref).astype(np.float32)
        final_pts_src = np.vstack(all_pts_src).astype(np.float32)
        final_conf = np.concatenate(all_conf).astype(np.float32)

        print(f"LoFTR total candidates: {len(final_pts_ref)}")

        return final_pts_ref, final_pts_src, final_conf


class RIFT2Runner:
    """
    Loads RIFT-2 config once; call .match(macro_patch_ref, macro_patch_src)
    per 1024x1024 macro-patch pair. No internal splitting -- the whole
    macro-patch is processed in one shot, which is exactly what phase
    congruency wants for stable orientation histograms.
    """

    def __init__(self, config_file=None, lowes_ratio=0.90, **rift_params):
        self.rift2 = RIFT2(config_file=config_file, **rift_params)
        self.lowes_ratio = lowes_ratio

    def match(self, macro_patch_ref, macro_patch_src):
        kp1, des1, kp2, des2 = self.rift2(macro_patch_ref, macro_patch_src)
        pts_ref, pts_src, conf = match_keypoints_nn(
            des1, des2, kp1, kp2, lowes_ratio=self.lowes_ratio, mutual=True
        )
        return pts_ref, pts_src, np.array(conf, dtype=np.float32)