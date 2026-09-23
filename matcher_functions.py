import cv2
import numpy as np
from matplotlib import pyplot as plt

def match_keypoints_nn(des1, des2, kp1, kp2, lowes_ratio=0.85, mutual=False):
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), []

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches1 = bf.knnMatch(des1, des2, k=2)

    ratios = [m.distance / n.distance for m, n in matches1 if n.distance > 0]
    if ratios:
        print(f"Best ratio: {min(ratios):.3f} | Median ratio: {np.median(ratios):.3f} | "
              f"Ratios under {lowes_ratio}: {sum(r < lowes_ratio for r in ratios)}/{len(ratios)}")
    else:
        print("No knn pairs returned at all")

    mutual_matches = []
    confidences = []

    if mutual:
        matches2 = bf.knnMatch(des2, des1, k=2)
        good_matches2_set = {
            (m.trainIdx, m.queryIdx) 
            for m, n in matches2 
            if m.distance < lowes_ratio * n.distance
        }
        for m, n in matches1:
            if m.distance < lowes_ratio * n.distance:
                if (m.queryIdx, m.trainIdx) in good_matches2_set:
                    mutual_matches.append(m)
                    # Convert distance ratio to a 0.0 - 1.0 confidence score
                    confidences.append(1.0 - (m.distance / n.distance))
    else:
        for m, n in matches1:
            if m.distance < lowes_ratio * n.distance:
                mutual_matches.append(m)
                confidences.append(1.0 - (m.distance / n.distance))

    if not mutual_matches:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), []

    points1 = np.array([kp1[m.queryIdx].pt for m in mutual_matches], dtype=np.float32)
    points2 = np.array([kp2[m.trainIdx].pt for m in mutual_matches], dtype=np.float32)

    return points1, points2, confidences


def outlier_removal(points1, points2):
    if len(points1) < 4:
        return [], [], []
        
    H, mask = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC, 5.0)
    if mask is None:
        return [], [], []
        
    matchesMask = mask.ravel().tolist()

    inliers1 = [points1[i] for i in range(len(points1)) if matchesMask[i]]
    inliers2 = [points2[i] for i in range(len(points2)) if matchesMask[i]]

    return inliers1, inliers2, matchesMask


def draw_matches(img1, img2, kp1, kp2, mutual_matches, matchesMask):
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, mutual_matches, None, **draw_params)
    num_inliers = np.sum(matchesMask) if matchesMask else 0
    num_outliers = len(mutual_matches) - num_inliers
    print(f'Number of kp1: {len(kp1)}')
    print(f'Number of kp2: {len(kp2)}')
    print(f'Number of matches with N.N : {len(mutual_matches)}')
    print(f'Number of inliers after MAGSAC: {num_inliers}')
    print(f'Number of outliers after MAGSAC: {num_outliers}')
    plt.imshow(img3), plt.show()