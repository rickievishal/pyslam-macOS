"""Pure Python fallback for the small pybind11 ``pyslam_utils`` API.

The native helper is useful on Linux, but linking it against Homebrew OpenCV
alongside pip's macOS OpenCV wheel loads two incompatible OpenCV runtimes.
These implementations keep the Python SLAM core usable on macOS without that
duplicate shared-library stack.
"""

from __future__ import annotations

import cv2
import numpy as np


def _match_pair(match_pair, ratio_test):
    if len(match_pair) < 2:
        return None
    first, second = match_pair[0], match_pair[1]
    if first.distance < ratio_test * second.distance:
        return first.queryIdx, first.trainIdx
    return None


def good_matches_simple(matches, ratio_test=0.7):
    pairs = [_match_pair(pair, ratio_test) for pair in matches]
    pairs = [pair for pair in pairs if pair is not None]
    if not pairs:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    return np.asarray(pairs, dtype=np.int32).T


def good_matches_one_to_one(matches, ratio_test=0.7):
    best_by_train = {}
    for pair in matches:
        selected = _match_pair(pair, ratio_test)
        if selected is None:
            continue
        query_idx, train_idx = selected
        distance = pair[0].distance
        previous = best_by_train.get(train_idx)
        if previous is None or distance < previous[0]:
            best_by_train[train_idx] = (distance, query_idx)
    pairs = [(query_idx, train_idx) for _, query_idx in best_by_train.values()]
    if not pairs:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    return np.asarray(pairs, dtype=np.int32).T


def _keypoint_xy(keypoints):
    if isinstance(keypoints, np.ndarray):
        return np.asarray(keypoints, dtype=np.float32)
    return np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32)


def row_matches_np(kps1_np, kps2_np, matches, max_distance, max_row_distance, max_disparity):
    kps1 = _keypoint_xy(kps1_np)
    kps2 = _keypoint_xy(kps2_np)
    kept = []
    for match in matches:
        query_idx, train_idx = int(match[0]), int(match[1])
        if query_idx >= len(kps1) or train_idx >= len(kps2):
            continue
        delta = kps1[query_idx] - kps2[train_idx]
        if np.linalg.norm(delta) <= max_distance and abs(delta[1]) <= max_row_distance and abs(delta[0]) <= max_disparity:
            kept.append((query_idx, train_idx))
    return np.asarray(kept, dtype=np.int32).reshape((-1, 2))


def row_matches_with_ratio_test_np(kps1_np, kps2_np, knn_matches, max_distance, max_row_distance, max_disparity, ratio_test):
    good = []
    for pair in knn_matches:
        selected = _match_pair(pair, ratio_test)
        if selected is not None:
            good.append(selected)
    return row_matches_np(kps1_np, kps2_np, good, max_distance, max_row_distance, max_disparity)


def filter_non_row_matches_np(kps1_np, kps2_np, idxs1_np, idxs2_np, max_row_distance, max_disparity):
    kps1 = _keypoint_xy(kps1_np)
    kps2 = _keypoint_xy(kps2_np)
    idxs1 = np.asarray(idxs1_np, dtype=np.int32)
    idxs2 = np.asarray(idxs2_np, dtype=np.int32)
    valid = (
        (idxs1 >= 0)
        & (idxs1 < len(kps1))
        & (idxs2 >= 0)
        & (idxs2 < len(kps2))
    )
    valid_indices = np.flatnonzero(valid)
    delta = kps1[idxs1[valid]] - kps2[idxs2[valid]]
    keep = (np.abs(delta[:, 1]) <= max_row_distance) & (np.abs(delta[:, 0]) <= max_disparity)
    return idxs1[valid][keep], idxs2[valid][keep]


def extract_patches(image, keypoints, patch_size, use_orientation=True, scale_factor=1.0, warp_flags=None):
    if warp_flags is None:
        warp_flags = cv2.WARP_INVERSE_MAP + cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS
    patches = []
    for keypoint in keypoints:
        scale = scale_factor * keypoint.size / patch_size
        angle = np.deg2rad(keypoint.angle) if use_orientation and keypoint.angle >= 0 else 0.0
        cosine, sine = np.cos(angle), np.sin(angle)
        matrix = np.array(
            [
                [scale * cosine, -scale * sine, (-scale * cosine + scale * sine) * patch_size / 2 + keypoint.pt[0]],
                [scale * sine, scale * cosine, (-scale * sine - scale * cosine) * patch_size / 2 + keypoint.pt[1]],
            ],
            dtype=np.float32,
        )
        patches.append(cv2.warpAffine(image, matrix, (patch_size, patch_size), flags=warp_flags))
    return patches


def extract_mean_colors(image, img_coords, delta, default_color):
    image = np.asarray(image)
    coords = np.asarray(img_coords, dtype=np.int32)
    height, width = image.shape[:2]
    output = []
    for x, y in coords:
        x0, x1 = max(0, x - delta), min(width, x + delta + 1)
        y0, y1 = max(0, y - delta), min(height, y + delta + 1)
        patch = image[y0:y1, x0:x1]
        output.append(np.mean(patch.reshape(-1, image.shape[2]), axis=0) if patch.size else default_color)
    return np.asarray(output, dtype=np.uint8)


def filter_2d_keypoints_by_mask(kps_np, mask_np):
    points = np.asarray(kps_np)
    mask = np.asarray(mask_np)
    height, width = mask.shape[:2]
    valid = (
        (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )
    valid_indices = np.flatnonzero(valid)
    valid[valid_indices] &= mask[points[valid_indices, 1].astype(int), points[valid_indices, 0].astype(int)] != 0
    return points[valid]


def filter_cv2_keypoints_by_mask(keypoints, mask_np):
    mask = np.asarray(mask_np)
    height, width = mask.shape[:2]
    return [
        keypoint
        for keypoint in keypoints
        if 0 <= int(keypoint.pt[0]) < width
        and 0 <= int(keypoint.pt[1]) < height
        and mask[int(keypoint.pt[1]), int(keypoint.pt[0])] != 0
    ]
