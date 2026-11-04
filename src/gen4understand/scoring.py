"""Verifier feature math (Appendix D). Inputs come from frozen model adapters."""

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.optimize import linear_sum_assignment

from .schema import SLOTS


def slot_atoms(event):
    atoms = set()
    for slot in SLOTS:
        value = event.get(slot)
        if value is None:
            continue
        for part in value if isinstance(value, list | tuple) else [value]:
            atoms.add((slot, str(part).strip().lower()))
    return atoms


def slot_f1(predicted, expected):
    a, b = slot_atoms(predicted), slot_atoms(expected)
    return 2 * len(a & b) / (len(a) + len(b)) if a or b else 0.0


def cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Features must have finite, nonzero norm")
    return float(np.clip(np.dot(a, b) / norm, -1, 1))


def mask_iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def matched_mask_iou(predicted, anchors):
    """Hungarian matching; missing/excess objects contribute zero."""
    if not len(predicted) or not len(anchors):
        return 0.0
    matrix = np.array([[mask_iou(a, b) for b in anchors] for a in predicted])
    rows, cols = linear_sum_assignment(-matrix)
    return float(matrix[rows, cols].sum() / max(len(predicted), len(anchors)))


def anchor_score(feature_pairs, mask_pairs, feature_weight=0.5):
    """Mean across endpoints. The feature/IoU blend is an explicit implementation choice."""
    if len(feature_pairs) != 2 or len(mask_pairs) != 2 or not 0 <= feature_weight <= 1:
        raise ValueError("Two endpoints and a valid blend weight are required")
    return float(
        np.mean(
            [
                feature_weight * cosine(*features) + (1 - feature_weight) * matched_mask_iou(*masks)
                for features, masks in zip(feature_pairs, mask_pairs, strict=False)
            ]
        )
    )


def _geometry(mask):
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError("Cannot derive geometry from an absent mask")
    return np.array([x.mean(), y.mean()]), (x.min(), y.min(), x.max() + 1, y.max() + 1)


def boundary_distance(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Masks must share H x W geometry")
    if not a.any() or not b.any():
        return float("inf")
    ba, bb = a & ~binary_erosion(a), b & ~binary_erosion(b)
    return float(distance_transform_edt(~bb)[ba].min() / np.hypot(*a.shape))


def relation_holds(a, b, relation):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Masks must share H x W geometry")
    if not a.any() or not b.any():
        return False
    ca, aa = _geometry(a)
    cb, ab = _geometry(b)
    height, width = a.shape
    if relation in {"touching", "held-by", "contact"}:
        return boundary_distance(a, b) <= 0.02
    if relation in {"inside", "in", "containment"}:
        x, y = np.rint(ca).astype(int)
        return bool(b[y, x] and np.logical_and(a, b).sum() / a.sum() >= 0.70)
    if relation in {"on", "support"}:
        gap = abs(ab[1] - aa[3]) / np.hypot(height, width)
        overlap = max(0, min(aa[2], ab[2]) - max(aa[0], ab[0]))
        return gap <= 0.03 and overlap / min(aa[2] - aa[0], ab[2] - ab[0]) >= 0.30
    signed = {
        "left-of": (cb[0] - ca[0]) / width,
        "right-of": (ca[0] - cb[0]) / width,
        "above": (cb[1] - ca[1]) / height,
        "below": (ca[1] - cb[1]) / height,
    }
    if relation not in signed:
        raise ValueError(f"Unsupported relation: {relation}")
    return bool(signed[relation] >= 0.05)


def window_relation(a, b, relation):
    if len(a) != 8 or len(b) != 8:
        raise ValueError("Physical checks require exactly eight frames per window")
    return sum(relation_holds(x, y, relation) for x, y in zip(a, b, strict=False)) >= 6


def release_holds(pre_a, pre_b, post_a, post_b):
    windows = (pre_a, pre_b, post_a, post_b)
    if any(len(x) != 8 for x in windows):
        raise ValueError("Release requires eight-frame boundary windows")
    distances = [
        np.array([boundary_distance(a, b) for a, b in zip(x, y, strict=False)])
        for x, y in ((pre_a, pre_b), (post_a, post_b))
    ]
    if not all(np.isfinite(d).all() for d in distances):
        return False
    pre, post = distances
    return bool(
        (pre <= 0.02).sum() >= 6 and (post > 0.02).sum() >= 6 and post.mean() - pre.mean() >= 0.03
    )


def physical_score(checks):
    if not checks:
        raise ValueError("No applicable physical checks; score cannot be fabricated")
    if any(not isinstance(value, bool | np.bool_) for value in checks):
        raise ValueError("Physical checks must be booleans, not strings or scores")
    return float(np.mean(checks))


def temporal_score(warp_error, centroids):
    points = np.asarray(centroids, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("At least three finite normalized 2D centroids are required")
    if not np.isfinite(warp_error) or warp_error < 0:
        raise ValueError("Invalid photometric warp error")
    acceleration = np.linalg.norm(np.diff(points, n=2, axis=0), axis=1).mean()
    return float(0.7 * np.exp(-warp_error / 0.05) + 0.3 * np.exp(-acceleration / 0.015))


def score_observation(observation, event):
    """Score model-exported measurements; quality is raw official DOVER output."""
    return {
        "sem": slot_f1(observation["parsed_event"], event),
        "anc": anchor_score(
            observation["feature_pairs"],
            observation["mask_pairs"],
            observation.get("anchor_feature_weight", 0.5),
        ),
        "phys": physical_score(observation["physical_checks"]),
        "tmp": temporal_score(observation["warp_error"], observation["centroids"]),
        "qual": float(observation["dover_score"]),
    }
