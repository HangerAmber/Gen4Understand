"""Leakage checks before generation and adaptation (Appendix E.1)."""

import hashlib

from .schema import SLOTS


def partition_recordings(recording_ids, seed=0, train=0.8, val=0.1):
    if not 0 < train < 1 or not 0 <= val < 1 or train + val >= 1:
        raise ValueError("Invalid partition fractions")
    result = {}
    for key in sorted(set(recording_ids)):
        u = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:16], 16) / 2**64
        result[key] = "train" if u < train else "val" if u < train + val else "test"
    return result


def validate_splits(events):
    ids, owners, signatures = set(), {}, {}
    for event in events:
        if event.event_id in ids:
            raise ValueError(f"Duplicate event ID: {event.event_id}")
        ids.add(event.event_id)
        for field in ("recording_id", "scene_id", "object_instance_id"):
            key = (field, getattr(event, field))
            if key in owners and owners[key] != event.split:
                raise ValueError(f"Cross-partition leakage: {field}={key[1]}")
            owners[key] = event.split
        signature = tuple(str(getattr(event, field)) for field in SLOTS) + (
            str(event.pre_state),
            event.motion,
        )
        if signature in signatures and signatures[signature] != event.split:
            raise ValueError("Cross-partition duplicate event tuple")
        signatures[signature] = event.split


def screen_endpoints(entries, threshold=0.995):
    """Return cross-split cosine near-duplicates; caller must adjudicate/remove."""
    import numpy as np

    matches = []
    for i, left in enumerate(entries):
        for right in entries[i + 1 :]:
            if left["split"] == right["split"]:
                continue
            a, b = np.asarray(left["feature"]), np.asarray(right["feature"])
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            if norm == 0 or not np.isfinite(norm):
                raise ValueError("Endpoint features must be finite nonzero vectors")
            similarity = float(np.dot(a, b) / norm)
            if similarity >= threshold:
                matches.append((left["id"], right["id"], similarity))
    return matches
