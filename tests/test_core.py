from dataclasses import replace

import numpy as np
import pytest

from gen4understand.metrics import (
    hallucination_metrics,
    paired_bootstrap,
    structural_metrics,
    wilson,
)
from gen4understand.schema import Event
from gen4understand.scoring import (
    matched_mask_iou,
    relation_holds,
    release_holds,
    temporal_score,
    window_relation,
)
from gen4understand.scripts import hard_negatives, qa_examples
from gen4understand.splits import validate_splits
from gen4understand.verifier import Calibrator, Candidate, calibrate
from gen4understand.video import concatenate, evidence_probe


@pytest.fixture
def event():
    return Event(
        "e1",
        "r1",
        "train",
        "right gripper",
        "place",
        "brush",
        ("held-by(gripper)",),
        "downward",
        ("inside(basket)",),
        "before:e2",
        0.0,
        1.0,
        "source.mp4",
        "s1",
        "o1",
        ("brush", "cup", "water"),
    )


def candidate(key="a", event="e1", scores=(0.9,) * 5, presence=1.0, split="train", audit=True):
    return Candidate(key, event, "clip.mp4", scores, presence, split, audit)


def test_missing_tracks_cannot_win():
    c = Calibrator((0.0,) * 5, (1.0,) * 5)
    selected = c.select(
        [candidate("a", scores=(1.0,) * 5, presence=0.79), candidate("b", presence=0.8)]
    )
    assert selected["e1"][0].candidate_id == "b"
    assert len(selected) == 1


def test_threshold_and_out_of_range_scores():
    c = Calibrator((0.0,) * 5, (1.0,) * 5)
    assert c.score(candidate(scores=(2.0,) * 5)) == pytest.approx(1)
    assert not c.select([candidate(scores=(0.69,) * 5)])
    assert c.select([candidate(scores=(0.7,) * 5)])


def test_invalid_calibration_and_leakage():
    with pytest.raises(ValueError):
        calibrate([candidate(split="test")])
    with pytest.raises(ValueError):
        Calibrator((0.0,) * 5, (0.0,) * 5)
    c = Calibrator((0.0,) * 5, (1.0,) * 5, calibration_event_ids=("e1",))
    with pytest.raises(ValueError, match="Calibration event"):
        c.select([candidate()])


def test_calibration_search_audited_precision():
    rows = [candidate("good", "e1", (1.0,) * 5), candidate("bad", "e2", (0.0,) * 5, audit=False)]
    fitted = calibrate(rows)
    assert set(fitted.select(rows, allow_calibration=True)) == {"e1"}


def test_split_recording_scene_instance_leakage(event):
    for field in ("recording_id", "scene_id", "object_instance_id"):
        changes = dict(
            event_id="e2",
            recording_id="r2",
            scene_id="s2",
            object_instance_id="o2",
            split="test",
            action="release",
        )
        changes[field] = getattr(event, field)
        with pytest.raises(ValueError, match="leakage"):
            validate_splits([event, replace(event, **changes)])


def test_scene_negatives_exclude_observed_actions(event):
    assert "clean" in hard_negatives([event])
    assert "clean" not in hard_negatives([replace(event, action="clean")])
    assert len(qa_examples([event], "clip.mp4", "atom", "e1", "paper-budget")) == 2


def test_boundary_dedup_preserves_distinct_frames():
    a = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    b = np.ones_like(a)
    assert len(concatenate([a, b])) == 4
    b[0] = a[-1]
    assert len(concatenate([a, b])) == 3


def test_physical_windows_and_hungarian_matching():
    obj, container = np.zeros((100, 100), bool), np.zeros((100, 100), bool)
    obj[40:50, 40:50] = True
    container[30:70, 30:70] = True
    assert relation_holds(obj, container, "inside")
    assert matched_mask_iou([obj, container], [container, obj]) == 1
    pre = np.stack([obj] * 8)
    target = np.stack([container] * 8)
    pre[:2] = False
    assert window_relation(pre, target, "inside")
    pre[2] = False
    assert not window_relation(pre, target, "inside")
    assert not release_holds(pre, target, np.zeros_like(pre), target)


def test_temporal_score_and_probes():
    assert temporal_score(0, [[0, 0], [0.1, 0], [0.2, 0]]) == pytest.approx(1.0)
    assert temporal_score(0.1, [[0, 0], [0.9, 0], [0.1, 0]]) < 0.1
    frames = np.arange(20)
    shuffled = evidence_probe(frames, "shuffle-interior")
    assert shuffled[0] == frames[0] and shuffled[-1] == frames[-1]
    assert sorted(shuffled) == list(frames)


def test_metrics_and_parse_failures(event):
    row = {
        "expected": [event.to_dict()],
        "predicted": [event.to_dict()],
        "expected_order": {"e1:e2": True},
        "predicted_order": {"e1:e2": True},
        "expected_relations": {"inside": True},
        "predicted_relations": {"inside": True},
    }
    assert structural_metrics([row])["fg_score"] == 1
    row.update(predicted=[], predicted_order={}, predicted_relations={})
    assert structural_metrics([row])["fg_score"] == 0
    records = [
        {
            "predicted_actions": ["clean", "wipe"],
            "unsupported_actions": ["clean", "wipe"],
            "predicted_intents": [],
            "unsupported_intents": [],
        }
    ]
    assert hallucination_metrics(records) == {"far": 1, "fir": 0, "dr": 0.5}
    records[0]["parse_valid"] = False
    assert hallucination_metrics(records)["dr"] == 0


def test_uncertainty_matches_paper():
    assert wilson(89, 120) == pytest.approx((0.6567, 0.8116), abs=0.001)
    report = paired_bootstrap([1, 2, 3, 4], [0, 1, 2, 3], ["a", "a", "b", "b"], replicates=100)
    assert report["ci_95"] == [1.0, 1.0]
