"""Structural, hallucination, and control metrics; all values are fractions."""

import math
import re

import numpy as np

from .schema import SLOTS
from .scoring import slot_atoms


def parse_slots(text):
    """Strict deterministic slot lines; malformed/missing fields stay missing."""
    result = {}
    for slot in SLOTS:
        hits = re.findall(rf"^{slot}\s*:\s*(.+)$", text, flags=re.MULTILINE | re.IGNORECASE)
        if len(hits) == 1:
            result[slot] = (
                [s.strip() for s in hits[0].split(";")] if slot == "post_state" else hits[0].strip()
            )
    return result


def structural_metrics(records):
    """Events are aligned by annotation IDs, not by language-model guesses.

    Each record: expected/predicted event lists, expected/predicted order dicts,
    expected/predicted relation dicts. A missing prediction is incorrect.
    """
    if not records:
        raise ValueError("No evaluation records")
    tp = fp = fn = events = state_ok = exact = sequences = 0
    order_ok = order_n = rel_ok = rel_n = 0
    for record in records:
        truth = {x["event_id"]: x for x in record["expected"]}
        guesses = record.get("predicted", [])
        pred = {x["event_id"]: x for x in guesses}
        if len(truth) != len(record["expected"]) or len(pred) != len(guesses):
            raise ValueError("Duplicate evaluation event IDs")
        all_exact = set(truth) == set(pred)
        for key in truth.keys() | pred.keys():
            target, guess = truth.get(key, {}), pred.get(key, {})
            a, b = slot_atoms(guess), slot_atoms(target)
            tp += len(a & b)
            fp += len(a - b)
            fn += len(b - a)
            if key in truth:
                events += 1
                state_ok += set(guess.get("post_state", [])) == set(target["post_state"])
                ok = a == b and bool(b)
                exact += ok
                all_exact &= ok
        sequences += all_exact
        for kind in ("order", "relations"):
            expected = record[f"expected_{kind}"]
            predicted = record.get(f"predicted_{kind}", {})
            correct = sum(
                key in predicted and predicted[key] == value for key, value in expected.items()
            )
            if kind == "order":
                order_n += len(expected)
                order_ok += correct
            else:
                rel_n += len(expected)
                rel_ok += correct
    if not events or not order_n or not rel_n:
        raise ValueError("FG-Score requires event, temporal-pair, and relation annotations")
    result = {
        "slot_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "state_accuracy": state_ok / events,
        "temporal_accuracy": order_ok / order_n,
        "relation_accuracy": rel_ok / rel_n,
        "event_exact_match": exact / events,
        "sequence_exact_match": sequences / len(records),
    }
    result["fg_score"] = np.mean(
        [result[k] for k in ("slot_f1", "state_accuracy", "temporal_accuracy", "relation_accuracy")]
    ).item()
    return result


def hallucination_metrics(records):
    if not records:
        raise ValueError("No distractor records")
    # Parse failures are incorrect in both tasks, rather than rewarded as silence.
    far = np.mean(
        [
            not r.get("parse_valid", True)
            or bool(set(r["predicted_actions"]) & set(r["unsupported_actions"]))
            for r in records
        ]
    ).item()
    fir = np.mean(
        [
            not r.get("parse_valid", True)
            or bool(set(r["predicted_intents"]) & set(r["unsupported_intents"]))
            for r in records
        ]
    ).item()
    return {"far": far, "fir": fir, "dr": 1 - (far + fir) / 2}


def wilson(successes, total, z=1.959963984540054):
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("Expected 0 <= successes <= total and total > 0")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return (center - radius, center + radius)


def control_metrics(trials):
    if not trials or any(t["required_stages"] <= 0 for t in trials):
        raise ValueError("Control metrics require annotated trials")
    successes = sum(t["success"] for t in trials)
    return {
        "task_success": successes / len(trials),
        "wilson_95": wilson(successes, len(trials)),
        "stage_completion": sum(t["ordered_stages"] for t in trials)
        / sum(t["required_stages"] for t in trials),
        "state_transition_success": np.mean([t["state_success"] for t in trials]).item(),
        "hallucination_error": np.mean([t["hallucination_error"] for t in trials]).item(),
    }


def paired_bootstrap(left, right, strata, statistic=np.mean, replicates=10000, seed=42):
    """Paired source-stratified clip resampling; statistic can recompute micro FG."""
    if not len(left) == len(right) == len(strata) or not len(left) or replicates < 1:
        raise ValueError("Bootstrap requires aligned nonempty clips and positive replicates")
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(np.asarray(strata) == key) for key in sorted(set(strata))]
    delta = []
    for _ in range(replicates):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        delta.append(statistic([left[i] for i in indices]) - statistic([right[i] for i in indices]))
    return {
        "difference": float(statistic(left) - statistic(right)),
        "ci_95": np.quantile(delta, [0.025, 0.975]).tolist(),
        "replicates": replicates,
    }
