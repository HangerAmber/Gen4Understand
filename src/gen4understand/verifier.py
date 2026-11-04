"""Paper-defined score aggregation and calibration, independent of model runtimes."""

import itertools
import math
from dataclasses import asdict, dataclass

import numpy as np

COMPONENTS = ("sem", "anc", "phys", "tmp", "qual")
PAPER_WEIGHTS = (0.35, 0.20, 0.20, 0.15, 0.10)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    event_id: str
    video: str
    scores: tuple[float, ...]
    track_presence: float
    split: str = "train"
    audit_valid: bool | None = None

    def __post_init__(self):
        if len(self.scores) != 5 or not all(math.isfinite(v) for v in self.scores):
            raise ValueError("Expected five finite scores in sem/anc/phys/tmp/qual order")
        if not 0 <= self.track_presence <= 1:
            raise ValueError("Invalid interior-frame track presence")
        if self.audit_valid is not None and type(self.audit_valid) is not bool:
            raise ValueError("audit_valid must be a human-audited boolean or null")

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if isinstance(value["scores"], dict):
            value["scores"] = tuple(value["scores"][k] for k in COMPONENTS)
        else:
            value["scores"] = tuple(value["scores"])
        return cls(**value)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Calibrator:
    minimum: tuple[float, ...]
    maximum: tuple[float, ...]
    weights: tuple[float, ...] = PAPER_WEIGHTS
    threshold: float = 0.70
    calibration_event_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if any(len(x) != 5 for x in (self.minimum, self.maximum, self.weights)):
            raise ValueError("Calibrator requires five components")
        if not all(math.isfinite(x) for v in (self.minimum, self.maximum, self.weights) for x in v):
            raise ValueError("Nonfinite calibration parameter")
        if any(hi <= lo for lo, hi in zip(self.minimum, self.maximum, strict=False)):
            raise ValueError("Each calibration component must have nonzero range")
        if min(self.weights) < 0 or not math.isclose(sum(self.weights), 1):
            raise ValueError("Weights must be nonnegative and sum to one")
        if not 0 <= self.threshold <= 1:
            raise ValueError("Invalid threshold")

    def normalize(self, scores):
        return np.clip(
            (np.asarray(scores) - self.minimum) / (np.asarray(self.maximum) - self.minimum), 0, 1
        )

    def score(self, candidate):
        return float(self.normalize(candidate.scores) @ self.weights)

    def select(self, candidates, allow_calibration=False):
        selected, seen = {}, set()
        for candidate in sorted(candidates, key=lambda c: c.candidate_id):
            if candidate.candidate_id in seen:
                raise ValueError("Duplicate candidate ID")
            seen.add(candidate.candidate_id)
            if candidate.split != "train":
                raise ValueError("Only training candidates may enter selection")
            if not allow_calibration and candidate.event_id in self.calibration_event_ids:
                raise ValueError("Calibration event reused in production selection")
            if candidate.track_presence < 0.80:
                continue
            score = self.score(candidate)
            if score >= self.threshold - 1e-12 and (
                candidate.event_id not in selected or score > selected[candidate.event_id][1]
            ):
                selected[candidate.event_id] = (candidate, score)
        return selected

    def to_dict(self):
        return asdict(self)


def calibrate(candidates, search=True, minimum_coverage=0.35):
    """Fit training-only extrema; select precision-maximizing weights on audited candidates.

    Ties prefer higher coverage, then weights closest to the reported configuration.
    The paper does not disclose its tie-breaking rule.
    """
    if not candidates or any(c.split != "train" for c in candidates):
        raise ValueError("Calibration needs a nonempty training-only pool")
    if len({c.candidate_id for c in candidates}) != len(candidates):
        raise ValueError("Duplicate calibration candidate IDs")
    candidates = sorted(candidates, key=lambda c: c.candidate_id)
    values = np.array([c.scores for c in candidates])
    lo, hi = tuple(values.min(0)), tuple(values.max(0))
    event_ids = tuple(sorted({c.event_id for c in candidates}))
    base = Calibrator(lo, hi, calibration_event_ids=event_ids)
    if not search:
        return base
    if not 0 < minimum_coverage <= 1 or any(c.audit_valid is None for c in candidates):
        raise ValueError("Search requires audit labels for every candidate and valid coverage")
    normalized = base.normalize(values)
    groups = [
        [i for i, c in enumerate(candidates) if c.event_id == key and c.track_presence >= 0.8]
        for key in event_ids
    ]
    best = None
    # Integer grid avoids floating-point sum errors: 2..8 units of 0.05.
    for units in itertools.product(range(2, 9), repeat=5):
        if sum(units) != 20:
            continue
        weights = tuple(x / 20 for x in units)
        scores = normalized @ weights
        winners = [max(group, key=lambda i: (scores[i], -i)) for group in groups if group]
        for threshold in (0.60, 0.65, 0.70, 0.75, 0.80):
            accepted = [i for i in winners if scores[i] >= threshold - 1e-12]
            coverage = len(accepted) / len(event_ids)
            if not accepted or coverage < minimum_coverage:
                continue
            precision = sum(candidates[i].audit_valid for i in accepted) / len(accepted)
            key = (
                precision,
                coverage,
                -sum(abs(a - b) for a, b in zip(weights, PAPER_WEIGHTS, strict=False)),
            )
            if best is None or key > best[0]:
                best = (key, Calibrator(lo, hi, weights, threshold, event_ids))
    if best is None:
        raise ValueError("No calibration satisfies minimum event coverage")
    return best[1]
