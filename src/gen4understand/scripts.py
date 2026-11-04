"""Deterministic supervision from conditioning scripts, never parser captions."""

from .schema import Event

# Contextual template bank: application extensions should be audited per scene.
NEGATIVE_RULES = (
    (frozenset({"brush", "cup", "water"}), ("wipe", "clean", "brush")),
    (frozenset({"towel", "table"}), ("wipe", "clean")),
)


def hard_negatives(events: list[Event]) -> tuple[str, ...]:
    observed = {e.action for e in events}
    context = {obj for e in events for obj in e.scene_objects}
    proposed = {a for e in events for a in e.unsupported_actions}
    for required, actions in NEGATIVE_RULES:
        if required <= context:
            proposed.update(actions)
    return tuple(sorted(proposed - observed))


def render(events: list[Event], include_negatives=True) -> str:
    if not events:
        raise ValueError("Cannot render an empty script")
    parts = [
        f"{e.actor} performs {e.action} on {e.object}, starting in "
        f"{', '.join(e.pre_state)}, moving {e.motion}, resulting in "
        f"{', '.join(e.post_state)}. Temporal relation: {e.temporal_relation}."
        for e in events
    ]
    negatives = hard_negatives(events) if include_negatives else ()
    if negatives:
        parts.append(f"No {', '.join(negatives)} action occurs in this segment.")
    return " ".join(parts)


def qa_examples(events, video, level, sample_id, mode="all"):
    """All templates by default; 'paper-budget' deterministically subsamples ~2 / ~1.5."""
    import hashlib

    if mode not in {"all", "paper-budget"}:
        raise ValueError("Unknown QA mode")
    examples = [("caption", "Describe the observed physical process.", render(events))]
    for e in events:
        examples.extend(
            [
                (
                    "action",
                    f"What action does {e.actor} perform on {e.object}?",
                    f"{e.action}; motion: {e.motion}.",
                ),
                (
                    "state",
                    f"What changes for {e.object} during event {e.event_id}?",
                    f"From {', '.join(e.pre_state)} to {', '.join(e.post_state)}.",
                ),
            ]
        )
    for a, b in zip(events, events[1:], strict=False):
        relation = "before" if a.end_s <= b.start_s else "while"
        examples.append(
            ("temporal", f"When does {a.event_id} occur relative to {b.event_id}?", relation)
        )
    for action in hard_negatives(events):
        examples.append(
            ("negative", f"Does a {action} action occur?", "No. " + render(events, False))
        )
    if mode == "paper-budget":
        key = int(hashlib.sha256(sample_id.encode()).hexdigest(), 16)
        if level == "atom":
            examples = [x for x in examples if x[0] in {"action", "state"}][:2]
        else:
            pool = [x for x in examples if x[0] in {"caption", "temporal", "negative"}]
            count = min(1 + key % 2, len(pool))
            offset = (key // 2) % len(pool)
            examples = [pool[(offset + i) % len(pool)] for i in range(count)]
    return [
        {
            "id": f"{sample_id}:{i}",
            "video": video,
            "level": level,
            "category": kind,
            "event_ids": [e.event_id for e in events],
            "recording_id": events[0].recording_id,
            "split": events[0].split,
            "question": q,
            "answer": a,
        }
        for i, (kind, q, a) in enumerate(examples)
    ]
