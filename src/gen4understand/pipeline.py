"""Construct atom/action/subtask/task supervision from selected candidates."""

import hashlib
from pathlib import Path

from .io import write_jsonl
from .scripts import qa_examples
from .splits import validate_splits
from .video import concatenate, read_video, resample, write_video


def build_dataset(events, units, selected, output, qa_mode="all", boundary_tolerance=0):
    validate_splits(events)
    by_id = {e.event_id: e for e in events}
    if len({u.unit_id for u in units}) != len(units):
        raise ValueError("Duplicate hierarchy unit IDs")
    if any(u.unit_id in by_id for u in units):
        raise ValueError("Event and hierarchy IDs must be distinct")
    for key in selected:
        if key not in by_id or by_id[key].split != "train":
            raise ValueError("Selected candidate references an unknown or non-training event")
        if selected[key][0].event_id != key or selected[key][0].split != "train":
            raise ValueError("Selected candidate provenance does not match the training event")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records, skipped = [], []
    for key, (candidate, _) in sorted(selected.items()):
        if not Path(candidate.video).is_file():
            raise ValueError(f"Missing accepted video: {candidate.video}")
        records.extend(
            qa_examples([by_id[key]], str(Path(candidate.video).resolve()), "atom", key, qa_mode)
        )
    for unit in units:
        if any(key not in by_id for key in unit.event_ids):
            raise ValueError(f"Unknown hierarchy child in {unit.unit_id}")
        children = [by_id[key] for key in unit.event_ids]
        if len({(e.recording_id, e.split) for e in children}) != 1:
            raise ValueError("Hierarchy cannot cross recordings or partitions")
        if any(key not in selected for key in unit.event_ids):
            skipped.append(
                {"unit_id": unit.unit_id, "reason": "incomplete accepted-child coverage"}
            )
            continue
        children.sort(key=lambda e: (e.start_s, e.event_id))
        if any(a.end_s > b.start_s for a, b in zip(children, children[1:], strict=False)):
            raise ValueError(
                "Concurrent events require a shared timeline; sequential concatenation is invalid"
            )
        clips = []
        for event in children:
            frames, fps = read_video(selected[event.event_id][0].video)
            clips.append(resample(frames, fps))
        name = hashlib.sha256(unit.unit_id.encode()).hexdigest()[:16]
        path = output / "videos" / f"{name}.mp4"
        write_video(path, concatenate(clips, boundary_tolerance))
        records.extend(qa_examples(children, str(path), unit.level, unit.unit_id, qa_mode))
    write_jsonl(output / "train.jsonl", records)
    write_jsonl(output / "skipped_units.jsonl", skipped)
    return records, skipped
