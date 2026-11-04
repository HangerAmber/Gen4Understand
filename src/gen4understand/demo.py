"""Small deterministic synthetic fixture, explicitly not learned video generation."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .io import write_json, write_jsonl
from .pipeline import build_dataset
from .schema import Event, Unit
from .verifier import Calibrator, Candidate
from .video import write_video


def run_demo(output):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    clips, events = [], []
    for index, action in enumerate(("move", "release")):
        frames = []
        for t in np.linspace(0, 1, 16):
            im = Image.new("RGB", (256, 144), "#142537")
            draw = ImageDraw.Draw(im)
            draw.rectangle((165, 75, 235, 130), outline="#74d4c0", width=3)
            x = round(40 + 155 * t) if index == 0 else 195
            y = 55 if index == 0 else round(55 + 45 * t)
            draw.rectangle((x - 14, y - 5, x + 14, y + 5), fill="#eac06e")
            draw.text((8, 8), "SYNTHETIC PIPELINE FIXTURE", fill="white")
            frames.append(np.asarray(im))
        clips.append(np.stack(frames))
        path = root / f"fixture-{index}.mp4"
        write_video(path, clips[-1])
        events.append(
            Event(
                f"event-{index}",
                "fixture-recording",
                "train",
                "right gripper",
                action,
                "brush",
                ("held-by(right gripper)",),
                "toward basket" if index == 0 else "downward into basket",
                ("above(basket)",) if index == 0 else ("inside(basket)",),
                "before:event-1" if index == 0 else "after:event-0",
                float(index),
                float(index + 1),
                str(root / "source.mp4"),
                "fixture-scene",
                "fixture-brush",
                ("brush", "cup", "water", "basket"),
            )
        )
    write_video(root / "source.mp4", np.concatenate(clips))
    candidates = []
    for i, event in enumerate(events):
        for j in range(4):
            scores = (0.92, 0.91, 0.90, 0.94, 0.89) if j == 0 else (0.40, 0.45, 0.50, 0.40, 0.60)
            candidates.append(
                Candidate(
                    f"{event.event_id}-candidate-{j}",
                    event.event_id,
                    str(root / f"fixture-{i}.mp4"),
                    scores,
                    1.0,
                )
            )
    calibration = Calibrator((0.0,) * 5, (1.0,) * 5)
    selected = calibration.select(candidates)
    units = [
        Unit(f"fixture-{level}", level, tuple(e.event_id for e in events))
        for level in ("act", "sub", "task")
    ]
    records, skipped = build_dataset(events, units, selected, root / "dataset")
    write_jsonl(root / "events.jsonl", [e.to_dict() for e in events])
    write_jsonl(root / "candidates.jsonl", [c.to_dict() for c in candidates])
    write_json(root / "calibrator.json", calibration.to_dict())
    write_jsonl(
        root / "units.jsonl",
        [{"unit_id": u.unit_id, "level": u.level, "event_ids": u.event_ids} for u in units],
    )
    summary = {
        "fixture_only": True,
        "learned_generation": False,
        "events": len(events),
        "candidates": len(candidates),
        "accepted_events": len(selected),
        "qa_examples": len(records),
        "skipped_units": len(skipped),
    }
    write_json(root / "summary.json", summary)
    return summary
