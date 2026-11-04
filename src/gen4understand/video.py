"""Video I/O, endpoint extraction, and duplicate-boundary removal."""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def read_video(path):
    reader = imageio.get_reader(str(path))
    try:
        fps = float(reader.get_meta_data()["fps"])
        frames = np.stack(list(reader.iter_data()))
    finally:
        reader.close()
    if fps <= 0 or len(frames) < 2:
        raise ValueError("Video needs positive fps and at least two frames")
    return frames, fps


def write_video(path, frames, fps=16):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        str(path),
        np.asarray(frames, dtype=np.uint8),
        fps=fps,
        codec="libx264",
        macro_block_size=1,
        ffmpeg_log_level="error",
    )


def resample(frames, source_fps, target_fps=16):
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("Frame rates must be positive")
    count = max(2, round(len(frames) * target_fps / source_fps))
    return frames[np.linspace(0, len(frames) - 1, count).round().astype(int)]


def extract_anchors(event, root):
    frames, fps = read_video(event.source_video)
    if event.end_s > len(frames) / fps + 1 / fps:
        raise ValueError(f"Event {event.event_id} exceeds source duration")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    result = []
    for name, time in (("start", event.start_s), ("end", event.end_s)):
        index = min(round(time * fps), len(frames) - 1)
        path = root / f"{name}.png"
        Image.fromarray(frames[index]).save(path)
        result.append(str(path.resolve()))
    return tuple(result)


def concatenate(clips, tolerance=0.0):
    """Drop a shared boundary only when images match (or explicit MAE tolerance)."""
    if not clips or any(len(c) == 0 for c in clips):
        raise ValueError("Cannot merge empty clips")
    if len({c.shape[1:] for c in clips}) != 1:
        raise ValueError("Clip dimensions must match")
    if tolerance < 0:
        raise ValueError("Boundary tolerance must be nonnegative")
    result = [clips[0]]
    previous = clips[0][-1]
    for clip in clips[1:]:
        duplicate = np.abs(previous.astype(float) - clip[0].astype(float)).mean() <= tolerance
        result.append(clip[1:] if duplicate else clip)
        previous = clip[-1]
    return np.concatenate(result)


def evidence_probe(frames, mode, seed=0):
    """Appendix G.3 interventions preserve the input frame budget."""
    frames = np.asarray(frames)
    if mode == "ordered":
        return frames.copy()
    if mode == "endpoints":
        return frames[np.where(np.arange(len(frames)) < len(frames) / 2, 0, len(frames) - 1)]
    if mode == "shuffle-interior":
        out = frames.copy()
        out[1:-1] = np.random.default_rng(seed).permutation(out[1:-1])
        return out
    raise ValueError(f"Unknown probe: {mode}")
