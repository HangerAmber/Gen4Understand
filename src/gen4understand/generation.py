"""Auditable requests for real first/last-frame generators; no invented vendor APIs."""

import hashlib
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .io import read_json, write_json
from .scripts import render
from .video import extract_anchors, read_video


@dataclass(frozen=True)
class GenerationRequest:
    candidate_id: str
    event_id: str
    model: str
    seed: int
    prompt: str
    start_frame: str
    end_frame: str
    duration_seconds: int = 5
    resolution: str = "720p"
    conditioning: str = "first-and-last-frame"


def plan_generation(events, root, base_seed=42):
    requests = []
    for event in events:
        if event.split != "train":
            raise ValueError("Evaluation events must not supply generation anchors")
        event_key = hashlib.sha256(event.event_id.encode()).hexdigest()[:16]
        start, end = extract_anchors(event, Path(root) / event_key)
        for model in ("Wan2.7", "Seedance-2.0"):
            for index in range(2):
                candidate_id = f"{event_key}-{model}-{index}"
                seed = int.from_bytes(
                    hashlib.sha256(f"{base_seed}:{candidate_id}".encode()).digest()[:4], "big"
                )
                requests.append(
                    asdict(
                        GenerationRequest(
                            candidate_id, event.event_id, model, seed, render([event]), start, end
                        )
                    )
                )
    return requests


def execute_request(request, command, root, timeout=1800):
    """Local adapter receives request.json response.json, with shell execution disabled.

    Adapter handles vendor auth/polling and downloads to a local video path.
    It must echo request identity and conditioning; the artifact is decoded here.
    """
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    input_path, output_path = root / "request.json", root / "response.json"
    write_json(input_path, request)
    if output_path.exists():
        output_path.unlink()
    subprocess.run(
        [*command, str(input_path), str(output_path)], check=True, timeout=timeout, shell=False
    )
    response = read_json(output_path)
    for key in ("candidate_id", "event_id", "model", "seed", "conditioning"):
        if response.get(key) != request[key]:
            raise ValueError(f"Adapter response mismatches {key}")
    video = Path(response["video"])
    if not video.is_absolute():
        video = root / video
    frames, fps = read_video(video)
    if abs(len(frames) / fps - request["duration_seconds"]) > max(1 / fps, 0.1):
        raise ValueError("Generated duration does not match request")
    if min(frames.shape[1:3]) != 720:
        raise ValueError("Expected 720p output")
    response["video"] = str(video.resolve())
    return response
