# External model adapters

## Generation request / response

`g4u plan` extracts real anchors and writes `requests.jsonl`. Each request contains:

```json
{
  "candidate_id": "event-key-Wan2.7-0",
  "event_id": "event-001",
  "model": "Wan2.7",
  "seed": 42,
  "prompt": "Structured process description with absent-action constraints.",
  "start_frame": "/absolute/path/start.png",
  "end_frame": "/absolute/path/end.png",
  "duration_seconds": 5,
  "resolution": "720p",
  "conditioning": "first-and-last-frame"
}
```

The configured executable receives two final arguments: a request JSON path and a response JSON path. It must call the selected generator using both frames, preserve the requested seed, wait for completion, download the output, and write:

```json
{
  "candidate_id": "event-key-Wan2.7-0",
  "event_id": "event-001",
  "model": "Wan2.7",
  "seed": 42,
  "conditioning": "first-and-last-frame",
  "video": "/absolute/path/generated.mp4"
}
```

Use environment variables or your provider's credential store. Do not put secrets into requests or response artifacts. Commands run as an argument vector with `shell=False`; adapter failures propagate. A stale response is removed before each invocation. Output identity, duration, resolution, and decode validity are checked. Model provenance cannot be proven from video pixels; preserve provider job metadata separately.

No unverified vendor URL or guessed request schema is embedded in the repository. If provider model versions differ, record the actual revision and report a modified protocol rather than labeling it the paper configuration.

## Frozen verifier observation contract

Run your feature exporters offline, using the following pretrained components:

| Signal | Frozen source | Required measurement |
|---|---|---|
| Semantics | Qwen2.5-VL-72B, greedy decode | Canonical actor/action/object/post-state/temporal fields |
| Anchors | DINOv2 ViT-L/14 | Generated/reference start and end features |
| Object masks | GroundingDINO Swin-T OGC + SAM2 Hiera-Large | Frame-aligned masks and target presence on interior frames |
| Physical relations | Exported masks | `window_relation` / `release_holds` results for every applicable scripted relation |
| Temporal warp | RAFT-Large | Mean RGB warp error, RGB normalized to [0,1], consistent forward/backward warp convention |
| Trajectory | Target mask centroids | Framewise `(x / width, y / height)` coordinates |
| Quality | Official DOVER.pth | Raw technical visual quality score |

Resample candidates to 16 fps before verification. Count target absence over **interior** frames only; do not include the fixed boundary anchors to inflate coverage. Do not interpolate absent tracks into observed detections. Do not drop failed relations from the denominator. If masks/features are unavailable, reject or repair the observation instead of assigning a favorable score.

Each JSONL row passed to `g4u score` has `candidate_id`, `event_id`, `video`, `track_presence`, optional human `audit_valid`, and `observation`:

```text
observation.parsed_event        canonical five-slot dictionary
observation.feature_pairs      [[generated_start, real_start], [generated_end, real_end]]
observation.mask_pairs         [[generated_start_masks, real_start_masks], [generated_end_masks, real_end_masks]]
observation.physical_checks    list of booleans from every applicable scripted check
observation.warp_error         scalar mean RGB photometric warp error
observation.centroids          T x 2 normalized target centroids
observation.dover_score        raw scalar DOVER output
observation.anchor_feature_weight  optional, default 0.5
```

For large masks, call `score_observation` directly from a Python exporter with NumPy arrays rather than serializing them to JSON. Mask arrays must be brought to the same frame geometry before IoU matching. Keep original model revision and exporter metadata in your experiment log. `g4u score` imports actual observations; it does not itself load the six frozen model stacks.

The local `g4u predict` command can produce parser responses using the 72B model and strict slot prompts, but automatic segmentation/tracking/flow/quality exporter runners remain external integrations.

## Integration references

- [Qwen2.5-VL Transformers documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl)
- [PEFT LoRA configuration](https://huggingface.co/docs/peft/en/package_reference/lora)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [SAM2](https://github.com/facebookresearch/sam2)
- [RAFT](https://github.com/princeton-vl/RAFT)
- [DOVER](https://github.com/VQAssessment/DOVER)

Obtain their checkpoints and follow upstream licenses/install instructions. The training loader explicitly targets language modules and saves projectors through PEFT's `modules_to_save` mechanism.
