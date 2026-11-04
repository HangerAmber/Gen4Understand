# Data contracts

All manifests are UTF-8 JSONL: one object per line. Paths are absolute or relative to the process working directory. Avoid moving a built dataset without rewriting its absolute video paths. The input paper and personal information should not be added to manifests.

## Events

See `examples/event.json` for a complete record. `pre_state` and `post_state` are lists of canonical symbolic predicates. The seven semantic fields correspond to Eq. (1). Metadata ties every event back to its original recording, scene, object instance, partition, and source interval. Times are finite seconds with `0 <= start_s < end_s`.

Use `before:event-id`, `after:event-id`, or `while:event-id` for temporal relations. Use a fixed vocabulary throughout generation, parsing, and evaluation. Negative labels must be unsupported in the actual segment; do not assume every unmentioned vocabulary action is a valid hard negative.

Split source recordings **before** extracting scripts or endpoints. `validate_splits` checks cross-partition recording, scene, object-instance, and event-tuple overlap. Run it on the combined train/validation/test manifest. Endpoint feature similarity screening is an additional Python helper requiring external DINOv2 features. It returns potential collisions for adjudication; it cannot guarantee absence of visual leakage.

## Hierarchy units

```json
{"unit_id":"return-brush-action","level":"act","event_ids":["event-0","event-1"]}
```

`level` is `act`, `sub`, or `task`. Each level explicitly lists all constituent **atomic** event IDs, even for a subtask/task. Units may overlap across levels but cannot mix recordings or partitions. Children are ordered by their annotated start time. Missing acceptance skips the entire unit. Concurrent children cannot be serialized by the current merge operator and raise an explicit error.

## Candidate scores

```json
{"candidate_id":"event-0-wan-0","event_id":"event-0","video":"data/candidate.mp4","scores":{"sem":0.9,"anc":0.8,"phys":1.0,"tmp":0.85,"qual":0.42},"track_presence":0.95,"split":"train","audit_valid":true}
```

Scores are raw measurements, not already normalized values. `audit_valid` is a human fidelity label used only for calibration search. Do not infer it from scores. Production selection does not require it. `track_presence` is the fraction of interior frames with an observed target track.

The calibrator artifact stores frozen minima/maxima, component weights, acceptance threshold, and calibration event IDs. Never refit on test data or normalize each candidate independently. The selected manifest adds `score` and contains at most one candidate per event.

## Training / inference records

```json
{"id":"event-0:0","video":"data/clip.mp4","level":"atom","category":"state","event_ids":["event-0"],"recording_id":"source-0","split":"train","question":"Where is the brush after the action?","answer":"Inside the basket."}
```

The student sees the video and question, and learns only the answer tokens. Script provenance fields remain metadata. Answers come from original scripts, not verifier-generated prose. Inference needs `id`, `video`, and `question`; an answer is not required.

For data selection experiments, preserve the same template policy, example count, event-category distribution, and adaptation configuration across conditions. Real-video controls can reference real clips using the same training record format.
