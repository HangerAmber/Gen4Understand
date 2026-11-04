# Evaluation protocol

Keep evaluation videos real and source-disjoint from all generation scripts, anchors, candidates, and calibration examples. Student outputs use greedy decoding. A malformed or missing prediction is incorrect, never silently removed from the denominator. Values emitted by the library are fractions in `[0,1]` except differences and confidence intervals of differences.

## Structural records

```json
{
  "clip_id": "held-out-001",
  "source": "DROID",
  "expected": [
    {"event_id":"e0","actor":"right gripper","action":"place","object":"brush","post_state":["inside(basket)"],"temporal_relation":"before:e1"}
  ],
  "predicted": [
    {"event_id":"e0","actor":"right gripper","action":"place","object":"brush","post_state":["inside(basket)"],"temporal_relation":"before:e1"}
  ],
  "expected_order": {"e0:e1": true},
  "predicted_order": {"e0:e1": true},
  "expected_relations": {"e0:post:inside-basket": true},
  "predicted_relations": {"e0:post:inside-basket": true}
}
```

`Slot-F1` micro-aggregates `(slot, value)` atoms across event IDs. Additional predicted events add false positives. State accuracy checks complete result-state sets. Temporal accuracy compares annotated event-pair order. Relation accuracy checks labeled source/result contact, support, and containment relations. FG is the arithmetic mean of those four components. Event/sequence exact match are reported separately. Input annotation IDs provide alignment; the implementation does not invent an unpublished alignment algorithm for free-form event sequences.

`parse_slots` supports explicit lines such as `actor: right gripper`, `action: place`, and `post_state: inside(basket); released(brush)`. Missing or duplicate fields remain absent. A model's raw response must be aligned with its question/event ID before building the evaluation manifest. The exact original regex bank and intent taxonomy were not supplied.

## Distractor records

```json
{"predicted_actions":["place"],"unsupported_actions":["wipe","clean"],"predicted_intents":[],"unsupported_intents":["prepare to clean the cup"],"parse_valid":true}
```

FAR/FIR are fractions of clips with at least one unsupported action/intent. `DR = 1 - (FAR + FIR)/2`. Explicit parse failures receive incorrect outcomes in both tasks. This conservative extension prevents empty malformed predictions from receiving perfect robustness. Use a fixed audited vocabulary and report parse-failure rates with results.

## Control records

```json
{"success":true,"ordered_stages":3,"required_stages":3,"state_success":true,"hallucination_error":false}
```

Metrics summarize task success, ordered stage completion, intended state transition success, and hallucination-induced downstream errors. Wilson 95% intervals are reported for task success. Keep policy, controller, task instructions, and rollout budget fixed while swapping only the perception-language module.

The paper uses 20 rollouts/task on the reported LIBERO suites and RoboTwin 2.0 tasks; real-robot evaluation uses three configurations × 40 trials per task family per model. These runners and hardware interfaces are external. The metrics command accepts actual evaluated trial logs rather than generating simulated success numbers.

## Paired bootstrap

Recompute the micro-aggregated FG statistic inside each replicate:

```python
from gen4understand.metrics import paired_bootstrap, structural_metrics

assert [r['clip_id'] for r in method_a] == [r['clip_id'] for r in method_b]
result = paired_bootstrap(
    method_a, method_b,
    strata=[r['source'] for r in method_a],
    statistic=lambda clips: structural_metrics(clips)['fg_score'],
    replicates=10_000, seed=42,
)
```

Sampling is paired at the clip level and stratified by source split. Do not subtract separately computed marginal interval endpoints to estimate paired uncertainty.

## Visual-process interventions

`video.evidence_probe` supports ordered frames, repeated endpoints, and shuffled interior frames. The shuffle retains both boundary frames and the frame count. Use exactly the same questions, model, decoding settings, and evaluation targets across conditions. A conflicting-script experiment requires independently prepared incorrect scripts; do not generate its labels from model outputs.
