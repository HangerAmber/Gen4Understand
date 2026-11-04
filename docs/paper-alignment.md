# Paper-to-code alignment

Source: *Generation-for-Understanding with Structured Action Scripts for Embodied Multimodal Learning*, NeurIPS 2026. This document distinguishes specified behavior from engineering choices; it does not claim an original-code release or recovered unpublished artifacts.

| Manuscript | Code | Details |
|---|---|---|
| §3.1, Eq. 1, App. C | `schema`, `scripts` | Actor, action, object, pre-state, motion, post-state, temporal relation; contextual negative bank |
| §3.2, Eq. 2, App. E.2 | `generation` | Two candidates each from Wan2.7/Seedance 2.0; independent seeds; start/end anchors; 5 s at 720p |
| §3.3, Eq. 3, App. D.1 | `scoring.slot_f1` | Frozen parser event slots compared with conditioning event; parser prose never used as student targets |
| App. D.2 | `scoring.anchor_score` | DINOv2 feature cosine and Hungarian-matched mask IoU; both boundary frames |
| App. D.2 | `Calibrator.select` | Reject candidates below 80% target-track presence before ranking |
| App. D.3 | `relation_holds`, `window_relation`, `release_holds` | Eight-frame pre/post windows; six-of-eight voting; exact reported geometric thresholds |
| App. D.4 | `temporal_score` | `0.7 exp(-Ewarp/0.05) + 0.3 exp(-mean acceleration/0.015)` |
| App. D.4 | `score_observation` | Official DOVER checkpoint score is imported without pretending a proxy is DOVER |
| App. D.5 | `calibrate` | Training-only min/max; clipping; weight grid 0.10–0.40 by 0.05; thresholds 0.60–0.80 by 0.05; >=35% event coverage |
| §3.3, Eq. 4 | `Calibrator.select` | Highest eligible candidate, at most one/event, threshold 0.70 with published weights |
| §3.4 | `pipeline`, `video` | All children required; chronological concatenation; duplicate boundaries removed; same accepted scripts generate language |
| Eq. 5, Table 5 | `training` | Response-only NLL; level weights 1/.5/.3/.2; frozen vision; LoRA 64/128; trained projector |
| App. B | `metrics` | Micro slot F1; state/order/relation accuracy; FG average; exact match; clip FAR/FIR; control statistics |
| App. E.1 | `splits` | Source/scene/object isolation, event tuple deduplication, endpoint similarity screening helper |
| App. F.6 | `paired_bootstrap` | 10,000 paired source-stratified clip replicates by default |
| App. G.3 | `evidence_probe` | Ordered, endpoint-only, and shuffled-interior visual probes |

## Published defaults

Score order is semantic, anchor, physical, temporal, quality. Weights are `(0.35, 0.20, 0.20, 0.15, 0.10)` and threshold `0.70`. Calibration uses 1,000 candidates / 250 training-only events. Retention rates such as 38% are reported outcomes, not a quota imposed by the implementation.

Relation geometry uses frame-diagonal normalization for contact/support/release distances, and width/height normalization for directional margins. Contact <=0.02; support vertical gap <=0.03 with horizontal overlap >=0.30 of the smaller width; containment requires centroid inclusion and >=0.70 contained area; release requires contact before, no contact after, and >=0.03 increased separation; directional margin >=0.05. These are image-plane checks.

## Explicit choices where the paper is incomplete

1. **Anchor fusion:** the paper names feature similarity and mask IoU without giving their internal blend. The default blend is equal weight, configurable through `anchor_feature_weight`. It is not a recovered paper hyperparameter.
2. **Canonicalization:** symbolic fields are normalized by whitespace/case for slot F1. No learned synonym mapper is introduced. Evaluation uses annotation event IDs; full detection/alignment from unsegmented videos is external.
3. **Calibration ties:** prefer higher event coverage, then proximity to the published weights. Constant calibration components cause an error. Production excludes calibration events for conservative isolation.
4. **Hierarchy:** exact duplicate boundary frames are removed by default. An explicit MAE tolerance is available at the Python API. Concurrent child events are rejected by the concatenation builder because serial concatenation cannot preserve overlap; they require a shared-timeline compositor. Atomic scripts can encode `while`.
5. **QA budget:** `all` emits every supported deterministic template. `paper-budget` emits two atomic examples and deterministically one or two hierarchical examples, approximating the disclosed 56K count. The original template bank and sampling seed are unavailable.
6. **Weighted training:** examples are sampled uniformly and level weights are applied once in the loss. This avoids double weighting. The manuscript also describes mixed sampling by level; its exact sampler is not specified. Token NLL is averaged per response before level weighting.
7. **Pixel budget:** a configurable processor pixel cap complements the paper's 1280-pixel maximum side to accommodate 16 frames in a 4096-token sequence. This is a memory-management choice, not a published setting. Overflow is rejected.
8. **Endpoint-only intervention:** repeat the two boundary frames to retain the input frame count. The manuscript does not state the repetition pattern.

## Artifacts needed for a full experimental reproduction

- Original structured splits and human-adjudicated event/state annotations for Ego-Exo4D, EPIC-KITCHENS-100, DROID, and the 600-video distractor split.
- Generation provider access, exact model revisions, original seed manifests, and generated candidate videos.
- Frozen Qwen2.5-VL-72B parser weights/prompts; DINOv2 ViT-L/14; GroundingDINO Swin-T OGC; SAM2 Hiera-Large; RAFT-Large; official DOVER checkpoint.
- Calibration extrema and the audited calibration labels (refit locally if unavailable).
- Human audit labels for fidelity/precision analyses; these cannot be inferred from verifier scores.
- Student model checkpoints, target GPU resources, and native benchmark environments.
- Fixed VLA policy/controller interfaces and synchronized real-robot trial logs. This code consumes evaluated trial outcomes; it does not implement a substitute controller.

## Scope across experiments

Matched-real controls can use the same schema/renderer and training configuration with real video paths. Distribution matching and equal training-instance counts must be enforced on the actual datasets. Calibration ablations can be expressed by alternate weights; start-frame-only generation is intentionally outside the main two-anchor adapter contract. Human audits, trajectory-to-script extraction, conflicting-script prompts, native LIBERO/RoboTwin rollouts, and robot execution depend on unavailable source artifacts/integrations. Their numerical results are not claimed by the demo.

The code is reconstructed from the manuscript. It provides operational core algorithms and integration boundaries; a paper alone cannot supply the missing data, model-service behavior, or experimental runs.
