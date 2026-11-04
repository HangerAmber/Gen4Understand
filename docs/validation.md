# Release validation

Local checks performed on 2026-11-04 with Python 3.10, NumPy 1.26.4, and PyTorch 2.5.1:

- CPU regression suite: 14 tests passed, including the response-only weighted-loss backward pass.
- A randomly initialized tiny Qwen2.5-VL model verified LLM LoRA gradients, frozen visual-backbone parameters, trainable projector parameters, and projector inclusion in the saved adapter. No pretrained weights were downloaded for this check.
- End-to-end synthetic fixture: 2 atomic events, 8 candidate records, 2 selected clips, 3 hierarchy units, 39 QA records, no incomplete units.
- Source schema and partition validation passed for the fixture.
- Four deterministic, independently seeded generation requests per event were constructed with real extracted fixture endpoints.
- Ruff lint and formatting checks passed.
- Source distribution and wheel built successfully.
- Project source was checked for manuscript author/contact identifiers and token patterns; none were included.

These checks validate software behavior on numerical fixtures and synthetic clips. They do not validate pretrained video quality, human-audited fidelity, full VLM adaptation, the reported benchmark scores, or robot success rates. The original data, model-service access, full checkpoints, and target hardware are needed for those experiments. See `paper-alignment.md` for the complete artifact boundary.
