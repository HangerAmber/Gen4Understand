"""Command-line entry points for each independently resumable pipeline stage."""

import argparse
import json
from pathlib import Path

from .io import read_json, read_jsonl, write_json, write_jsonl
from .schema import Event, Unit
from .verifier import Calibrator, Candidate, calibrate


def _events(path):
    return [Event.from_dict(row) for row in read_jsonl(path)]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Gen4Understand data, training, and evaluation pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run a CPU synthetic pipeline fixture")
    demo.add_argument("--output", default="runs/demo")
    validate = sub.add_parser("validate", help="Validate schema and partition isolation")
    validate.add_argument("events")
    plan = sub.add_parser("plan", help="Extract real anchors and plan four candidates per event")
    plan.add_argument("events")
    plan.add_argument("--output", required=True)
    plan.add_argument("--seed", type=int, default=42)
    execute = sub.add_parser(
        "generate", help="Execute an explicitly configured local generator adapter"
    )
    execute.add_argument("requests")
    execute.add_argument("--output", required=True)
    execute.add_argument("--adapter", nargs=argparse.REMAINDER, required=True)
    score = sub.add_parser("score", help="Compute raw scores from frozen model observations")
    score.add_argument("observations")
    score.add_argument("events")
    score.add_argument("--output", required=True)
    fit = sub.add_parser("calibrate", help="Fit frozen extrema and optionally audited grid search")
    fit.add_argument("candidates")
    fit.add_argument("--output", required=True)
    fit.add_argument(
        "--paper-weights",
        action="store_true",
        help="Keep paper weights/threshold; still fit extrema",
    )
    fit.add_argument("--strict-paper-size", action="store_true")
    select = sub.add_parser("select")
    select.add_argument("candidates")
    select.add_argument("calibrator")
    select.add_argument("--output", required=True)
    build = sub.add_parser("build")
    build.add_argument("events")
    build.add_argument("units")
    build.add_argument("selected")
    build.add_argument("--output", required=True)
    build.add_argument("--qa-mode", choices=("all", "paper-budget"), default="all")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("kind", choices=("structural", "hallucination", "control"))
    evaluate.add_argument("records")
    evaluate.add_argument("--output", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--data", required=True)
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--resume")
    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--data", required=True)
    predict_parser.add_argument("--output", required=True)
    predict_parser.add_argument("--adapter")
    args = parser.parse_args(argv)
    if args.command == "demo":
        from .demo import run_demo

        print(json.dumps(run_demo(args.output), indent=2))
    elif args.command == "validate":
        from .splits import validate_splits

        events = _events(args.events)
        validate_splits(events)
        print(f"Validated {len(events)} events")
    elif args.command == "plan":
        from .generation import plan_generation
        from .splits import validate_splits

        events = _events(args.events)
        validate_splits(events)
        requests = plan_generation(events, Path(args.output) / "anchors", args.seed)
        write_jsonl(Path(args.output) / "requests.jsonl", requests)
    elif args.command == "generate":
        from .generation import execute_request

        if not args.adapter:
            parser.error("--adapter requires an executable and optional arguments")
        results = []
        for i, request in enumerate(read_jsonl(args.requests)):
            results.append(execute_request(request, args.adapter, Path(args.output) / str(i)))
            write_jsonl(Path(args.output) / "generated.jsonl", results)
    elif args.command == "score":
        from .scoring import score_observation

        events = {e.event_id: e for e in _events(args.events)}
        result = []
        for row in read_jsonl(args.observations):
            event = events[row["event_id"]]
            scores = score_observation(row["observation"], event.to_dict())
            result.append(
                {
                    "candidate_id": row["candidate_id"],
                    "event_id": event.event_id,
                    "video": row["video"],
                    "scores": scores,
                    "track_presence": row["track_presence"],
                    "split": event.split,
                    "audit_valid": row.get("audit_valid"),
                }
            )
        write_jsonl(args.output, result)
    elif args.command == "calibrate":
        candidates = [Candidate.from_dict(row) for row in read_jsonl(args.candidates)]
        if args.strict_paper_size and (
            len(candidates) != 1000 or len({c.event_id for c in candidates}) != 250
        ):
            raise ValueError("Paper calibration requires 1,000 candidates from 250 events")
        write_json(args.output, calibrate(candidates, search=not args.paper_weights).to_dict())
    elif args.command == "select":
        candidates = [Candidate.from_dict(row) for row in read_jsonl(args.candidates)]
        selected = Calibrator(**read_json(args.calibrator)).select(candidates)
        write_jsonl(
            args.output,
            [{**candidate.to_dict(), "score": score} for candidate, score in selected.values()],
        )
    elif args.command == "build":
        from .pipeline import build_dataset

        units = [Unit(**row) for row in read_jsonl(args.units)]
        selected = {}
        for row in read_jsonl(args.selected):
            score = row.pop("score")
            candidate = Candidate.from_dict(row)
            if candidate.event_id in selected:
                raise ValueError("At most one candidate per event is permitted")
            selected[candidate.event_id] = (candidate, score)
        records, skipped = build_dataset(
            _events(args.events), units, selected, args.output, args.qa_mode
        )
        print(f"Built {len(records)} QA examples; skipped {len(skipped)} incomplete units")
    elif args.command == "evaluate":
        from .metrics import control_metrics, hallucination_metrics, structural_metrics

        metric = {
            "structural": structural_metrics,
            "hallucination": hallucination_metrics,
            "control": control_metrics,
        }[args.kind]
        write_json(args.output, metric(list(read_jsonl(args.records))))
    elif args.command == "train":
        from .training import train

        train(args.config, args.data, args.output, args.resume)
    elif args.command == "predict":
        from .inference import predict

        predict(args.model, args.data, args.output, args.adapter)


if __name__ == "__main__":
    main()
