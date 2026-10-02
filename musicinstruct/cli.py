"""Command-line interface for MIDI-Instruct."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from .benchmark_runner import run_baseline_suite
from .dataset import load_jsonl, resolve_item_paths, validate_dataset
from .evaluation import gold_predictions, score_records, self_test_passed, write_predictions
from .generate import REAL_TARGET_ITEMS, generate_pilot_dataset, generate_real_from_midicaps
from .llm_client import DEFAULT_LLAMA_MODEL, LlamaPlanClient
from .midi import midi_summary
from .plan_runner import run_plan_executor_suite
from .stress import run_stress_test


def _print_json(data: dict) -> None:
    print(json.dumps(data, indent=2))


def cmd_validate(args: argparse.Namespace) -> int:
    report = validate_dataset(args.dataset)
    _print_json(report)
    return 0 if report["valid"] else 1


def cmd_score(args: argparse.Namespace) -> int:
    try:
        results = score_records(
            args.gold,
            args.predictions,
            joint_threshold=args.joint_threshold,
            split=args.split,
            strict=args.strict,
        )
    except ValueError as exc:
        _print_json({"error": str(exc)})
        return 1
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    _print_json(results)
    return 0


def cmd_generate_pilot(args: argparse.Namespace) -> int:
    try:
        items = generate_pilot_dataset(args.output_dir, target=args.target, force=args.force)
    except FileExistsError as exc:
        _print_json({"error": str(exc)})
        return 1
    manifest = Path(args.output_dir) / "pilot.jsonl"
    report = validate_dataset(manifest)
    _print_json({"generated": len(items), "manifest": str(manifest), "validation": report})
    return 0 if report["valid"] else 1


def cmd_generate_real(args: argparse.Namespace) -> int:
    try:
        items, seeds = generate_real_from_midicaps(
            args.output_dir,
            args.lakh_root,
            target=args.target,
            seed_limit=args.seed_limit,
            midicaps_limit=args.midicaps_limit,
            force=args.force,
            manifest_name=args.manifest_name,
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        _print_json({"error": str(exc)})
        return 1
    manifest = Path(args.output_dir) / args.manifest_name
    report = validate_dataset(manifest)
    split_counts = {}
    composition_counts = {}
    for item in items:
        split_counts[item.split] = split_counts.get(item.split, 0) + 1
        if item.split == "test":
            composition_counts[item.composition_id] = composition_counts.get(item.composition_id, 0) + 1
    _print_json(
        {
            "generated": len(items),
            "seeds": len(seeds),
            "manifest": str(manifest),
            "splits": split_counts,
            "test_compositions": len(composition_counts),
            "validation": report,
        }
    )
    return 0 if report["valid"] else 1


def cmd_self_test(args: argparse.Namespace) -> int:
    manifest = Path(args.dataset)
    root = manifest.parent
    items = [resolve_item_paths(item, root) for item in load_jsonl(manifest)]
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as handle:
        preds_path = Path(handle.name)
    try:
        write_predictions(preds_path, gold_predictions(items))
        results = score_records(
            manifest, preds_path, joint_threshold=args.joint_threshold, strict=True
        )
    finally:
        preds_path.unlink(missing_ok=True)
    passed = self_test_passed(results["overall"])
    _print_json(
        {
            "self_test_passed": passed,
            "overall": results["overall"],
            "prediction_coverage": results["prediction_coverage"],
        }
    )
    return 0 if passed else 1


def cmd_midi_summary(args: argparse.Namespace) -> int:
    _print_json(midi_summary(args.midi))
    return 0


def cmd_run_baselines(args: argparse.Namespace) -> int:
    summary = run_baseline_suite(
        args.dataset,
        args.output_dir,
        split=args.split,
        baselines=args.baselines,
        joint_threshold=args.joint_threshold,
    )
    _print_json(summary)
    return 0 if "error" not in summary else 1


def cmd_run_plan_executor(args: argparse.Namespace) -> int:
    client = LlamaPlanClient(
        model_id=args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    summary = run_plan_executor_suite(
        args.dataset,
        client,
        args.output_dir,
        split=args.split,
        joint_threshold=args.joint_threshold,
        runner_name=f"plan_executor_{Path(args.model).name}",
        max_items=args.max_items,
        offset=args.offset,
        resume=args.resume,
    )
    _print_json(summary)
    if "error" in summary:
        return 1
    if summary.get("batch_failure_count", 0) > 0:
        return 1
    return 0


def cmd_stress_test(args: argparse.Namespace) -> int:
    if not args.midi_dir and not (args.lakh_root and args.midicaps_limit > 0):
        _print_json(
            {
                "error": "provide --midi-dir and/or (--lakh-root with --midicaps-limit > 0)",
            }
        )
        return 1
    report = run_stress_test(
        args.output_dir,
        midi_dir=args.midi_dir,
        lakh_root=args.lakh_root,
        midicaps_limit=args.midicaps_limit,
        midicaps_test_set_only=args.midicaps_test_set_only,
        max_files=args.max_files,
        joint_threshold=args.joint_threshold,
    )
    _print_json(report)
    return 0 if report.get("passed") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="musicinstruct", description="MIDI-Instruct benchmark toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="Validate a JSONL dataset")
    validate.add_argument("dataset")
    validate.set_defaults(func=cmd_validate)

    score = sub.add_parser("score", help="Score predictions against gold items")
    score.add_argument("gold")
    score.add_argument("predictions")
    score.add_argument("--output", default=None)
    score.add_argument("--joint-threshold", type=float, default=0.9)
    score.add_argument("--split", default=None, choices=["train", "validation", "test"])
    score.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any gold item lacks a prediction",
    )
    score.set_defaults(func=cmd_score)

    plan_exec = sub.add_parser(
        "run-plan-executor",
        help="Run Llama plan-then-execute on a split (requires .[llm])",
    )
    plan_exec.add_argument("dataset")
    plan_exec.add_argument("--output-dir", default="results/plan_executor_llama")
    plan_exec.add_argument("--split", default="test", choices=["train", "validation", "test"])
    plan_exec.add_argument("--model", default=DEFAULT_LLAMA_MODEL)
    plan_exec.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    plan_exec.add_argument("--max-new-tokens", type=int, default=256)
    plan_exec.add_argument("--temperature", type=float, default=0.0)
    plan_exec.add_argument("--max-items", type=int, default=None, help="Limit items processed this batch")
    plan_exec.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N pending items in the split before processing",
    )
    plan_exec.add_argument(
        "--resume",
        action="store_true",
        help="Skip items already in predictions.jsonl and merge new results",
    )
    plan_exec.add_argument("--joint-threshold", type=float, default=0.9)
    plan_exec.set_defaults(func=cmd_run_plan_executor)

    baselines = sub.add_parser("run-baselines", help="Run reference baselines on a split")
    baselines.add_argument("dataset")
    baselines.add_argument("--output-dir", default="results/baselines")
    baselines.add_argument("--split", default="test", choices=["train", "validation", "test"])
    baselines.add_argument(
        "--baselines",
        nargs="+",
        default=None,
        choices=["oracle", "copy_source", "wrong_transform", "no_output"],
    )
    baselines.add_argument("--joint-threshold", type=float, default=0.9)
    baselines.set_defaults(func=cmd_run_baselines)

    stress = sub.add_parser("stress-test", help="Stress-test grader on external MIDIs")
    stress.add_argument("--output-dir", default="data/stress")
    stress.add_argument("--midi-dir", default=None, help="Directory of .mid files to scan")
    stress.add_argument("--lakh-root", default=None, help="Root of Lakh MIDI (for MidiCaps paths)")
    stress.add_argument("--midicaps-limit", type=int, default=0, help="Max MidiCaps rows to resolve")
    stress.add_argument("--midicaps-test-set-only", action="store_true")
    stress.add_argument("--max-files", type=int, default=50)
    stress.add_argument("--joint-threshold", type=float, default=0.9)
    stress.set_defaults(func=cmd_stress_test)

    generate = sub.add_parser("generate-pilot", help="Generate synthetic pilot dataset")
    generate.add_argument("--output-dir", default="data/pilot")
    generate.add_argument("--target", type=int, default=300)
    generate.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output directory",
    )
    generate.set_defaults(func=cmd_generate_pilot)

    generate_real = sub.add_parser(
        "generate-real",
        help="Generate benchmark from filtered MidiCaps/Lakh seeds (requires .[stress])",
    )
    generate_real.add_argument("--lakh-root", required=True, help="Root of Lakh MIDI tree (lmd_full/)")
    generate_real.add_argument("--output-dir", default="data/v0.2")
    generate_real.add_argument("--target", type=int, default=REAL_TARGET_ITEMS)
    generate_real.add_argument("--seed-limit", type=int, default=500, help="Max composition seeds")
    generate_real.add_argument("--midicaps-limit", type=int, default=5000, help="Max MidiCaps rows to scan")
    generate_real.add_argument("--manifest-name", default="manifest.jsonl")
    generate_real.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output directory",
    )
    generate_real.set_defaults(func=cmd_generate_real)

    self_test = sub.add_parser("self-test", help="Score gold MIDIs against themselves")
    self_test.add_argument("dataset")
    self_test.add_argument("--joint-threshold", type=float, default=0.9)
    self_test.set_defaults(func=cmd_self_test)

    summary = sub.add_parser("midi-summary", help="Summarize a MIDI file")
    summary.add_argument("midi")
    summary.set_defaults(func=cmd_midi_summary)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
