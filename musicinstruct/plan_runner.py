"""Run plan-then-execute models and score predictions."""

from __future__ import annotations

import json
from pathlib import Path

from .dataset import load_jsonl, resolve_item_paths
from .evaluation import filter_items, load_predictions, merge_predictions, score_records, write_predictions
from .plan_executor import PlanClient, predict_plan_executor
from .schema import BenchmarkItem, Prediction


def _recover_predictions_from_midi(
    output_dir: Path,
    items: list[BenchmarkItem],
    existing: dict[str, Prediction],
) -> dict[str, Prediction]:
    midi_dir = output_dir / "midi"
    if not midi_dir.is_dir():
        return existing
    recovered = dict(existing)
    for item in items:
        if item.item_id in recovered:
            continue
        midi_path = midi_dir / f"{item.item_id}.mid"
        if midi_path.is_file():
            recovered[item.item_id] = Prediction(
                item_id=item.item_id,
                midi_path=str(midi_path),
                metadata={"recovered_from_midi": True},
            )
    return recovered


def run_plan_executor_suite(
    manifest_path: str | Path,
    client: PlanClient,
    output_dir: str | Path,
    split: str | None = "test",
    joint_threshold: float = 0.9,
    runner_name: str = "plan_executor",
    max_items: int | None = None,
    offset: int = 0,
    resume: bool = False,
) -> dict:
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    root = manifest_path.parent
    split_items = filter_items(
        [resolve_item_paths(item, root) for item in load_jsonl(manifest_path)],
        split=split,
    )
    pred_path = output_dir / "predictions.jsonl"
    existing_predictions: dict[str, Prediction] = {}
    if resume:
        if pred_path.is_file():
            existing_predictions = load_predictions(pred_path)
        existing_predictions = _recover_predictions_from_midi(
            output_dir,
            list(split_items),
            existing_predictions,
        )
        if existing_predictions and not pred_path.is_file():
            write_predictions(pred_path, list(existing_predictions.values()))

    items = list(split_items)
    if resume:
        items = [item for item in items if item.item_id not in existing_predictions]
    if offset:
        items = items[offset:]
    if max_items is not None:
        items = items[:max_items]
    if not items:
        if not split_items:
            return {
                "manifest": str(manifest_path),
                "split": split,
                "runner": runner_name,
                "error": "no_items_for_split",
            }
        scores = score_records(
            manifest_path,
            pred_path,
            joint_threshold=joint_threshold,
            split=split,
        ) if pred_path.is_file() else {"overall": {}, "prediction_coverage": 0.0}
        return {
            "manifest": str(manifest_path),
            "split": split,
            "runner": runner_name,
            "batch_count": 0,
            "total_predictions": len(existing_predictions),
            "prediction_coverage": scores.get("prediction_coverage", 0.0),
            "message": "no new items to process",
            "predictions": str(pred_path) if pred_path.is_file() else None,
        }

    midi_dir = output_dir / "midi"
    batch_predictions: list[Prediction] = []
    merged_predictions = dict(existing_predictions)
    for item in items:
        pred = predict_plan_executor(item, client, midi_dir)
        batch_predictions.append(pred)
        merged_predictions = merge_predictions(merged_predictions, [pred])
        write_predictions(pred_path, list(merged_predictions.values()))

    scores = score_records(
        manifest_path,
        pred_path,
        joint_threshold=joint_threshold,
        split=split,
    )
    failures = [
        {
            "item_id": pred.item_id,
            "error": pred.metadata.get("error"),
            "raw_response": pred.metadata.get("raw_response"),
        }
        for pred in batch_predictions
        if pred.metadata.get("error") or pred.midi_path is None
    ]

    summary = {
        "manifest": str(manifest_path),
        "split": split,
        "runner": runner_name,
        "batch_count": len(items),
        "total_predictions": len(merged_predictions),
        "split_count": len(split_items),
        "prediction_coverage": scores["prediction_coverage"],
        "missing_prediction_ids": scores["missing_prediction_ids"],
        "joint_threshold": joint_threshold,
        "predictions": str(pred_path),
        "overall": scores["overall"],
        "by_op_family": scores["by_op_family"],
        "batch_failure_count": len(failures),
        "batch_failures": failures[:20],
    }
    report_path = output_dir / "plan_executor_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["report"] = str(report_path)
    return summary
