"""Run plan-then-execute models and score predictions."""

from __future__ import annotations

import json
from pathlib import Path

from .dataset import load_jsonl, resolve_item_paths
from .evaluation import filter_items, score_records, write_predictions
from .plan_executor import PlanClient, run_plan_executor


def run_plan_executor_suite(
    manifest_path: str | Path,
    client: PlanClient,
    output_dir: str | Path,
    split: str | None = "test",
    joint_threshold: float = 0.9,
    runner_name: str = "plan_executor",
    max_items: int | None = None,
) -> dict:
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    root = manifest_path.parent
    items = filter_items(
        [resolve_item_paths(item, root) for item in load_jsonl(manifest_path)],
        split=split,
    )
    if max_items is not None:
        items = items[:max_items]
    if not items:
        return {
            "manifest": str(manifest_path),
            "split": split,
            "runner": runner_name,
            "error": "no_items_for_split",
        }

    predictions = run_plan_executor(items, client, output_dir / "midi")
    pred_path = output_dir / "predictions.jsonl"
    write_predictions(pred_path, predictions)

    scores = score_records(
        manifest_path,
        pred_path,
        joint_threshold=joint_threshold,
        split=split,
        item_ids={item.item_id for item in items},
    )
    failures = [
        {
            "item_id": pred.item_id,
            "error": pred.metadata.get("error"),
            "raw_response": pred.metadata.get("raw_response"),
        }
        for pred in predictions
        if pred.metadata.get("error") or pred.midi_path is None
    ]

    summary = {
        "manifest": str(manifest_path),
        "split": split,
        "runner": runner_name,
        "item_count": len(items),
        "joint_threshold": joint_threshold,
        "predictions": str(pred_path),
        "overall": scores["overall"],
        "by_op_family": scores["by_op_family"],
        "failure_count": len(failures),
        "failures": failures[:20],
    }
    report_path = output_dir / "plan_executor_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["report"] = str(report_path)
    return summary
