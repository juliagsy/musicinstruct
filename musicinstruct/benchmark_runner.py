"""Run reference baselines and aggregate scores."""

from __future__ import annotations

import json
from pathlib import Path

from .baselines import run_baselines
from .dataset import load_jsonl, resolve_item_paths
from .evaluation import filter_items, score_records, write_predictions


def run_baseline_suite(
    manifest_path: str | Path,
    output_dir: str | Path,
    split: str | None = "test",
    baselines: list[str] | None = None,
    joint_threshold: float = 0.9,
) -> dict:
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    root = manifest_path.parent
    items = filter_items(
        [resolve_item_paths(item, root) for item in load_jsonl(manifest_path)],
        split=split,
    )
    if not items:
        return {
            "manifest": str(manifest_path),
            "split": split,
            "error": "no_items_for_split",
            "baselines": {},
        }

    cache_dir = output_dir / "cache"
    predictions_by_baseline = run_baselines(items, cache_dir, baselines=baselines)

    summary: dict = {
        "manifest": str(manifest_path),
        "split": split,
        "item_count": len(items),
        "joint_threshold": joint_threshold,
        "baselines": {},
    }

    for name, predictions in predictions_by_baseline.items():
        pred_path = output_dir / f"predictions_{name}.jsonl"
        write_predictions(pred_path, predictions)
        scores = score_records(
            manifest_path,
            pred_path,
            joint_threshold=joint_threshold,
            split=split,
        )
        summary["baselines"][name] = {
            "predictions": str(pred_path),
            "overall": scores["overall"],
            "by_op_family": scores["by_op_family"],
        }

    report_path = output_dir / "baseline_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["report"] = str(report_path)
    return summary
