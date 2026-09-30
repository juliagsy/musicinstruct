"""Benchmark scoring over gold items and model predictions."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from .dataset import load_jsonl, resolve_item_paths
from .metrics import ItemScores, aggregate_scores, score_item
from .schema import BenchmarkItem, Plan, Prediction

SELF_TEST_EDIT_SUCCESS_MIN = 0.999
SELF_TEST_PRESERVE_MIN = 1.0
SELF_TEST_JOINT_MIN = 1.0


def filter_items(items: list[BenchmarkItem], split: str | None = None) -> list[BenchmarkItem]:
    if split is None:
        return items
    return [item for item in items if item.split == split]


def _relative_midi_path(path: str | None, base_dir: Path) -> str | None:
    if path is None:
        return None
    midi_path = Path(path)
    base = base_dir.resolve()
    if midi_path.is_absolute():
        try:
            return str(midi_path.resolve().relative_to(base))
        except ValueError:
            return str(midi_path)
    return str(midi_path)


def score_records(
    gold_path: str | Path,
    predictions_path: str | Path,
    joint_threshold: float = 0.9,
    split: str | None = None,
    item_ids: set[str] | None = None,
    strict: bool = False,
) -> dict:
    gold_root = Path(gold_path).parent
    items = filter_items(
        [resolve_item_paths(item, gold_root) for item in load_jsonl(gold_path)],
        split=split,
    )
    if item_ids is not None:
        items = [item for item in items if item.item_id in item_ids]
    predictions = load_predictions(predictions_path)

    gold_ids = {item.item_id for item in items}
    matched_ids = gold_ids & set(predictions)
    missing_prediction_ids = sorted(gold_ids - matched_ids)
    prediction_coverage = len(matched_ids) / len(items) if items else 0.0

    if strict and missing_prediction_ids:
        raise ValueError(
            f"missing predictions for {len(missing_prediction_ids)} gold item(s): "
            f"{missing_prediction_ids[:5]}"
            + (" ..." if len(missing_prediction_ids) > 5 else "")
        )

    pred_root = Path(predictions_path).parent
    rows: list[ItemScores] = []
    for item in items:
        pred = predictions.get(item.item_id)
        hypothesis: str | None = None
        plan = pred.plan if pred else None
        if pred and pred.midi_path:
            path = Path(pred.midi_path)
            hypothesis = str(path if path.is_absolute() else (pred_root / path).resolve())
        rows.append(score_item(item, hypothesis, plan, joint_threshold=joint_threshold))

    by_op: dict[str, list[ItemScores]] = {}
    for item, row in zip(items, rows, strict=True):
        by_op.setdefault(item.op_family, []).append(row)

    return {
        "n_gold": len(items),
        "n_predictions": len(predictions),
        "n_scored": len(rows),
        "prediction_coverage": prediction_coverage,
        "missing_prediction_ids": missing_prediction_ids,
        "split": split,
        "joint_threshold": joint_threshold,
        "overall": aggregate_scores(rows),
        "by_op_family": {op: aggregate_scores(group) for op, group in sorted(by_op.items())},
        "items": [row.as_dict() for row in rows],
        "unmatched_prediction_ids": sorted(set(predictions) - gold_ids),
    }


def load_predictions(path: str | Path) -> dict[str, Prediction]:
    predictions: dict[str, Prediction] = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_no} in {path}: {exc}") from exc
            try:
                pred = Prediction.model_validate(row)
            except ValidationError as exc:
                raise ValueError(f"invalid prediction schema at line {line_no} in {path}: {exc}") from exc
            if pred.item_id in predictions:
                raise ValueError(
                    f"duplicate prediction for item_id={pred.item_id!r} at line {line_no} in {path}"
                )
            predictions[pred.item_id] = pred
    return predictions


def gold_predictions(items: list[BenchmarkItem]) -> list[Prediction]:
    """Identity predictions using gold MIDI paths (sanity check)."""
    return [
        Prediction(item_id=item.item_id, midi_path=item.gold_midi, plan=item.plan)
        for item in items
        if item.gold_midi
    ]


def write_predictions(path: str | Path, predictions: list[Prediction]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base_dir = path.parent.resolve()
    with path.open("w", encoding="utf-8") as stream:
        for pred in predictions:
            payload = pred.model_dump()
            payload["midi_path"] = _relative_midi_path(pred.midi_path, base_dir)
            stream.write(json.dumps(payload) + "\n")


def self_test_passed(overall: dict) -> bool:
    return (
        overall.get("joint", 0.0) >= SELF_TEST_JOINT_MIN
        and overall.get("edit_success", 0.0) >= SELF_TEST_EDIT_SUCCESS_MIN
        and overall.get("preserve", 0.0) >= SELF_TEST_PRESERVE_MIN
    )
