"""Aggregate dual-metric scores for MIDI-Instruct."""

from __future__ import annotations

from dataclasses import dataclass

from .midi import extract_notes, load_midi, note_f1, over_edit_rate, partition_notes, validate_midi
from .predicates import PredicateResult, evaluate_predicates
from .schema import BenchmarkItem, Plan


@dataclass
class ItemScores:
    item_id: str
    validity: float
    edit_success: float
    preserve: float
    joint: float
    over_edit: float
    gold_note_f1: float | None
    plan_match: float | None
    edit_details: list[dict]
    preserve_details: list[dict]

    def as_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "validity": self.validity,
            "edit_success": self.edit_success,
            "preserve": self.preserve,
            "joint": self.joint,
            "over_edit": self.over_edit,
            "gold_note_f1": self.gold_note_f1,
            "plan_match": self.plan_match,
            "edit_details": self.edit_details,
            "preserve_details": self.preserve_details,
        }


def _mean_score(results: list[PredicateResult]) -> float:
    if not results:
        return 0.0
    return sum(r.score for r in results) / len(results)


def _details(results: list[PredicateResult]) -> list[dict]:
    return [
        {
            "name": r.name,
            "passed": bool(r.passed),
            "score": float(r.score),
            "detail": r.detail,
        }
        for r in results
    ]


PLAN_PARAM_FLOAT_TOLERANCE = 1e-6


def _plan_param_values_match(gold_value: object, predicted_value: object) -> bool:
    if isinstance(gold_value, float) or isinstance(predicted_value, float):
        try:
            return abs(float(gold_value) - float(predicted_value)) <= PLAN_PARAM_FLOAT_TOLERANCE
        except (TypeError, ValueError):
            return False
    if isinstance(gold_value, list) and isinstance(predicted_value, list):
        if len(gold_value) != len(predicted_value):
            return False
        return all(
            _plan_param_values_match(g, p) for g, p in zip(gold_value, predicted_value, strict=True)
        )
    return gold_value == predicted_value


def plan_params_match(gold_params: dict, predicted_params: dict) -> bool:
    if set(gold_params) != set(predicted_params):
        return False
    return all(
        _plan_param_values_match(gold_params[key], predicted_params[key]) for key in gold_params
    )


def plan_exact_match(gold: Plan | None, predicted: Plan | None) -> float | None:
    if gold is None:
        return None
    if predicted is None:
        return 0.0
    if gold.op != predicted.op:
        return 0.0
    return 1.0 if plan_params_match(gold.params, predicted.params) else 0.0


def score_item(
    item: BenchmarkItem,
    hypothesis_path: str | None,
    predicted_plan: Plan | None = None,
    joint_threshold: float = 0.9,
) -> ItemScores:
    if hypothesis_path is None:
        return ItemScores(
            item_id=item.item_id,
            validity=0.0,
            edit_success=0.0,
            preserve=0.0,
            joint=0.0,
            over_edit=1.0,
            gold_note_f1=None,
            plan_match=plan_exact_match(item.plan, predicted_plan),
            edit_details=[],
            preserve_details=[],
        )

    valid, _ = validate_midi(hypothesis_path)
    validity = 1.0 if valid else 0.0
    if not valid:
        return ItemScores(
            item_id=item.item_id,
            validity=0.0,
            edit_success=0.0,
            preserve=0.0,
            joint=0.0,
            over_edit=1.0,
            gold_note_f1=None,
            plan_match=plan_exact_match(item.plan, predicted_plan),
            edit_details=[],
            preserve_details=[],
        )

    if not item.must_change or not item.must_preserve:
        return ItemScores(
            item_id=item.item_id,
            validity=validity,
            edit_success=0.0,
            preserve=0.0,
            joint=0.0,
            over_edit=1.0,
            gold_note_f1=None,
            plan_match=plan_exact_match(item.plan, predicted_plan),
            edit_details=[],
            preserve_details=[],
        )

    try:
        source_midi = load_midi(item.midi_in)
        hyp_midi = load_midi(hypothesis_path)
    except FileNotFoundError:
        return ItemScores(
            item_id=item.item_id,
            validity=0.0,
            edit_success=0.0,
            preserve=0.0,
            joint=0.0,
            over_edit=1.0,
            gold_note_f1=0.0 if item.gold_mode == "unique" and item.gold_midi else None,
            plan_match=plan_exact_match(item.plan, predicted_plan),
            edit_details=[],
            preserve_details=[],
        )
    source_notes = extract_notes(source_midi)
    hyp_notes = extract_notes(hyp_midi)
    edit_results = evaluate_predicates(
        item.must_change,
        item.midi_in,
        hypothesis_path,
        item.edit_mask,
        source_midi=source_midi,
        hyp_midi=hyp_midi,
    )
    preserve_results = evaluate_predicates(
        item.must_preserve,
        item.midi_in,
        hypothesis_path,
        item.edit_mask,
        source_midi=source_midi,
        hyp_midi=hyp_midi,
    )
    edit_success = _mean_score(edit_results)
    preserve = _mean_score(preserve_results)
    joint = 1.0 if edit_success >= joint_threshold and preserve >= joint_threshold else 0.0

    over_edit = over_edit_rate(source_notes, hyp_notes, item.edit_mask)

    gold_f1: float | None = None
    if item.gold_mode == "unique" and item.gold_midi:
        inside_mask, _ = partition_notes(extract_notes(load_midi(item.gold_midi)), item.edit_mask)
        hyp_inside, _ = partition_notes(hyp_notes, item.edit_mask)
        gold_f1 = note_f1(inside_mask, hyp_inside)

    return ItemScores(
        item_id=item.item_id,
        validity=validity,
        edit_success=edit_success,
        preserve=preserve,
        joint=joint,
        over_edit=over_edit,
        gold_note_f1=gold_f1,
        plan_match=plan_exact_match(item.plan, predicted_plan),
        edit_details=_details(edit_results),
        preserve_details=_details(preserve_results),
    )


def aggregate_scores(rows: list[ItemScores]) -> dict:
    if not rows:
        return {"count": 0}
    n = len(rows)
    plan_rows = [float(r.plan_match) for r in rows if r.plan_match is not None]
    return {
        "count": n,
        "validity": float(sum(r.validity for r in rows) / n),
        "edit_success": float(sum(r.edit_success for r in rows) / n),
        "preserve": float(sum(r.preserve for r in rows) / n),
        "joint": float(sum(r.joint for r in rows) / n),
        "over_edit": float(sum(r.over_edit for r in rows) / n),
        "gold_note_f1": float(
            sum(float(r.gold_note_f1) if r.gold_note_f1 is not None else 0.0 for r in rows) / n
        ),
        "plan_match": float(sum(plan_rows) / len(plan_rows)) if plan_rows else None,
    }


def aggregate_by_composition(
    items: list[BenchmarkItem],
    rows: list[ItemScores],
) -> tuple[dict, dict[str, dict]]:
    """Return composition-macro scores and per-composition item-micro aggregates."""
    by_composition: dict[str, list[ItemScores]] = {}
    for item, row in zip(items, rows, strict=True):
        by_composition.setdefault(item.composition_id, []).append(row)

    if not by_composition:
        return {"count": 0, "n_compositions": 0}, {}

    per_composition = {
        composition_id: aggregate_scores(comp_rows)
        for composition_id, comp_rows in sorted(by_composition.items())
    }
    n_compositions = len(per_composition)
    composition_macro = {
        "count": sum(stats["count"] for stats in per_composition.values()),
        "n_compositions": n_compositions,
        "validity": float(
            sum(stats["validity"] for stats in per_composition.values()) / n_compositions
        ),
        "edit_success": float(
            sum(stats["edit_success"] for stats in per_composition.values()) / n_compositions
        ),
        "preserve": float(
            sum(stats["preserve"] for stats in per_composition.values()) / n_compositions
        ),
        "joint": float(sum(stats["joint"] for stats in per_composition.values()) / n_compositions),
        "over_edit": float(
            sum(stats["over_edit"] for stats in per_composition.values()) / n_compositions
        ),
        "gold_note_f1": float(
            sum(stats["gold_note_f1"] for stats in per_composition.values()) / n_compositions
        ),
        "plan_match": _macro_plan_match(per_composition),
    }
    return composition_macro, per_composition


def _macro_plan_match(per_composition: dict[str, dict]) -> float | None:
    plan_values = [
        stats["plan_match"]
        for stats in per_composition.values()
        if stats.get("plan_match") is not None
    ]
    if not plan_values:
        return None
    return float(sum(plan_values) / len(plan_values))
