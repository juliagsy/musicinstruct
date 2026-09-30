"""JSONL dataset loading, validation, and path resolution."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from pydantic import ValidationError

from .schema import BenchmarkItem


def load_jsonl(path: str | Path) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                items.append(BenchmarkItem.model_validate_json(line))
            except ValidationError as exc:
                raise ValueError(f"Invalid item at line {line_no}: {exc}") from exc
    return items


def save_jsonl(path: str | Path, items: list[BenchmarkItem]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for item in items:
            stream.write(item.model_dump_json() + "\n")


def resolve_item_paths(item: BenchmarkItem, root: str | Path) -> BenchmarkItem:
    root = Path(root)

    def _resolve(value: str | None) -> str | None:
        if value is None:
            return None
        path = Path(value)
        if path.is_absolute():
            return str(path)
        return str((root / path).resolve())

    data = item.model_dump()
    data["midi_in"] = _resolve(item.midi_in)
    data["gold_midi"] = _resolve(item.gold_midi)
    return BenchmarkItem.model_validate(data)


def validate_dataset(path: str | Path) -> dict:
    root = Path(path).parent
    items = load_jsonl(path)
    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()
    composition_splits: dict[str, set[str]] = defaultdict(set)

    for item in items:
        if item.item_id in seen_ids:
            errors.append(f"duplicate item_id: {item.item_id}")
        seen_ids.add(item.item_id)
        composition_splits[item.composition_id].add(item.split)

        resolved = resolve_item_paths(item, root)
        if not Path(resolved.midi_in).is_file():
            errors.append(f"{item.item_id}: missing midi_in {resolved.midi_in}")
        if item.gold_mode == "unique" and not item.gold_midi:
            errors.append(f"{item.item_id}: unique gold requires gold_midi")
        if item.gold_mode == "unique":
            if not Path(str(resolved.gold_midi)).is_file():
                errors.append(f"{item.item_id}: missing gold_midi {resolved.gold_midi}")
        if not item.must_change:
            errors.append(f"{item.item_id}: must_change is empty")
        if not item.must_preserve:
            errors.append(f"{item.item_id}: must_preserve is empty")

    for composition_id, splits in composition_splits.items():
        if len(splits) > 1:
            errors.append(f"composition {composition_id} appears in multiple splits: {sorted(splits)}")

    split_counts: dict[str, int] = defaultdict(int)
    op_counts: dict[str, int] = defaultdict(int)
    for item in items:
        split_counts[item.split] += 1
        op_counts[item.op_family] += 1

    return {
        "path": str(path),
        "count": len(items),
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "split_counts": dict(sorted(split_counts.items())),
        "op_family_counts": dict(sorted(op_counts.items())),
    }
