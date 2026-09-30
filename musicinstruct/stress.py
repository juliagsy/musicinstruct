"""Grader stress tests on external MIDI files (e.g. MidiCaps / Lakh)."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from .dataset import save_jsonl
from .evaluation import SELF_TEST_EDIT_SUCCESS_MIN, gold_predictions, score_records, write_predictions
from .midi import validate_midi
from .schema import BenchmarkItem
from .transforms import non_drum_tracks, tempo_scale, transpose

STRESS_OPS = ("transpose", "tempo_scale")


def _slug(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode()).hexdigest()[:10]
    return f"{path.stem}_{digest}"


def resolve_midicaps_paths(
    lakh_root: str | Path,
    limit: int = 100,
    test_set_only: bool = False,
) -> list[Path]:
    """Resolve MidiCaps `location` fields to local Lakh MIDI paths."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "MidiCaps loading requires optional dependency: pip install -e '.[stress]'"
        ) from exc

    lakh_root = Path(lakh_root)
    dataset = load_dataset("amaai-lab/MidiCaps", split="train")
    paths: list[Path] = []
    for row in dataset:
        if test_set_only and not row.get("test_set", False):
            continue
        location = row.get("location")
        if not location:
            continue
        candidate = lakh_root / str(location)
        if not candidate.is_file():
            continue
        ok, _ = validate_midi(candidate)
        if not ok:
            continue
        paths.append(candidate.resolve())
        if len(paths) >= limit:
            break
    return paths


def discover_midi_files(
    midi_dir: str | Path | None = None,
    lakh_root: str | Path | None = None,
    midicaps_limit: int = 0,
    midicaps_test_set_only: bool = False,
    max_files: int = 100,
) -> tuple[list[Path], str]:
    """Collect MIDI paths from a directory and/or MidiCaps manifest."""
    found: list[Path] = []
    source = "none"

    if midi_dir:
        root = Path(midi_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"midi_dir not found: {root}")
        for path in sorted(root.rglob("*.mid")):
            ok, _ = validate_midi(path)
            if ok:
                found.append(path.resolve())
        source = "midi_dir"

    if lakh_root and midicaps_limit > 0:
        caps = resolve_midicaps_paths(
            lakh_root,
            limit=midicaps_limit,
            test_set_only=midicaps_test_set_only,
        )
        found.extend(caps)
        if midi_dir:
            source = "midi_dir+midicaps+lakh"
        else:
            source = "midicaps+lakh"

    # Stable dedupe while preserving order
    seen: set[str] = set()
    unique: list[Path] = []
    for path in found:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)

    if max_files > 0:
        unique = unique[:max_files]

    if not unique and source == "none":
        source = "empty"
    return unique, source


def build_stress_items(
    midi_paths: list[Path],
    output_dir: str | Path,
    semitones: int = 3,
    tempo_factor: float = 1.25,
) -> list[BenchmarkItem]:
    """Build unique-gold stress items (transpose + tempo per file)."""
    output_dir = Path(output_dir)
    midi_in_dir = output_dir / "midi_in"
    gold_dir = output_dir / "gold"
    items: list[BenchmarkItem] = []

    for index, source_path in enumerate(midi_paths):
        slug = _slug(source_path)
        composition_id = f"stress_{index:04d}_{slug}"
        local_in = midi_in_dir / f"{composition_id}.mid"
        local_in.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, local_in)

        pitched = non_drum_tracks(local_in)
        if not pitched:
            continue

        transpose_gold = gold_dir / f"{composition_id}_transpose.mid"
        t_result = transpose(local_in, transpose_gold, semitones=semitones, tracks=pitched)
        items.append(
            BenchmarkItem(
                item_id=f"{composition_id}_transpose_{semitones:+d}",
                composition_id=composition_id,
                gold_mode="unique",
                op_family="transpose",
                instruction=f"Transpose non-drum tracks by {semitones:+d} semitones.",
                midi_in=str(local_in.relative_to(output_dir)),
                gold_midi=str(transpose_gold.relative_to(output_dir)),
                plan=t_result.plan,
                must_change=t_result.must_change,
                must_preserve=t_result.must_preserve,
                edit_mask=t_result.edit_mask,
                split="test",
                source="stress",
                license="see-original",
                annotation_status="reviewed",
                metadata={"origin": str(source_path), "stress_op": "transpose"},
            )
        )

        tempo_gold = gold_dir / f"{composition_id}_tempo.mid"
        s_result = tempo_scale(local_in, tempo_gold, factor=tempo_factor)
        items.append(
            BenchmarkItem(
                item_id=f"{composition_id}_tempo_{tempo_factor}",
                composition_id=composition_id,
                gold_mode="unique",
                op_family="tempo_scale",
                instruction=f"Change tempo by factor {tempo_factor}.",
                midi_in=str(local_in.relative_to(output_dir)),
                gold_midi=str(tempo_gold.relative_to(output_dir)),
                plan=s_result.plan,
                must_change=s_result.must_change,
                must_preserve=s_result.must_preserve,
                edit_mask=s_result.edit_mask,
                split="test",
                source="stress",
                license="see-original",
                annotation_status="reviewed",
                metadata={"origin": str(source_path), "stress_op": "tempo_scale"},
            )
        )

    return items


def run_stress_test(
    output_dir: str | Path,
    midi_dir: str | Path | None = None,
    lakh_root: str | Path | None = None,
    midicaps_limit: int = 0,
    midicaps_test_set_only: bool = False,
    max_files: int = 50,
    joint_threshold: float = 0.9,
) -> dict:
    output_dir = Path(output_dir)
    paths, discovery_source = discover_midi_files(
        midi_dir=midi_dir,
        lakh_root=lakh_root,
        midicaps_limit=midicaps_limit,
        midicaps_test_set_only=midicaps_test_set_only,
        max_files=max_files,
    )
    if not paths:
        return {
            "passed": False,
            "reason": "no_valid_midi_found",
            "discovery_source": discovery_source,
            "output_dir": str(output_dir),
        }

    items = build_stress_items(paths, output_dir)
    if not items:
        return {
            "passed": False,
            "reason": "no_stress_items_built",
            "discovery_source": discovery_source,
            "midi_files": len(paths),
            "output_dir": str(output_dir),
        }

    manifest = output_dir / "stress.jsonl"
    save_jsonl(manifest, items)
    preds_path = output_dir / "_gold_predictions.jsonl"
    write_predictions(preds_path, gold_predictions(items))
    results = score_records(manifest, preds_path, joint_threshold=joint_threshold)

    failures = [
        {
            "item_id": row["item_id"],
            "edit_success": row["edit_success"],
            "preserve": row["preserve"],
            "joint": row["joint"],
        }
        for row in results["items"]
        if row["joint"] < 1.0
    ]

    passed = (
        results["overall"]["joint"] == 1.0
        and results["overall"]["edit_success"] >= SELF_TEST_EDIT_SUCCESS_MIN
    )
    return {
        "passed": passed,
        "discovery_source": discovery_source,
        "midi_files": len(paths),
        "items": len(items),
        "manifest": str(manifest),
        "joint_threshold": joint_threshold,
        "overall": results["overall"],
        "by_op_family": results["by_op_family"],
        "failures": failures[:20],
        "failure_count": len(failures),
    }
