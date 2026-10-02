"""Benchmark dataset generation (synthetic pilot and real MidiCaps/Lakh seeds)."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

from .dataset import save_jsonl
from .schema import BenchmarkItem, Split
from .seeds import SeedRecord, filter_midicaps_seeds
from .transforms import (
    make_seed_midi,
    mute_tracks,
    non_drum_tracks,
    program_change,
    tempo_scale,
    transpose,
    velocity_scale,
)

SEED_COUNT = 12
TARGET_ITEMS = 300
REAL_TARGET_ITEMS = 1000
DEFAULT_TEST_MAX_PER_COMPOSITION = 6

TRANSPOSE_SEMITONES = [-7, -5, -3, -2, 2, 3, 5, 7]
VELOCITY_FACTORS = [0.5, 0.75, 1.25]
TEMPO_FACTORS = [0.75, 1.25, 1.5]
PROGRAMS = [25, 40, 48, 56]

TRANSPOSE_TEMPLATES = [
    "Transpose the non-drum tracks up by {n} semitones.",
    "Raise all pitched instruments by {n} semitones.",
    "Shift the melody and bass up {n} semitones.",
    "Transpose the non-drum tracks down by {n} semitones.",
    "Lower all pitched instruments by {abs_n} semitones.",
]

VELOCITY_TEMPLATE = "Scale velocities on track(s) {tracks} by factor {factor}."
TEMPO_TEMPLATE = "Change the tempo by factor {factor} (faster if >1)."
MUTE_TEMPLATE = "Mute track(s) {tracks} completely."
PROGRAM_TEMPLATE = "Change track {track} to General MIDI program {program}."


def _split_for_seed(seed_index: int) -> str:
    if seed_index < 8:
        return "train"
    if seed_index < 10:
        return "validation"
    return "test"


def _trim_items_by_op_family(items: list[BenchmarkItem], target: int) -> list[BenchmarkItem]:
    if len(items) <= target:
        return items
    by_op: dict[str, list[BenchmarkItem]] = {}
    for item in items:
        by_op.setdefault(item.op_family, []).append(item)
    for bucket in by_op.values():
        bucket.sort(key=lambda item: item.item_id)
    total = len(items)
    trimmed: list[BenchmarkItem] = []
    remaining = target
    op_names = sorted(by_op)
    for index, op in enumerate(op_names):
        bucket = by_op[op]
        if index == len(op_names) - 1:
            take = remaining
        else:
            take = max(1, round(target * len(bucket) / total))
            remaining -= take
        trimmed.extend(bucket[:take])
    return sorted(trimmed, key=lambda item: item.item_id)[:target]


def subsample_test_items(
    items: list[BenchmarkItem],
    *,
    max_per_composition: int = DEFAULT_TEST_MAX_PER_COMPOSITION,
    seed: int = 0,
) -> list[BenchmarkItem]:
    """Keep train/validation intact; cap test to one random item per op_family per composition."""
    if max_per_composition <= 0:
        return items

    rng = random.Random(seed)
    train_val = [item for item in items if item.split != "test"]
    test_items = [item for item in items if item.split == "test"]
    if not test_items:
        return items

    by_composition: dict[str, list[BenchmarkItem]] = {}
    for item in test_items:
        by_composition.setdefault(item.composition_id, []).append(item)

    subsampled_test: list[BenchmarkItem] = []
    for composition_id in sorted(by_composition):
        comp_items = by_composition[composition_id]
        by_op: dict[str, list[BenchmarkItem]] = {}
        for item in comp_items:
            by_op.setdefault(item.op_family, []).append(item)

        selected: list[BenchmarkItem] = []
        op_names = sorted(by_op)
        rng.shuffle(op_names)
        for op in op_names:
            if len(selected) >= max_per_composition:
                break
            bucket = sorted(by_op[op], key=lambda item: item.item_id)
            selected.append(rng.choice(bucket))
        subsampled_test.extend(selected)

    return sorted(train_val + subsampled_test, key=lambda item: item.item_id)


def _trim_to_target(items: list[BenchmarkItem], target: int) -> list[BenchmarkItem]:
    """Trim proportionally within each composition so every seed keeps representation."""
    if len(items) <= target:
        return items
    by_composition: dict[str, list[BenchmarkItem]] = {}
    for item in items:
        by_composition.setdefault(item.composition_id, []).append(item)
    composition_ids = sorted(by_composition)
    base = target // len(composition_ids)
    remainder = target % len(composition_ids)
    trimmed: list[BenchmarkItem] = []
    for index, composition_id in enumerate(composition_ids):
        take = base + (1 if index < remainder else 0)
        trimmed.extend(_trim_items_by_op_family(by_composition[composition_id], take))
    return sorted(trimmed, key=lambda item: item.item_id)


def _copy_seed(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise OutputDirectoryExistsError(
                f"{output_dir} already exists; pass force=True or use --force to overwrite"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _build_items_for_composition(
    *,
    output_dir: Path,
    composition_id: str,
    split: Split,
    seed_in: Path,
    source: str,
    license_tag: str,
    metadata_base: dict | None = None,
) -> list[BenchmarkItem]:
    """Emit unique-gold items for one source MIDI using the v0.1 operator mix."""
    metadata_base = metadata_base or {}
    pitched = non_drum_tracks(seed_in)
    if not pitched:
        return []

    velocity_track = pitched[0]
    program_track = pitched[0]
    items: list[BenchmarkItem] = []

    for semitones in TRANSPOSE_SEMITONES:
        templates = (
            TRANSPOSE_TEMPLATES[:2]
            if semitones > 0
            else [TRANSPOSE_TEMPLATES[3], TRANSPOSE_TEMPLATES[4]]
        )
        for template_idx, template in enumerate(templates):
            item_id = f"{composition_id}_transpose_{semitones:+d}_p{template_idx}"
            gold_path = output_dir / "gold" / f"{item_id}.mid"
            result = transpose(seed_in, gold_path, semitones=semitones, tracks=pitched)
            instruction = template.format(n=abs(semitones), abs_n=abs(semitones))
            items.append(
                BenchmarkItem(
                    item_id=item_id,
                    composition_id=composition_id,
                    gold_mode="unique",
                    op_family="transpose",
                    instruction=instruction,
                    instruction_type="specific",
                    midi_in=str(seed_in.relative_to(output_dir)),
                    gold_midi=str(gold_path.relative_to(output_dir)),
                    plan=result.plan,
                    must_change=result.must_change,
                    must_preserve=result.must_preserve,
                    edit_mask=result.edit_mask,
                    split=split,
                    source=source,
                    license=license_tag,
                    annotation_status="reviewed",
                    metadata={**metadata_base, "semitones": semitones, "template_idx": template_idx},
                )
            )

    for factor in VELOCITY_FACTORS:
        item_id = f"{composition_id}_velocity_{factor}"
        gold_path = output_dir / "gold" / f"{item_id}.mid"
        tracks = [velocity_track]
        result = velocity_scale(seed_in, gold_path, factor=factor, tracks=tracks)
        items.append(
            BenchmarkItem(
                item_id=item_id,
                composition_id=composition_id,
                gold_mode="unique",
                op_family="velocity_scale",
                instruction=VELOCITY_TEMPLATE.format(tracks=tracks, factor=factor),
                midi_in=str(seed_in.relative_to(output_dir)),
                gold_midi=str(gold_path.relative_to(output_dir)),
                plan=result.plan,
                must_change=result.must_change,
                must_preserve=result.must_preserve,
                edit_mask=result.edit_mask,
                split=split,
                source=source,
                license=license_tag,
                annotation_status="reviewed",
                metadata={**metadata_base, "factor": factor},
            )
        )

    for factor in TEMPO_FACTORS:
        item_id = f"{composition_id}_tempo_{factor}"
        gold_path = output_dir / "gold" / f"{item_id}.mid"
        result = tempo_scale(seed_in, gold_path, factor=factor)
        items.append(
            BenchmarkItem(
                item_id=item_id,
                composition_id=composition_id,
                gold_mode="unique",
                op_family="tempo_scale",
                instruction=TEMPO_TEMPLATE.format(factor=factor),
                midi_in=str(seed_in.relative_to(output_dir)),
                gold_midi=str(gold_path.relative_to(output_dir)),
                plan=result.plan,
                must_change=result.must_change,
                must_preserve=result.must_preserve,
                edit_mask=result.edit_mask,
                split=split,
                source=source,
                license=license_tag,
                annotation_status="reviewed",
                metadata={**metadata_base, "factor": factor},
            )
        )

    for track in pitched[:2]:
        item_id = f"{composition_id}_mute_{track}"
        gold_path = output_dir / "gold" / f"{item_id}.mid"
        result = mute_tracks(seed_in, gold_path, tracks=[track])
        items.append(
            BenchmarkItem(
                item_id=item_id,
                composition_id=composition_id,
                gold_mode="unique",
                op_family="mute_tracks",
                instruction=MUTE_TEMPLATE.format(tracks=[track]),
                midi_in=str(seed_in.relative_to(output_dir)),
                gold_midi=str(gold_path.relative_to(output_dir)),
                plan=result.plan,
                must_change=result.must_change,
                must_preserve=result.must_preserve,
                edit_mask=result.edit_mask,
                split=split,
                source=source,
                license=license_tag,
                annotation_status="reviewed",
                metadata={**metadata_base, "tracks": [track]},
            )
        )

    for program in PROGRAMS:
        item_id = f"{composition_id}_program_{program}"
        gold_path = output_dir / "gold" / f"{item_id}.mid"
        result = program_change(seed_in, gold_path, track=program_track, program=program)
        items.append(
            BenchmarkItem(
                item_id=item_id,
                composition_id=composition_id,
                gold_mode="unique",
                op_family="program_change",
                instruction=PROGRAM_TEMPLATE.format(track=program_track, program=program),
                midi_in=str(seed_in.relative_to(output_dir)),
                gold_midi=str(gold_path.relative_to(output_dir)),
                plan=result.plan,
                must_change=result.must_change,
                must_preserve=result.must_preserve,
                edit_mask=result.edit_mask,
                split=split,
                source=source,
                license=license_tag,
                annotation_status="reviewed",
                metadata={**metadata_base, "program": program},
            )
        )

    return items


def build_seeds(output_dir: Path) -> list[Path]:
    seed_dir = output_dir / "seeds"
    seed_dir.mkdir(parents=True, exist_ok=True)
    return [make_seed_midi(seed_dir / f"seed_{idx:02d}.mid", seed_index=idx) for idx in range(SEED_COUNT)]


class OutputDirectoryExistsError(FileExistsError):
    """Raised when generation would overwrite an existing output directory."""


def generate_pilot_dataset(
    output_dir: str | Path,
    target: int = TARGET_ITEMS,
    force: bool = False,
) -> list[BenchmarkItem]:
    output_dir = Path(output_dir)
    _prepare_output_dir(output_dir, force)

    seeds = build_seeds(output_dir)
    items: list[BenchmarkItem] = []

    for seed_index, seed_path in enumerate(seeds):
        composition_id = f"seed_{seed_index:02d}"
        split = _split_for_seed(seed_index)
        seed_in = output_dir / "midi_in" / composition_id / "source.mid"
        _copy_seed(seed_path, seed_in)
        items.extend(
            _build_items_for_composition(
                output_dir=output_dir,
                composition_id=composition_id,
                split=split,
                seed_in=seed_in,
                source="synthetic",
                license_tag="CC0-1.0",
            )
        )

    items.sort(key=lambda item: item.item_id)
    if len(items) > target:
        items = _trim_to_target(items, target)

    manifest = output_dir / "pilot.jsonl"
    save_jsonl(manifest, items)
    return items


def generate_real_dataset(
    output_dir: str | Path,
    seeds: list[SeedRecord],
    *,
    target: int = REAL_TARGET_ITEMS,
    force: bool = False,
    manifest_name: str = "manifest.jsonl",
    test_max_per_composition: int | None = DEFAULT_TEST_MAX_PER_COMPOSITION,
    test_subsample_seed: int = 0,
) -> list[BenchmarkItem]:
    """Generate unique-gold items from filtered real MIDI seeds."""
    if not seeds:
        raise ValueError("no seeds provided for real dataset generation")

    output_dir = Path(output_dir)
    _prepare_output_dir(output_dir, force)

    items: list[BenchmarkItem] = []
    for seed in seeds:
        seed_in = output_dir / "midi_in" / seed.composition_id / "source.mid"
        _copy_seed(seed.source_path, seed_in)
        metadata_base = {
            "origin": str(seed.source_path),
            "cluster_id": seed.cluster_id,
        }
        if seed.location:
            metadata_base["midicaps_location"] = seed.location
        if seed.caption:
            metadata_base["midicaps_caption"] = seed.caption
        if seed.midicaps_test_set:
            metadata_base["midicaps_test_set"] = True

        items.extend(
            _build_items_for_composition(
                output_dir=output_dir,
                composition_id=seed.composition_id,
                split=seed.split,
                seed_in=seed_in,
                source="midicaps",
                license_tag="CC-BY-4.0",
                metadata_base=metadata_base,
            )
        )

    items.sort(key=lambda item: item.item_id)
    if len(items) > target:
        items = _trim_to_target(items, target)
    if test_max_per_composition is not None:
        items = subsample_test_items(
            items,
            max_per_composition=test_max_per_composition,
            seed=test_subsample_seed,
        )

    manifest = output_dir / manifest_name
    save_jsonl(manifest, items)
    return items


def generate_real_from_midicaps(
    output_dir: str | Path,
    lakh_root: str | Path,
    *,
    target: int = REAL_TARGET_ITEMS,
    seed_limit: int = 500,
    midicaps_limit: int = 5000,
    force: bool = False,
    manifest_name: str = "manifest.jsonl",
    test_max_per_composition: int | None = DEFAULT_TEST_MAX_PER_COMPOSITION,
    test_subsample_seed: int = 0,
) -> tuple[list[BenchmarkItem], list[SeedRecord]]:
    """Discover MidiCaps/Lakh seeds and generate a real-MIDI benchmark manifest."""
    seeds = filter_midicaps_seeds(
        lakh_root,
        seed_limit=seed_limit,
        midicaps_limit=midicaps_limit,
    )
    if not seeds:
        raise RuntimeError(
            "no MidiCaps seeds passed filters; check --lakh-root (lmd_full/ or parent),"
            " extraction, and seed filter thresholds (tracks/bars/notes)"
        )
    items = generate_real_dataset(
        output_dir,
        seeds,
        target=target,
        force=force,
        manifest_name=manifest_name,
        test_max_per_composition=test_max_per_composition,
        test_subsample_seed=test_subsample_seed,
    )
    return items, seeds
