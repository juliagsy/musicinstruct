"""Synthetic pilot dataset generation (~300 unique-gold items)."""

from __future__ import annotations

import shutil
from pathlib import Path

from .dataset import save_jsonl
from .schema import BenchmarkItem
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
TARGET_ITEMS = 336

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


def _copy_seed(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_seeds(output_dir: Path) -> list[Path]:
    seed_dir = output_dir / "seeds"
    seed_dir.mkdir(parents=True, exist_ok=True)
    return [make_seed_midi(seed_dir / f"seed_{idx:02d}.mid", seed_index=idx) for idx in range(SEED_COUNT)]


class OutputDirectoryExistsError(FileExistsError):
    """Raised when generate-pilot would overwrite an existing output directory."""


def generate_pilot_dataset(
    output_dir: str | Path,
    target: int = TARGET_ITEMS,
    force: bool = False,
) -> list[BenchmarkItem]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise OutputDirectoryExistsError(
                f"{output_dir} already exists; pass force=True or use --force to overwrite"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = build_seeds(output_dir)
    items: list[BenchmarkItem] = []

    for seed_index, seed_path in enumerate(seeds):
        composition_id = f"seed_{seed_index:02d}"
        split = _split_for_seed(seed_index)
        seed_in = output_dir / "midi_in" / composition_id / "source.mid"
        _copy_seed(seed_path, seed_in)
        pitched = non_drum_tracks(seed_in)

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
                        annotation_status="reviewed",
                        metadata={"semitones": semitones, "template_idx": template_idx},
                    )
                )

        for factor in VELOCITY_FACTORS:
            item_id = f"{composition_id}_velocity_{factor}"
            gold_path = output_dir / "gold" / f"{item_id}.mid"
            tracks = [0]
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
                    annotation_status="reviewed",
                    metadata={"factor": factor},
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
                    annotation_status="reviewed",
                    metadata={"factor": factor},
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
                    annotation_status="reviewed",
                    metadata={"tracks": [track]},
                )
            )

        for program in PROGRAMS:
            track = 0
            item_id = f"{composition_id}_program_{program}"
            gold_path = output_dir / "gold" / f"{item_id}.mid"
            result = program_change(seed_in, gold_path, track=track, program=program)
            items.append(
                BenchmarkItem(
                    item_id=item_id,
                    composition_id=composition_id,
                    gold_mode="unique",
                    op_family="program_change",
                    instruction=PROGRAM_TEMPLATE.format(track=track, program=program),
                    midi_in=str(seed_in.relative_to(output_dir)),
                    gold_midi=str(gold_path.relative_to(output_dir)),
                    plan=result.plan,
                    must_change=result.must_change,
                    must_preserve=result.must_preserve,
                    edit_mask=result.edit_mask,
                    split=split,
                    annotation_status="reviewed",
                    metadata={"program": program},
                )
            )

    items.sort(key=lambda item: item.item_id)
    if len(items) > target:
        by_op: dict[str, list[BenchmarkItem]] = {}
        for item in items:
            by_op.setdefault(item.op_family, []).append(item)
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
        items = sorted(trimmed, key=lambda item: item.item_id)[:target]

    manifest = output_dir / "pilot.jsonl"
    save_jsonl(manifest, items)
    return items
