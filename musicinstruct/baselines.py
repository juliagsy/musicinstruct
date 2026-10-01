"""Reference baselines for MIDI-Instruct evaluation."""

from __future__ import annotations

import shutil
from pathlib import Path

from .midi import load_midi
from .schema import BenchmarkItem, Plan, Prediction
from .transforms import (
    mute_tracks,
    non_drum_tracks,
    program_change,
    tempo_scale,
    transpose,
    velocity_scale,
)

BASELINE_NAMES = ("oracle", "copy_source", "wrong_transform", "no_output")


def _cache_path(cache_dir: Path, item_id: str, baseline: str) -> Path:
    safe_id = item_id.replace("/", "_")
    return cache_dir / baseline / f"{safe_id}.mid"


def _wrong_semitones(expected: int) -> int:
    return expected + 2 if expected >= 0 else expected - 2


def _wrong_plan(item: BenchmarkItem) -> Plan | None:
    if item.plan is None:
        return None

    op = item.plan.op
    params = dict(item.plan.params)
    source = item.midi_in

    if op == "transpose":
        params["semitones"] = _wrong_semitones(int(params["semitones"]))
        if "tracks" not in params:
            params["tracks"] = non_drum_tracks(source)
    elif op == "tempo_scale":
        factor = float(params["factor"])
        params["factor"] = factor * 1.35 if factor >= 1.0 else factor * 0.65
    elif op == "velocity_scale":
        factor = float(params["factor"])
        params["factor"] = min(1.5, factor * 1.4) if factor <= 1.0 else max(0.5, factor * 0.6)
        if "tracks" not in params:
            params["tracks"] = [0]
    elif op == "mute_tracks":
        tracks = list(params["tracks"])
        instruments = len(load_midi(source).instruments)
        wrong_track = (tracks[0] + 1) % instruments if instruments else 0
        if wrong_track in tracks and instruments > 1:
            wrong_track = (wrong_track + 1) % instruments
        params["tracks"] = [wrong_track]
    elif op == "program_change":
        program = int(params["program"])
        params["program"] = 25 if program != 25 else 40
    else:
        return None
    return Plan(op=op, params=params)


def _render_wrong_transform(item: BenchmarkItem, destination: Path) -> Path:
    wrong_plan = _wrong_plan(item)
    if wrong_plan is None:
        shutil.copy2(item.midi_in, destination)
        return destination

    op = wrong_plan.op
    params = wrong_plan.params
    source = item.midi_in

    if op == "transpose":
        transpose(
            source,
            destination,
            semitones=int(params["semitones"]),
            tracks=list(params.get("tracks", non_drum_tracks(source))),
        )
    elif op == "tempo_scale":
        tempo_scale(source, destination, factor=float(params["factor"]))
    elif op == "velocity_scale":
        velocity_scale(
            source,
            destination,
            factor=float(params["factor"]),
            tracks=list(params.get("tracks", [0])),
        )
    elif op == "mute_tracks":
        mute_tracks(source, destination, tracks=[int(t) for t in params["tracks"]])
    elif op == "program_change":
        program_change(
            source,
            destination,
            track=int(params["track"]),
            program=int(params["program"]),
        )
    else:
        shutil.copy2(source, destination)
    return destination


def predict_baseline(item: BenchmarkItem, baseline: str, cache_dir: Path) -> Prediction:
    if baseline not in BASELINE_NAMES:
        raise ValueError(f"unknown baseline {baseline!r}; expected one of {BASELINE_NAMES}")

    if baseline == "no_output":
        return Prediction(item_id=item.item_id, midi_path=None, plan=None, metadata={"baseline": baseline})

    if baseline == "oracle":
        if not item.gold_midi:
            raise ValueError(f"{item.item_id}: oracle baseline requires gold_midi")
        return Prediction(
            item_id=item.item_id,
            midi_path=item.gold_midi,
            plan=item.plan,
            metadata={"baseline": baseline},
        )

    if baseline == "copy_source":
        return Prediction(
            item_id=item.item_id,
            midi_path=item.midi_in,
            plan=None,
            metadata={"baseline": baseline},
        )

    destination = _cache_path(cache_dir, item.item_id, baseline)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _render_wrong_transform(item, destination)
    return Prediction(
        item_id=item.item_id,
        midi_path=str(destination),
        plan=_wrong_plan(item),
        metadata={"baseline": baseline},
    )


def run_baselines(
    items: list[BenchmarkItem],
    cache_dir: str | Path,
    baselines: list[str] | None = None,
) -> dict[str, list[Prediction]]:
    cache_dir = Path(cache_dir)
    names = list(baselines or BASELINE_NAMES)
    out: dict[str, list[Prediction]] = {name: [] for name in names}
    for item in items:
        for name in names:
            out[name].append(predict_baseline(item, name, cache_dir))
    return out
