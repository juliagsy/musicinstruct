"""Deterministic MIDI transforms for unique-gold benchmark items."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pretty_midi

from .midi import load_midi
from .schema import EditMask, Plan, Predicate


@dataclass
class TransformResult:
    gold_path: Path
    plan: Plan
    must_change: list[Predicate]
    must_preserve: list[Predicate]
    edit_mask: EditMask


def _write(midi: pretty_midi.PrettyMIDI, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
    return path


def _clone(source: str | Path) -> pretty_midi.PrettyMIDI:
    return load_midi(source)


def _default_preserve() -> list[Predicate]:
    return [
        Predicate(name="notes_unchanged_outside_mask"),
        Predicate(name="pitch_histogram_unchanged_outside_mask", params={"max_js": 0.01}),
        Predicate(name="track_set_unchanged"),
    ]


def transpose(
    source: str | Path,
    destination: str | Path,
    semitones: int,
    tracks: list[int] | None = None,
) -> TransformResult:
    midi = _clone(source)
    target_tracks = tracks if tracks is not None else [
        i for i, inst in enumerate(midi.instruments) if not inst.is_drum
    ]
    for track_idx in target_tracks:
        instrument = midi.instruments[track_idx]
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            note.pitch = max(0, min(127, note.pitch + semitones))
    gold_path = _write(midi, Path(destination))
    return TransformResult(
        gold_path=gold_path,
        plan=Plan(op="transpose", params={"semitones": semitones, "tracks": target_tracks}),
        must_change=[
            Predicate(
                name="pitch_shifted_by",
                params={"semitones": semitones, "tracks": target_tracks, "tolerance": 0.5},
            )
        ],
        must_preserve=_default_preserve(),
        edit_mask=EditMask(tracks=target_tracks, include_drums=False),
    )


def tempo_scale(source: str | Path, destination: str | Path, factor: float) -> TransformResult:
    midi = _clone(source)
    scale = 1.0 / factor
    for instrument in midi.instruments:
        for note in instrument.notes:
            note.start *= scale
            note.end *= scale
    gold_path = _write(midi, Path(destination))
    return TransformResult(
        gold_path=gold_path,
        plan=Plan(op="tempo_scale", params={"factor": factor}),
        must_change=[Predicate(name="tempo_scaled_by", params={"factor": factor, "tolerance": 0.08})],
        must_preserve=[
            Predicate(name="pitches_unchanged"),
            Predicate(name="track_set_unchanged"),
        ],
        edit_mask=EditMask(include_drums=True),
    )


def velocity_scale(
    source: str | Path,
    destination: str | Path,
    factor: float,
    tracks: list[int] | None = None,
) -> TransformResult:
    midi = _clone(source)
    target_tracks = tracks if tracks is not None else list(range(len(midi.instruments)))
    for track_idx in target_tracks:
        for note in midi.instruments[track_idx].notes:
            note.velocity = max(1, min(127, round(note.velocity * factor)))
    gold_path = _write(midi, Path(destination))
    return TransformResult(
        gold_path=gold_path,
        plan=Plan(op="velocity_scale", params={"factor": factor, "tracks": target_tracks}),
        must_change=[
            Predicate(
                name="velocity_scaled_by",
                params={"factor": factor, "tracks": target_tracks, "tolerance": 0.15},
            )
        ],
        must_preserve=_default_preserve(),
        edit_mask=EditMask(tracks=target_tracks, include_drums=True),
    )


def mute_tracks(source: str | Path, destination: str | Path, tracks: list[int]) -> TransformResult:
    """Silence tracks by zeroing velocity while preserving instrument structure."""
    midi = _clone(source)
    for track_idx in tracks:
        for note in midi.instruments[track_idx].notes:
            note.velocity = 1
    gold_path = _write(midi, Path(destination))
    return TransformResult(
        gold_path=gold_path,
        plan=Plan(op="mute_tracks", params={"tracks": tracks}),
        must_change=[Predicate(name="track_muted", params={"tracks": tracks, "max_velocity": 1})],
        must_preserve=[
            *_default_preserve(),
            Predicate(name="note_structure_unchanged_except_velocity"),
        ],
        edit_mask=EditMask(tracks=tracks, include_drums=True),
    )


def program_change(
    source: str | Path,
    destination: str | Path,
    track: int,
    program: int,
) -> TransformResult:
    midi = _clone(source)
    midi.instruments[track].program = max(0, min(127, program))
    gold_path = _write(midi, Path(destination))
    return TransformResult(
        gold_path=gold_path,
        plan=Plan(op="program_change", params={"track": track, "program": program}),
        must_change=[Predicate(name="program_is", params={"track": track, "program": program})],
        must_preserve=_default_preserve(),
        edit_mask=EditMask(tracks=[track], include_drums=False),
    )


def make_seed_midi(path: str | Path, seed_index: int, bars: int = 4, bpm: float = 120.0) -> Path:
    """Create a small multi-track seed (piano, bass, drums)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    beat = 60.0 / bpm
    bar = 4 * beat
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    piano = pretty_midi.Instrument(program=0, name="piano")
    bass = pretty_midi.Instrument(program=32, name="bass")
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="drums")

    roots = [60, 62, 64, 65]
    root = roots[seed_index % len(roots)]
    pattern = seed_index % 3

    for bar_idx in range(bars):
        base = bar_idx * bar
        for beat_idx in range(4):
            t0 = base + beat_idx * beat
            t1 = t0 + beat * 0.9
            pitch = root + (beat_idx % 2) * 4 + (pattern * 2)
            piano.notes.append(pretty_midi.Note(velocity=80 + seed_index % 20, pitch=pitch, start=t0, end=t1))
            if beat_idx % 2 == 0:
                bass.notes.append(
                    pretty_midi.Note(velocity=70, pitch=max(28, root - 24), start=t0, end=t1)
                )
            if pattern == 0:
                drums.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=t0, end=t0 + 0.05))
                if beat_idx in (1, 3):
                    drums.notes.append(
                        pretty_midi.Note(velocity=90, pitch=38, start=t0, end=t0 + 0.05)
                    )
            elif pattern == 1:
                drums.notes.append(pretty_midi.Note(velocity=85, pitch=42, start=t0, end=t0 + 0.03))
            else:
                if beat_idx == 0:
                    drums.notes.append(pretty_midi.Note(velocity=110, pitch=36, start=t0, end=t0 + 0.05))

    midi.instruments.extend([piano, bass, drums])
    midi.write(str(path))
    return path


def non_drum_tracks(source: str | Path) -> list[int]:
    midi = load_midi(source)
    return [i for i, inst in enumerate(midi.instruments) if not inst.is_drum]


def pitched_tracks(source: str | Path) -> list[int]:
    return non_drum_tracks(source)
