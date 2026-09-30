"""MIDI loading, validation, and note-level comparison helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pretty_midi

from .schema import EditMask

TIME_EPS = 1e-3
PITCH_EPS = 0.5


@dataclass(frozen=True)
class NoteEvent:
    track: int
    pitch: int
    start: float
    end: float
    velocity: int
    is_drum: bool
    program: int

    @property
    def duration(self) -> float:
        return self.end - self.start


def load_midi(path: str | Path) -> pretty_midi.PrettyMIDI:
    return pretty_midi.PrettyMIDI(str(path))


def validate_midi(path: str | Path) -> tuple[bool, str]:
    try:
        midi = load_midi(path)
    except Exception as exc:  # noqa: BLE001 — validity check must not raise
        return False, str(exc)
    if not midi.instruments:
        return False, "no instruments"
    for instrument in midi.instruments:
        for note in instrument.notes:
            if not 0 <= note.pitch <= 127:
                return False, f"pitch out of range: {note.pitch}"
            if note.end <= note.start:
                return False, "non-positive note duration"
            if not 0 <= note.velocity <= 127:
                return False, f"velocity out of range: {note.velocity}"
    return True, "ok"


def extract_notes(midi: pretty_midi.PrettyMIDI) -> list[NoteEvent]:
    events: list[NoteEvent] = []
    for track_idx, instrument in enumerate(midi.instruments):
        for note in instrument.notes:
            events.append(
                NoteEvent(
                    track=track_idx,
                    pitch=note.pitch,
                    start=note.start,
                    end=note.end,
                    velocity=note.velocity,
                    is_drum=instrument.is_drum,
                    program=instrument.program,
                )
            )
    events.sort(key=lambda n: (n.track, n.start, n.pitch, n.velocity))
    return events


def estimate_bpm(midi: pretty_midi.PrettyMIDI) -> float:
    if midi.get_tempo_changes()[1].size:
        return float(np.mean(midi.get_tempo_changes()[1]))
    return 120.0


def note_in_mask(note: NoteEvent, mask: EditMask) -> bool:
    if note.is_drum and not mask.include_drums:
        return False
    if mask.tracks is not None and note.track not in mask.tracks:
        return False
    if mask.start_time is not None and note.start < mask.start_time - TIME_EPS:
        return False
    if mask.end_time is not None and note.start >= mask.end_time + TIME_EPS:
        return False
    return True


def partition_notes(notes: list[NoteEvent], mask: EditMask) -> tuple[list[NoteEvent], list[NoteEvent]]:
    inside = [n for n in notes if note_in_mask(n, mask)]
    outside = [n for n in notes if not note_in_mask(n, mask)]
    return inside, outside


def pitch_histogram(notes: list[NoteEvent]) -> np.ndarray:
    hist = np.zeros(128, dtype=np.float64)
    for note in notes:
        hist[note.pitch] += 1.0
    total = hist.sum()
    if total <= 0:
        return hist
    return hist / total


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def ioi_sequence(notes: list[NoteEvent]) -> np.ndarray:
    if len(notes) < 2:
        return np.array([], dtype=np.float64)
    starts = sorted(n.start for n in notes)
    return np.diff(starts)


def note_signature(note: NoteEvent, time_quant: float = 0.01) -> tuple:
    return (
        note.track,
        note.pitch,
        round(note.start / time_quant),
        round(note.end / time_quant),
        note.velocity,
    )


def note_set(notes: list[NoteEvent]) -> set[tuple]:
    return {note_signature(n) for n in notes}


def note_f1(reference: list[NoteEvent], hypothesis: list[NoteEvent]) -> float:
    ref = note_set(reference)
    hyp = note_set(hypothesis)
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0
    tp = len(ref & hyp)
    precision = tp / len(hyp)
    recall = tp / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def notes_unchanged(reference: list[NoteEvent], hypothesis: list[NoteEvent]) -> float:
    return 1.0 if note_set(reference) == note_set(hypothesis) else 0.0


def over_edit_rate(
    source: list[NoteEvent],
    hypothesis: list[NoteEvent],
    mask: EditMask,
) -> float:
    _, preserve_src = partition_notes(source, mask)
    _, preserve_hyp = partition_notes(hypothesis, mask)
    if not preserve_src:
        return 0.0 if note_set(preserve_hyp) == set() else 1.0
    changed = len(note_set(preserve_src) ^ note_set(preserve_hyp))
    return changed / max(len(note_set(preserve_src)), 1)


def midi_summary(path: str | Path) -> dict:
    midi = load_midi(path)
    notes = extract_notes(midi)
    return {
        "path": str(path),
        "valid": validate_midi(path)[0],
        "tracks": len(midi.instruments),
        "notes": len(notes),
        "duration_sec": float(midi.get_end_time()),
        "estimated_bpm": estimate_bpm(midi),
        "programs": [inst.program for inst in midi.instruments],
    }
