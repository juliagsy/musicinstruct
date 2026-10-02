"""Seed discovery and filtering for real-MIDI benchmark generation."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .midi import estimate_bpm, extract_notes, load_midi, validate_midi
from .schema import Split
from .transforms import non_drum_tracks

_LAKH_ID = re.compile(r"([0-9a-fA-F]{24,})")


@dataclass(frozen=True)
class SeedRecord:
    """One composition seed drawn from an external MIDI corpus."""

    composition_id: str
    source_path: Path
    split: Split
    cluster_id: str
    location: str | None = None
    caption: str | None = None
    midicaps_test_set: bool = False


@dataclass(frozen=True)
class SeedFilterCriteria:
    min_tracks: int = 2
    max_tracks: int = 8
    min_bars: float = 8.0
    max_bars: float = 64.0
    min_notes: int = 20
    min_pitched_tracks: int = 1


def cluster_id_from_path(path: str | Path) -> str:
    """Stable cluster key from a Lakh-style path (shared song arrangements)."""
    match = _LAKH_ID.search(str(path))
    if match:
        return match.group(1).lower()
    stem = Path(path).stem.lower()
    return hashlib.sha1(stem.encode()).hexdigest()[:24]


def estimate_bars(path: str | Path) -> float:
    midi = load_midi(path)
    bpm = max(estimate_bpm(midi), 1.0)
    bar_sec = 4.0 * (60.0 / bpm)
    return float(midi.get_end_time()) / bar_sec


def passes_seed_filter(path: str | Path, criteria: SeedFilterCriteria | None = None) -> tuple[bool, str]:
    """Return whether a MIDI file is suitable as a benchmark seed."""
    criteria = criteria or SeedFilterCriteria()
    ok, reason = validate_midi(path)
    if not ok:
        return False, reason

    midi = load_midi(path)
    track_count = len(midi.instruments)
    if track_count < criteria.min_tracks:
        return False, f"too_few_tracks:{track_count}"
    if track_count > criteria.max_tracks:
        return False, f"too_many_tracks:{track_count}"

    notes = extract_notes(midi)
    if len(notes) < criteria.min_notes:
        return False, f"too_few_notes:{len(notes)}"

    pitched = non_drum_tracks(path)
    if len(pitched) < criteria.min_pitched_tracks:
        return False, "no_pitched_tracks"

    bars = estimate_bars(path)
    if bars < criteria.min_bars:
        return False, f"too_short:{bars:.1f}bars"
    if bars > criteria.max_bars:
        return False, f"too_long:{bars:.1f}bars"

    return True, "ok"


def composition_id_from_cluster(cluster_id: str) -> str:
    return f"lmd_{cluster_id[:12]}"


def assign_split(
    cluster_id: str,
    *,
    reserved_test: bool = False,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Split:
    """Assign a composition cluster to train/validation/test without leakage."""
    if reserved_test:
        return "test"
    bucket = int(hashlib.sha1(cluster_id.encode()).hexdigest(), 16) % 10_000
    train_cutoff = int(train_ratio * 10_000)
    val_cutoff = int((train_ratio + val_ratio) * 10_000)
    if bucket < train_cutoff:
        return "train"
    if bucket < val_cutoff:
        return "validation"
    return "test"


def resolve_lakh_midi_path(lakh_root: str | Path, location: str) -> Path | None:
    """Resolve a MidiCaps location against a local Lakh tree.

    ``lakh_root`` may be either the ``lmd_full/`` directory or its parent extract
    root (as in the MidiCaps tarball). ``location`` values typically look like
    ``lmd_full/0/<hash>.mid``.
    """
    root = Path(lakh_root)
    ref = Path(location)
    candidates = [
        root / ref,
        root / ref.name,
        root / "lmd_full" / ref,
    ]
    if location.startswith("lmd_full/"):
        candidates.append(root / location.removeprefix("lmd_full/"))
    if len(ref.parts) >= 2:
        candidates.append(root / ref.parts[-2] / ref.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_midicaps_paths(
    lakh_root: str | Path,
    *,
    limit: int = 5000,
    test_set_only: bool = False,
) -> list[dict]:
    """Load MidiCaps rows and resolve local Lakh MIDI paths."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "MidiCaps loading requires optional dependency: pip install -e '.[stress]'"
        ) from exc

    lakh_root = Path(lakh_root)
    dataset = load_dataset("amaai-lab/MidiCaps", split="train")
    rows: list[dict] = []
    scanned = 0
    for row in dataset:
        scanned += 1
        if test_set_only and not row.get("test_set", False):
            continue
        location = row.get("location")
        if not location:
            continue
        candidate = resolve_lakh_midi_path(lakh_root, str(location))
        if candidate is None:
            continue
        rows.append(
            {
                "path": candidate,
                "location": str(location),
                "caption": row.get("caption") or row.get("text") or row.get("description"),
                "test_set": bool(row.get("test_set", False)),
            }
        )
        if len(rows) >= limit:
            break
    if scanned and not rows and lakh_root.is_dir():
        sample = next(
            (str(row.get("location")) for row in dataset if row.get("location")),
            None,
        )
        probe = resolve_lakh_midi_path(lakh_root, sample) if sample else None
        hint = (
            f" scanned {scanned} MidiCaps rows, 0 paths resolved under {lakh_root}"
            f"; sample location={sample!r} -> {probe}"
        )
        raise RuntimeError(
            "No MidiCaps MIDI files found on disk."
            f"{hint}. Ensure midicaps.tar.gz is extracted and --lakh-root points at"
            " lmd_full/ (or its parent extract directory)."
        )
    return rows


def filter_midicaps_seeds(
    lakh_root: str | Path,
    *,
    seed_limit: int = 500,
    midicaps_limit: int = 5000,
    criteria: SeedFilterCriteria | None = None,
    exclude_test_set_from_train: bool = True,
) -> list[SeedRecord]:
    """Select filtered, de-duplicated MidiCaps/Lakh seeds with composition-level splits."""
    criteria = criteria or SeedFilterCriteria()
    rows = resolve_midicaps_paths(lakh_root, limit=midicaps_limit)

    # Cluster → best candidate path (first row that passes filter per cluster)
    clusters: dict[str, dict] = {}
    for row in rows:
        cluster = cluster_id_from_path(row["location"])
        if cluster in clusters:
            continue
        ok, _reason = passes_seed_filter(row["path"], criteria)
        if not ok:
            continue
        clusters[cluster] = row

    reserved_test_clusters = {
        cluster
        for cluster, row in clusters.items()
        if row.get("test_set") and exclude_test_set_from_train
    }

    seeds: list[SeedRecord] = []
    for cluster, row in sorted(clusters.items()):
        split = assign_split(
            cluster,
            reserved_test=cluster in reserved_test_clusters,
        )
        seeds.append(
            SeedRecord(
                composition_id=composition_id_from_cluster(cluster),
                source_path=row["path"],
                split=split,
                cluster_id=cluster,
                location=row.get("location"),
                caption=row.get("caption"),
                midicaps_test_set=bool(row.get("test_set")),
            )
        )
        if len(seeds) >= seed_limit:
            break

    return seeds
