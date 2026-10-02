from pathlib import Path

import pytest

from musicinstruct.seeds import (
    SeedFilterCriteria,
    assign_split,
    cluster_id_from_path,
    composition_id_from_cluster,
    filter_midicaps_seeds,
    passes_seed_filter,
    resolve_lakh_midi_path,
)
from musicinstruct.transforms import make_seed_midi


def test_cluster_id_from_lakh_path() -> None:
    path = "lmd_full/0/000abc123def456789012345678901234567890"
    assert cluster_id_from_path(path) == "000abc123def456789012345678901234567890"


def test_composition_id_is_stable_prefix() -> None:
    cluster = "000abc123def456789012345678901234567890"
    assert composition_id_from_cluster(cluster) == "lmd_000abc123def"


def test_assign_split_respects_reserved_test() -> None:
    assert assign_split("cluster_a", reserved_test=True) == "test"


def test_assign_split_is_deterministic() -> None:
    assert assign_split("cluster_b") == assign_split("cluster_b")
    assert assign_split("cluster_b") in {"train", "validation", "test"}


def test_passes_seed_filter_accepts_programmatic_seed(tmp_path: Path) -> None:
    midi = make_seed_midi(tmp_path / "seed.mid", seed_index=0, bars=16)
    criteria = SeedFilterCriteria(min_bars=4)
    ok, reason = passes_seed_filter(midi, criteria)
    assert ok, reason


def test_passes_seed_filter_rejects_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty.mid"
    empty.write_bytes(b"not midi")
    ok, _reason = passes_seed_filter(empty)
    assert not ok


def test_resolve_lakh_midi_path_accepts_lmd_full_root(tmp_path: Path) -> None:
    cluster = "000abc123def456789012345678901234567890"
    midi = make_seed_midi(tmp_path / "seed.mid", seed_index=0, bars=16)
    lmd_full = tmp_path / "lmd_full" / "0"
    lmd_full.mkdir(parents=True)
    target = lmd_full / f"{cluster}.mid"
    target.write_bytes(midi.read_bytes())

    resolved = resolve_lakh_midi_path(tmp_path / "lmd_full", f"lmd_full/0/{cluster}.mid")
    assert resolved == target.resolve()


def test_resolve_lakh_midi_path_accepts_extract_root(tmp_path: Path) -> None:
    cluster = "000abc123def456789012345678901234567890"
    midi = make_seed_midi(tmp_path / "seed.mid", seed_index=1, bars=16)
    target = tmp_path / "lmd_full" / "1" / f"{cluster}.mid"
    target.parent.mkdir(parents=True)
    target.write_bytes(midi.read_bytes())

    resolved = resolve_lakh_midi_path(tmp_path, f"lmd_full/1/{cluster}.mid")
    assert resolved == target.resolve()


def test_filter_midicaps_seeds_deduplicates_clusters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    midi_a = make_seed_midi(tmp_path / "a.mid", seed_index=0, bars=16)
    cluster = cluster_id_from_path(str(midi_a))

    def _fake_rows(*_args, **_kwargs):
        return [
            {
                "path": midi_a,
                "location": f"lmd_full/0/{cluster}.mid",
                "caption": "upbeat pop",
                "test_set": False,
            },
            {
                "path": midi_a,
                "location": f"lmd_full/1/{cluster}.mid",
                "caption": "duplicate cluster",
                "test_set": False,
            },
        ]

    monkeypatch.setattr("musicinstruct.seeds.resolve_midicaps_paths", _fake_rows)
    seeds = filter_midicaps_seeds(
        tmp_path,
        seed_limit=10,
        midicaps_limit=10,
        criteria=SeedFilterCriteria(min_bars=4),
    )
    assert len(seeds) == 1
    assert seeds[0].composition_id == composition_id_from_cluster(cluster)


def test_filter_midicaps_test_set_forces_test_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    midi = make_seed_midi(tmp_path / "held.mid", seed_index=1, bars=16)
    cluster = cluster_id_from_path(str(midi))

    def _fake_rows(*_args, **_kwargs):
        return [
            {
                "path": midi,
                "location": f"lmd_full/9/{cluster}.mid",
                "caption": "held-out",
                "test_set": True,
            }
        ]

    monkeypatch.setattr("musicinstruct.seeds.resolve_midicaps_paths", _fake_rows)
    seeds = filter_midicaps_seeds(
        tmp_path,
        seed_limit=5,
        criteria=SeedFilterCriteria(min_bars=4),
    )
    assert len(seeds) == 1
    assert seeds[0].split == "test"
    assert seeds[0].midicaps_test_set is True
