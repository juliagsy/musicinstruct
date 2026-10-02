from pathlib import Path

from musicinstruct.generate import (
    DEFAULT_TEST_MAX_PER_COMPOSITION,
    REAL_TARGET_ITEMS,
    generate_real_dataset,
    subsample_test_items,
)
from musicinstruct.seeds import SeedRecord
from musicinstruct.transforms import make_seed_midi


def _seed_record(path: Path, composition_id: str, split: str) -> SeedRecord:
    return SeedRecord(
        composition_id=composition_id,
        source_path=path,
        split=split,
        cluster_id=f"cluster_{composition_id}",
    )


def test_subsample_test_items_caps_per_composition(tmp_path: Path) -> None:
    seeds = []
    for index in range(8):
        midi = make_seed_midi(tmp_path / f"seed_{index}.mid", seed_index=index, bars=16)
        split = "test" if index >= 6 else "train"
        seeds.append(_seed_record(midi, f"lmd_test{index:02d}", split))

    out = tmp_path / "full"
    full_items = generate_real_dataset(
        out,
        seeds,
        target=REAL_TARGET_ITEMS,
        force=True,
        test_max_per_composition=None,
    )
    subsampled = subsample_test_items(
        full_items,
        max_per_composition=DEFAULT_TEST_MAX_PER_COMPOSITION,
        seed=0,
    )

    test_items = [item for item in subsampled if item.split == "test"]
    train_items = [item for item in subsampled if item.split == "train"]
    full_test = [item for item in full_items if item.split == "test"]
    full_train = [item for item in full_items if item.split == "train"]

    assert len(train_items) == len(full_train)
    assert len(test_items) < len(full_test)
    assert len({item.composition_id for item in test_items}) == 2
    per_composition = {}
    for item in test_items:
        per_composition[item.composition_id] = per_composition.get(item.composition_id, 0) + 1
    assert all(count <= DEFAULT_TEST_MAX_PER_COMPOSITION for count in per_composition.values())
    op_families = {item.op_family for item in test_items if item.composition_id == "lmd_test06"}
    assert len(op_families) == 5


def test_generate_real_applies_test_subsample(tmp_path: Path) -> None:
    seeds = []
    for index in range(4):
        midi = make_seed_midi(tmp_path / f"seed_{index}.mid", seed_index=index, bars=16)
        split = "test" if index == 3 else "train"
        seeds.append(_seed_record(midi, f"lmd_seed{index:02d}", split))

    out = tmp_path / "real"
    items = generate_real_dataset(
        out,
        seeds,
        target=REAL_TARGET_ITEMS,
        force=True,
        test_max_per_composition=DEFAULT_TEST_MAX_PER_COMPOSITION,
    )
    test_items = [item for item in items if item.split == "test"]
    assert len(test_items) <= DEFAULT_TEST_MAX_PER_COMPOSITION
