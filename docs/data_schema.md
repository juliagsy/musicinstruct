# Data schema

Each benchmark record is one JSON object per line (JSONL).

## Required fields

| Field | Type | Description |
|-------|------|-------------|
| `item_id` | string | Unique item identifier |
| `composition_id` | string | Source grouping for split assignment |
| `gold_mode` | `"unique"` \| `"constraint"` | Unique gold MIDI vs constraint-only (v0.1 uses `unique`) |
| `op_family` | string | Operator category (`transpose`, `tempo_scale`, …) |
| `instruction` | string | Natural-language edit request |
| `midi_in` | string | Path to source MIDI (relative to manifest directory) |
| `must_change` | list | Predicates that must pass after editing |
| `must_preserve` | list | Predicates that must pass outside the edit intent |
| `source` | string | Provenance label |
| `license` | string | License tag |
| `split` | string | `train`, `validation`, or `test` |

## Optional fields

| Field | Type | Description |
|-------|------|-------------|
| `gold_midi` | string | Reference output for `unique` gold |
| `plan` | object | `{ "op": "...", "params": {...} }` for MIDI-Reason |
| `edit_mask` | object | Tracks/time region allowed to change |
| `instruction_type` | string | `specific`, `descriptive`, or `stylistic` |
| `metadata` | object | Free-form construction metadata |

## Prediction format

```json
{"item_id": "seed_00_transpose_+3_p0", "midi_path": "outputs/model.mid", "plan": {"op": "transpose", "params": {"semitones": 3}}}
```

## Real seeds (v0.2)

Composition IDs use `lmd_<cluster_prefix>` from Lakh/MidiCaps paths. Items set `source="midicaps"` and `license="CC-BY-4.0"`. Splits are assigned at the composition (cluster) level; MidiCaps `test_set` clusters are held out in test only.

Generate with:

```bash
musicinstruct generate-real --lakh-root /path/to/lmd_full --output-dir data/v0.2
```

## Predicate names (v1)

**Must-change:** `pitch_shifted_by`, `track_muted`, `tempo_scaled_by`, `velocity_scaled_by`, `program_is`

**Must-preserve:** `notes_unchanged_outside_mask`, `pitch_histogram_unchanged_outside_mask`, `ioi_unchanged_outside_mask`, `track_set_unchanged`, `pitches_unchanged`
