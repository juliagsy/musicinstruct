# Google Colab — MIDI-Instruct plan executor (T4)

Notebooks to run **Llama plan-then-execute** on the MIDI-Instruct test split using a Colab **T4** GPU (~30–60 min for 50 items vs ~8–10 h on Mac CPU).

## One runtime per notebook

Colab gives **each notebook its own VM**. Pass artifacts through **Google Drive** (`USE_DRIVE=True`, default).

| Drive path | Contents |
|------------|----------|
| `MyDrive/musicinstruct/data/pilot/` | Synthetic pilot dataset (300 items) |
| `MyDrive/musicinstruct/data/v0.2/` | Real MidiCaps/Lakh dataset (`manifest.jsonl`) |
| `MyDrive/musicinstruct/midicaps/` | Extracted MidiCaps `lmd_full/` tree (~1.6 GB) |
| `MyDrive/musicinstruct/runs/<RUN_ID>/plan_executor_llama/` | Predictions, MIDI outputs, reports |
| `MyDrive/musicinstruct/runs/<RUN_ID>/baselines/` | Programmatic baseline predictions + report |

Keep **`RUN_ID`** identical across notebooks (default: `pilot_v1`). Local clone paths mirror Drive (`results/plan_executor_llama/`, `results/baselines/`).

## Prerequisites

1. [Accept the Llama 3.2 license](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) on Hugging Face.
2. Create a [HF access token](https://huggingface.co/settings/tokens) (read access).
3. Colab: **Runtime → Change runtime type → T4 GPU**.

## Notebooks (run in order)

| Notebook | Purpose |
|----------|---------|
| `01_setup_and_data.ipynb` | Clone, install `.[llm,stress]`, generate pilot or **real v0.2**, validate, self-test, **sync to Drive** |
| `02_run_plan_executor.ipynb` | Restore data + prior results from Drive, run Llama on test split, **sync results to Drive** |
| `03_score_and_baselines.ipynb` | Rescore predictions, run programmatic baselines (optional) |

## Quick start

1. Push `musicinstruct` to GitHub; edit repo URL in the config cell if needed.
2. Open **`01_setup_and_data.ipynb`** → run all → confirm Drive sync.
3. Open **`02_run_plan_executor.ipynb`** → set `MAX_ITEMS` (e.g. `5` smoke, `None` for full test) → run all.
4. Optional: **`03_score_and_baselines.ipynb`** for tables without re-running the LLM.

## Tips

- Use **`02`** with `RESUME=True` to continue after disconnect (merges into existing `predictions.jsonl`).
- Prefer **one session** with `MAX_ITEMS=None` on T4 instead of restarting the model per item.
- Stop local CPU loops before starting Colab to avoid duplicate work.
