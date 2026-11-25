# Query, key and value subspaces of attention heads: an invariant, chance-calibrated comparison across pre-trained Transformers

Code for the paper *Query, key and value subspaces of attention heads: an invariant, chance-calibrated comparison across pre-trained Transformers*. The notebook `qkv_geometry_pretrained.ipynb` computes every number, table and figure of the paper
from public checkpoints of the Hugging Face Hub.

## Contents

| File | Content |
|---|---|
| `qkv_geometry_pretrained.ipynb` | The notebook (one section per experiment; outputs cleared). |
| `reference_results.json` | The values obtained in the run reported in the paper: for each result id, the producing cell, the value and its LaTeX text. |
| `requirements.txt` | Python packages. |

Notebook sections:

| Section | Content |
|---|---|
| 0–1 | Configuration, results registry and geometry library (weight extraction, principal angles, null calibration, statistics, training helpers) |
| A | Model-free checks: gauge invariance, invariance table, analytical null, RoPE proposition, toy figures, validation of the weight extraction |
| B0 | The twelve reference checkpoints: raw versus calibrated overlaps, ordering tests, coupling, depth profiles; comparison of similarity measures on trained weights |
| B1 | Seed noise floor of the 25 MultiBERTs |
| B2 | Model size at a fixed recipe (24 BERT miniatures, Pythia sizes), threshold or gradual change, distilled models, Pythia training checkpoints |
| B3 | Rotary embeddings: effective bilinear forms per offset and per frequency plane; activation-level analysis |
| B6 | Broader architectures (encoder-decoders, mixture-of-experts, 7-8B GQA models) and family-level tests |
| Z | Results summary, archive and (optional) fill of the paper source |

Each number in the paper is produced by one notebook cell (`R.cell("<section>.<cell>")`), which stores it with
`R.set("<id>", ...)` in `outputs/results.json`; the paper lists the cells it uses (A.1, A.2, A.3, A.4, A.5, A.6, B0.1, B0.2, B0.3, B1.1, B2.1, B2.2, B2.3, B2.4, B3.1, B3.2, B6.1, B6.2).

## Running

**Kaggle** (the setting of the paper). Import the notebook (*File → Import Notebook*), then
*Settings → Accelerator → GPU T4 x2*, *Settings → Internet → On*, and *Add-ons → Secrets*: add a secret `HF_TOKEN`
(a Hugging Face token with access to the gated Llama checkpoints) and tick it for this notebook. Run with
*Save Version → Save & Run All*. The first cells stop at once if the GPU, Internet access or the token is missing.
A session stops cleanly before the 12-hour limit and the last cell prints `STILL TO RUN: ...`; attach the output of
the finished version (*Add input → Your Work → this notebook*) and run again: completed sections are skipped and
interrupted trainings resume from their last checkpoint. The last cell prints `ALL SECTIONS OF THIS NOTEBOOK ARE
COMPLETE` when everything has run.

**Elsewhere.**

```bash
pip install -r requirements.txt
export HF_TOKEN=...        # never write the token in the notebook
jupyter nbconvert --to notebook --execute qkv_geometry_pretrained.ipynb --ExecutePreprocessor.timeout=-1
```

Options (environment variables):

* `QKV_SECTIONS="B1,Z"` runs only some sections; results accumulate in `outputs/results.json`.
* `QKV_SMOKE=1` replaces every checkpoint by a tiny random model of the same architecture and every corpus by random
  tokens. It runs all code paths on a CPU in a few minutes (outputs in `outputs_smoke/`); its numbers are meaningless.
* `QKV_SKIP_TRAIN=1` skips the trainings and fine-tunings; `QKV_GPUS=0` restricts the run to one GPU.
* `QKV_PROJECT_DIR` sets the working directory (default: current directory).

Outputs: `outputs/results.json`, `outputs/figures/*.pdf`, `outputs/tables/*.tex|csv`, `outputs/environment.json`
and the archive `outputs_results.zip`. If the LaTeX source of the paper, with `[[RESULT:id]]` placeholders, is
placed at `paper/qkv_geometry_pretrained.tex`, section Z also writes `paper/qkv_geometry_pretrained_filled.tex`.

## Reproduction

A complete run of this notebook, exactly as published here (Kaggle, Tesla T4 GPU, 2026-11-24), gave the values in
`reference_results.json`. Every number, table and figure of the paper comes from this run.

## Citation

[Reference of the paper, to be added after publication.]
