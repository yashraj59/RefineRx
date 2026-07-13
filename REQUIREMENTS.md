# Requirements & environments

This project used two distinct environments. Neither is pinned to an exact lockfile here; the versions
below are the ones the results were produced under.

## Methods / training environment (STATE, grafts, training)

Used for the STATE oracle-ACT training, the halting grafts, and all model training.

- `torch==2.6.0` (CUDA 12.4 build, `cu124`)
- `arc-state==0.11.2`       # ARC Institute STATE Transition foundation model
- `cell-load==0.10.4`       # ARC cell-load data pipeline
- `lightning==2.6.5`
- `scanpy`
- `anndata`

GPU is required for training (results were produced on NVIDIA L40 / equivalent). The ARC pretrained
checkpoints (`arcinstitute/ST-SE-Replogle`) are fetched separately and are **not** vendored in this
repo (see `.gitignore` — `*.ckpt`).

## Analysis environment (signatures, biology, figures)

Used for signature extraction, cross-line/cross-donor statistics, UMAP/Leiden, drug-target discovery,
and all figures.

- `numpy`
- `pandas`
- `scipy`
- `scanpy`
- `umap-learn`

(plus `matplotlib` / `seaborn` for figures, `anndata` for I/O)

## Notes

- Trained halt-head weights are included as small `.pt` files under
  `04_state_oracle_act/models/`. Large checkpoints (`.ckpt`) and `.h5ad` data are intentionally
  git-ignored — they are re-fetched / re-generated from the code in each stage.
- `arc-state` and `cell-load` are ARC Institute packages; install from their respective distributions.
