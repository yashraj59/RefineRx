# Halting experiment setup (working) — `best_model_ram_halt`

A sandboxed copy of `best_model_ram` for studying the ACT/PonderNet **halting behavior** of the
`HybridMemoryRAMLite` core, plus the fixes that make it run at scale. The main `best_model_ram`
directory is untouched. All changes here are **opt-in via env flags** (default behavior = original),
so existing checkpoints/probes are unaffected unless a flag is set.

---

## 1. How halting actually works (the mechanism)

The core is **Graves ACT** (no separate halting loss):
- Each round `t`: `p_t = halt_head(feat_norm(_build_feats(...)))`  (`core.py`)
- `w_t = p_t · Π_{i<t}(1 − p_i)`  → halting distribution; `y_mix = Σ_t w_t · state_t`
- Halt when `remaining ≤ epsilon` after `min_rounds`.
- The halting is trained **implicitly** through `y_mix` against the **full-magnitude** `gaussian_nll` —
  there is no scale-free halting objective. Effective depth `E[N] = Σ_t w_t · t`.

**The magnitude channel (important):** `_build_feats` feeds the halt head four raw scalars —
`l2` and `mean_abs` of the pathway effect and of `(current − x0)` (= **response magnitude**). So the
halt head *can* see effect size. `feat_norm` (LayerNorm) partially de-scales it; `RAMPERT_SCALEINV_HALT`
(below) removes it entirely.

---

## 2. Flags (env vars — all opt-in, default = original behavior)

| env var | default | effect |
|---|---|---|
| `RAMPERT_SCALEINV_HALT=1` | off | **Scale-invariant halt features**: `_stats` returns unit-direction shape (`(x/‖x‖).abs().mean`, `.std`) instead of `l2`/`mean_abs`. The halt head becomes **blind to response magnitude**. Same head_input_dim, so checkpoints load — **but you MUST set the same flag when probing a checkpoint trained with it**, else the halt features mismatch. |
| `RAMPERT_MAX_ROUNDS=N` | 4 | ACT depth ceiling (`max_rounds`). **4 → adaptive** per-pert depth; **8 → collapses** to a uniform ~2 (halt head spreads instead of specializing). |
| `RAMPERT_PONDER_KL=λ` | 0.0 | Adds a **PonderNet geometric-prior KL** on the halting distribution to the loss: `λ·KL(w ‖ Geom(1/μ))`. `λ=0.02` weak, `0.1` strong. Uses grad-carrying `det["weights_soft"]`. |
| `RAMPERT_PONDER_MEAN=μ` | 3.0 | Target mean depth for the geometric prior (only used when `RAMPERT_PONDER_KL>0`). |
| `RAMPERT_MIN_CELLS=N` | 50 | Min cells/pert for the pseudobulk **prior bank**. **Set to 2 for pseudobulk-input data** (guide/donor profiles have ~7 rows/pert; default 50 drops everything → "empty prior bank"). |
| `RAMPERT_CE_PROFILE=pds` | pds | `cell_eval` profile in `rescore_renorm.py`. `pds` = fast (discrimination only); `vcc` = full de/PDS/MAE. |

---

## 3. Code fixes that make it run (in this copy only)

| file | fix | why |
|---|---|---|
| `utils.py::pds_from_l1` | **cap to 64 perts + `scipy.cdist`** instead of `\|d_pred[:,None,:] − d_true[None,:,:]\|.sum(2)` | the `[N,N,G]` broadcast OOMs on large val splits: `1699² × 18080 × 8 = 417 GB`. cdist is chunked L1; cap bounds it. (final reported PDS still uses the real `cell_eval` package.) |
| `utils.py::group_mean_from_adata` | **vectorized sparse group-by** `G @ X` instead of a Python loop over perts | old loop was `O(perts × cells)` + one sparse fancy-index per pert (8000 × 60k scans) — minutes single-threaded → now seconds. |
| `rampert_model.py` | `--min_cells_per_pert` default from `RAMPERT_MIN_CELLS` | pseudobulk-input support. |
| `core.py::_stats` | `RAMPERT_SCALEINV_HALT`-gated scale-invariant features | magnitude-blind halting. |
| `core.py` details | added grad-carrying `weights_soft` | needed for the PonderNet KL to train the halting. |

---

## 4. Working recipes (commands)

Base env (embeddings generated once via `rampert embeddings`; GO/TF from `assets/`):
```bash
export PYTHONPATH=src
export RAMPERT_ESM2=runs/emb_cd4/esm2_gene_embeddings.tsv
export RAMPERT_SCGPT=runs/emb_cd4/scgpt_gene_embeddings.tsv
```

**Plain ACT-4 (the config that passes non-redundancy):** no extra flags.
```bash
python -m rampert.cli train --support S.h5ad --target T.h5ad --variant pds \
  --heldout hepg2 --embedding-dir runs/emb_cd4 --out runs/act4 --device auto
```

**Scale-invariant (magnitude-blind) halting:** `RAMPERT_SCALEINV_HALT=1` on train AND probe.
```bash
RAMPERT_SCALEINV_HALT=1 python -m rampert.cli train ... --out runs/si
RAMPERT_SCALEINV_HALT=1 python act_probe_one.py runs/si/stage2/best_model.pt "SI"
```

**PonderNet (per-pert geometric prior):** `RAMPERT_PONDER_KL=0.02 RAMPERT_PONDER_MEAN=3`.

**Deeper/shallower ACT ceiling:** `RAMPERT_MAX_ROUNDS=8`.

**Pseudobulk-input target (e.g. genome-scale CD4):** add `RAMPERT_MIN_CELLS=2`.

**ACT-signature probe** (effective depth E[N] per pert; set the same halting flags used in training):
```bash
python act_probe_one.py <ckpt> "<label>"        # single ckpt -> depth mean/CV/deepest
python runs_act_probe.py <model_dir>            # probes stage1/2/3 (skips missing)
```

---

## 5. Findings so far (which halting is "real")

| config | context | depth CV | ρ(depth, effect) | verdict |
|---|---|---|---|---|
| **ACT-4** (max_rounds=4) | ARC_H1 stage-3 | **0.108** | **−0.06** | adaptive + effect-size-independent; GO permutation p=0.001 |
| ACT-8 (max_rounds=8) | ARC_H1 stage-3 | 0.004 | +0.23 | collapsed |
| PonderNet-weak (λ=0.02) | ARC_H1 stage-3 | 0.158 | **+0.41** | most adaptive but **effect-coupled → fails non-redundancy** |
| broad multi-celltype | stage-2 | 0.003 | — | flat (adaptive depth needs **single-context** fine-tuning) |
| **scale-invariant** | stage-2 / T-cell s3 | 0.037 / 0.021 | (magnitude-blind by design) | removing magnitude **increases** spread vs `l2` (0.003), **same PDS** — architectural magnitude-independence |

**Takeaways:**
- Adaptive per-pert depth requires **single-context stage-3** + **enough perturbations** (20 = underpowered; 150 works; genome-scale is the proper test).
- Depth tracks a **regulatory-complexity axis** (Mediator/mito/chromatin regulators deep; direct
  effectors / translation / proximal TCR signaling shallow), effect-size-independent, GO-coherent.
- Scale-invariant halting is the **architectural** proof that demand ≠ magnitude proxy: blind the
  halt head to amplitude and depth still varies (more, even) and stays biologically ordered.
- **Non-adaptive by default** for: multi-context training, `max_rounds=8`, tiny pert counts.
