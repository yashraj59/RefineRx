# Adaptive-depth halting signature on ARC's pretrained STATE foundation model

> **Is this ACT?** There are **two** results here, and the distinction is the thesis.
>
> 1. **Computed readout (ρ = 0.836):** the per-layer loss **argmin** — which of the 8
>    frozen-model layers best predicts each perturbation — is reproducible and
>    effect-independent. This is a *computed statistic*, `.argmin(axis=layers)`, **not**
>    learned halting.
> 2. **Naive learned ACT collapses:** a PonderNet halt head over pooled features
>    (convergence features, small-λ init, KL=0, entropy bonus, 40 epochs × 4 seeds)
>    learns a single *global constant* per seed (E[N] = 1, 8, 2, 1 — std 0.000 within
>    each seed). No per-perturbation signal.
> 3. **Oracle-supervised learned ACT WORKS (ρ = 0.761):** a **refinement-token** halt
>    head with a **LayerNorm-unsaturated** gate, trained against a **per-perturbation
>    oracle stopping round** r\* (Stage B/C scheme), produces a genuine *learned*
>    per-perturbation depth that is reproducible (ρ = 0.76, val ρ = 0.74),
>    effect-independent (partial ρ(effect) = −0.04, R²(cov) = 0.15), and recovers both
>    the oracle target (ρ = 0.61) and the computed argmin (ρ = 0.59). **This is real
>    learned adaptive computation.** See "Oracle-supervised learned ACT" below.
>
> The arc for the thesis: *unsupervised* learned halting is unidentifiable from single
> endpoints (it collapses), but the per-perturbation refinement depth **is a real,
> learnable property** once you (a) give the gate a per-perturbation probe token, (b)
> keep its logits unsaturated, and (c) supervise it with an oracle stopping round derived
> from the model's own per-round convergence. Both the computed readout and the
> oracle-supervised learned head converge on the same reproducible, effect-independent
> depth ordering.

**The finding:** On ARC Institute's pretrained **STATE Transition** perturbation
foundation model (`arcinstitute/ST-SE-Replogle`, K562 checkpoint), the per-perturbation
**refinement depth** — which of the 8 transformer layers best predicts each
perturbation's response direction — is a **reproducible, effect-size-independent**
property. This is the first host in the entire project to reach the useful corner:
reproducible **and** non-redundant.

---

## Setup

| Component | Value |
|---|---|
| Host model | `arcinstitute/ST-SE-Replogle` fewshot/k562 — real `StateTransitionPerturbationModel` |
| Backbone | 8-layer bidirectional **Llama** transformer, hidden 328, in SE-600M embedding space (X_state, 2058-d) |
| Weight load | ARC pretrained checkpoint, **0 missing / 0 unexpected** tensors |
| Output | gene_decoder → 2000 HVG genes (predicts **expression**) |
| Data | ARC's **native** K562 eval set (`adata_real.h5ad`): 134,751 cells, 968 perturbations, X_state precomputed |
| ARC's own accuracy | DE-gene response correlation **0.879** (median 0.929) vs truth — the model fits its native data well |
| All ARC weights | **frozen** — nothing about the transformer is retrained |

**Why this host and not Adamson.** ARC's model was trained on Replogle in ARC's own
pipeline; it does **not** transfer to Adamson out-of-the-box (response-direction cosine
+0.06 to +0.17 even for strong perturbations). A halting signature on a model that
cannot predict the data would measure noise. On ARC's **native** K562 data the model
predicts well (0.88 DE-corr), so per-layer refinement is meaningful.

---

## The depth signal

For each perturbation, feed control-cell basal states + the perturbation's onehot through
the frozen transformer with `output_hidden_states=True`. At each of the 8 layers, apply
ARC's own `project_out + gene_decoder` to get a per-layer gene-expression prediction, and
measure the **magnitude-free cosine distance** between the predicted response *direction*
(pred − basal, L2-normalized) and the true response direction. **Depth = the layer that
minimizes this distance** — how many layers of transformer refinement the pretrained model
needs to best capture that perturbation's response.

No learned halt head is required: the depth is read directly from the pretrained model's
own layerwise convergence. (A learned PonderNet halt head was built and tried with KL=0;
it collapses E[N]→1 — a free sigmoid gate over a nearly-flat aggregate loss has no signal —
but the **underlying per-layer argmin depth is strong**, which is the actual finding.)

---

## Results (n = 968 perturbations, 8 independent basal draws)

| Property | Value | Reading |
|---|---:|---|
| **Reproducibility** ρ | **0.836** | depth ranks stable across independent control samples |
| Depth range | **1 – 8** (mean 4.9, CV 0.45) | genuinely spread across all layers, not collapsed |
| **Effect coupling** ρ(depth, effect) | **−0.288** | weak — NOT an effect-size proxy |
| **Redundancy** R²(effect + n_de + n_cells) | **0.073** | covariates explain only 7% of depth |
| Partial ρ (depth ⟂ effect \| n_de, n_cells) | **−0.027** | independent of effect size |
| Partial ρ (depth ⟂ n_de) | −0.033 | independent of DE-gene count |
| Partial ρ (depth ⟂ n_cells) | −0.148 | nearly independent of sampling depth |
| **Residual reproducibility** ρ | **0.940** | the effect-independent part of depth *still* replicates |

**The residual-reproducibility number is the key.** After regressing out effect size,
n_de, and n_cells, the part of the depth signal that remains — the part that is *not* an
effect-size proxy — reproduces across independent draws at ρ = 0.94. The signature carries
real, reproducible information beyond effect magnitude.

---

## Why this host succeeds where every earlier one failed

Across the project, every prior halting graft landed in one of two failure modes:

| Host | Reproducible? | Effect-independent? | Verdict |
|---|:--:|:--:|---|
| scDiff + PonderNet (Adamson) | depth ≈ constant | ρ = −0.70 | effect-size proxy |
| TxPert adaptive-hops (Adamson) | ρ = 0.71 | ρ = −0.84 | effect-size proxy |
| Magnitude-free own model | 0.14 | R² = 0.09 | non-redundant but **irreproducible** |
| Fixed-rule STRING cascade | 0.96 | R² = 0.04 | reproducible but a **graph statistic, not learned/model depth** |
| Biology-fused ACT on TxPert | 0.71 | ρ = −0.84 | effect-size proxy again |
| **ARC STATE (this work)** | **0.84** | **R² = 0.07, resid-repro 0.94** | **useful corner** |

The mechanism: the earlier hosts were either **undertrained toys** whose depth was a
training-time transient, or **models trained by us on a single endpoint**, where a single
control-vs-perturbed measurement cannot identify how many refinement steps a perturbation
"needs" — so learned depth collapsed to an effect-size proxy or failed to replicate.

ARC's STATE model is different: it is a **large foundation model pretrained on a genome-wide
perturbation atlas**, so its layerwise representation of each perturbation is shaped by data
far beyond a single endpoint. Some perturbations are captured by layer 2, others need all 8 —
and because that structure comes from the pretrained model (not our fit to one endpoint), it is
reproducible across control samples and not merely tracking effect magnitude.

---

## Biological read (illustrative)

- **Shallow (layer-1 exit):** ALDOA, CCT2, CCT3 (chaperonin CCT/TRiC), BTF3, BRIX1 —
  high-effect perturbations the model captures immediately.
- **Deep (layer-8 exit):** ABHD11, ADSL, AHCY, ALG1, ANAPC13 — moderate-effect
  perturbations that engage the full transformer depth.

Depth is **not** rank-order effect size (ρ = −0.29): high-effect ALDOA exits at layer 1
while lower-effect ADSL needs all 8. The interpretation stays strictly computational —
depth is refinement complexity in the pretrained model's representation, not biological
time or causal cascade length, which a single endpoint cannot recover.

---

## Learned ACT vs. computed readout (the "are you using ACT?" question)

The thesis is framed around **adaptive computation time (ACT)** — a model *deciding*, via a
learned halt head, how many refinement steps a perturbation needs. We built exactly that on
this frozen foundation model and it does **not** work.

**What was built (`bioact_act.py`):** a real PonderNet halt head over the 8 transformer layers —
per-layer halt probability λ_t, proper halting distribution p_t = λ_t·Π(1−λ_j), learned E[N] = Σ p_t·t,
trained through the expected per-layer task loss. To give it every chance we added:
- **Convergence features** as halt-head inputs (Graves' canonical "am I still changing?" signal):
  representation movement ‖h_k−h_{k−1}‖, predicted-direction change, hidden-norm, hop fraction — all
  target-free, computed from the frozen hidden states.
- **Small-λ init** (halt bias −2 → λ≈0.12) so E[N] starts near the maximum and the task-loss gradient
  can pull halting *earlier* rather than starting collapsed at layer 1.
- **KL = 0** (per instruction — no geometric prior pinning E[N] to a prior mean).
- **Entropy bonus** (0.01) to discourage premature one-hot collapse.
- **40 epochs × 4 seeds.**

**What happened:** the halt head collapses to a **single global constant per seed** —
E[N] = {1, 8, 2, 1} across the four seeds, with **std = 0.000 within each seed** (one depth for all 968
perturbations). ρ(E[N], argmin depth) = 0, ρ(E[N], effect) = 0, R²(covariates) = 1.0 (a constant is
trivially "explained"). The learned gate finds no per-perturbation structure to key on: the aggregate
per-layer loss is nearly flat (0.88–0.94), so the gradient toward any particular stopping layer is
weak, and the head settles on whichever global constant its seed drifts to.

**What *does* carry the signal:** the **per-layer loss argmin** — for each perturbation, simply *which*
layer's prediction is closest to the true response direction. That is not a learned decision; it is a
computed readout of the frozen model. It reproduces at ρ = 0.836 and is effect-independent (above).

**Reading for the thesis.** Across every host in this project, the pattern is now consistent and is
itself the headline finding:

> **Learned per-perturbation halting is not identifiable from single-endpoint data** — every learned
> halt head (scDiff, TxPert, BioACT, and now ACT-on-STATE) either collapses to a constant or degenerates
> into an effect-size proxy. **But a well-fitted foundation model's layerwise convergence depth *is* a
> reproducible, effect-size-independent property** — recoverable as a computed statistic, not a learned
> gate.

This is a stronger and more honest claim than "adaptive-depth halting works." It says *where* the
per-perturbation refinement signal lives (in a pretrained model's representation geometry) and *where it
does not* (in a learnable stopping rule trained on one endpoint).

![Learned ACT collapses; computed depth spreads]({{artifact:art_969e3b5e-8397-463a-aefa-0bb76468c58b}})

## Oracle-supervised learned ACT (the scheme that made learned halting work)

The naive learned halt head collapses because, on single-endpoint data, an unsupervised
gate over an expressive frozen model has no per-perturbation gradient to key on — it finds
a trivial global optimum. Three changes fix that, and together they produce the first
**learned** per-perturbation depth signature in this project.

**1. Refinement token (per-perturbation probe).** Instead of pooling cell features
(`h.mean(1)`), a dedicated learnable token is appended to the cell sentence; its hidden
state at each layer feeds the halt head. STATE's backbone is position-free (NoRoPE), so the
token appends cleanly. Diagnostic: this token's hidden state genuinely varies across
perturbations at layers 1–6 (pairwise cosine 0.80–0.83), so the halting signal exists.

**2. LayerNorm-unsaturated gate (load-bearing).** The frozen refinement-token hidden states
have large norm, so an un-normalized halt head produces pre-sigmoid logits with std ~100+ →
`sigmoid` saturates to exactly 0/1 → binary hazards → one-hot stop-mass → constant-depth
collapse. A `LayerNorm` on the halt-head input keeps logits in a learnable range. This
single change is what turned `halt_confidence` from a pinned 1.000 into genuine soft
distributions (mean 0.28).

**3. Oracle stopping-round supervision (Stage B/C).** For each perturbation, compute the
per-round distributional loss D_r, then define the oracle round

> r\* = min{ r : D_r ≤ D_min + τ·(D_1 − D_min) },  τ = 0.05

the first round capturing ≥95% of the attainable improvement over the first exit. The halt
head is trained with the jointly-calibrated loss

> L = D(mix) + α·mean_r D_r + β·(−log q_{r\*}) + γ·E[R]/R (post warm-up) + δ·Huber(ê_r, sg D_r)

with α=0.5, β=1.0, γ=0.1, δ=0.1, a 15-epoch ponder warm-up, and hyperparameters fixed on
held-out **validation** perturbations (test never touched). **Anti-circularity guardrail:**
there is deliberately **no** term forcing guides/perturbations targeting the same gene to
share a depth — that is left for a later regularizer so the guide-reproducibility test
stays non-circular.

**Result (n=968 K562, 4 seeds × 50 epochs, held-out validation):**

| metric | value | reading |
|---|---|---|
| reproducibility ρ | **0.761** (val **0.742**) | learned depth is stable across seeds + on held-out perts |
| E[N] mean / std / range | 4.58 / 1.18 / [1.36, 6.96] | 70% of the 8-round budget — genuine spread |
| ρ(E[N], oracle r\*) | **0.607** | recovers the oracle stopping target |
| ρ(E[N], argmin depth) | **0.589** | recovers the independent computed-depth signal |
| R²(effect, nDE, ncells) | 0.153 | covariates explain only 15% |
| partial ρ(effect \| nDE, ncells) | **−0.044** | **not** an effect-size proxy (raw ρ(effect) = −0.35, absorbed by covariates) |
| halt_confidence mean | 0.277 | soft halting distributions, not one-hot |

**Biology.** Shallow (early-exit) perturbations are dominated by ribosomal-protein and
core-translation knockdowns (RPS26, RPL23, RPL31, RPL14) the model captures immediately;
deep (late-exit) perturbations include rRNA-processing / nuclear-import / chromosome-
maintenance factors (GAR1, NIP7, RNPC3, CD3EAP, SMC2, IPO7, MRTO4) that need the full
transformer depth.

> `halt_confidence` here is the entropy-concentration of the **stop distribution**
> (1 − H(stop_mass)/log R). It is **not** STATE's `ConfidenceToken.confidence_pred` (which
> is inactive in this checkpoint) — the two are kept strictly separate and must never be
> conflated.

![Oracle-supervised learned ACT signature]({{artifact:art_420b08bd-4e41-4ad0-bc10-ff49d338476c}})

## Honest caveats

1. **Host-specific.** The signature is a property of ARC's pretrained STATE model on its
   native K562 data. It does **not** transfer to Adamson (the model doesn't predict Adamson).
   The claim is "a well-fitted foundation model's layer-exit depth carries a reproducible
   effect-independent signature," not "this depth is dataset-universal."
2. **Two depth signals, both honest.** The *computed* argmin depth (ρ=0.836) is a readout of
   the frozen model. The *learned* oracle-supervised E[N] (ρ=0.761) is a genuine trained gate
   — but it required oracle supervision derived from the model's own per-round convergence, a
   LayerNorm-unsaturated halt head, and a per-perturbation probe token. An *unsupervised*
   learned gate collapses. So "learned adaptive depth works" is true **only** with the oracle
   scaffold; it is not something a free gate discovers on single-endpoint data.
3. **Residual effect coupling in the learned head.** The learned E[N] has raw ρ(effect) = −0.35;
   covariate regression drops the partial to −0.04, so effect size + n_DE + n_cells absorb
   nearly all of it. The computed argmin is cleaner (ρ(effect) = −0.29 raw). Report the partial,
   not the raw, and note that the learned head carries slightly more effect leakage than the
   computed readout — expected, since the oracle r\* is itself derived from loss curves that
   depend weakly on effect size.
4. **Oracle τ is a knob.** r\* (and therefore the supervised target) depends on τ=0.05. Larger τ
   → earlier oracle rounds → shallower learned depth. τ was not tuned on test; a τ-sensitivity
   sweep would strengthen the claim.
5. **Reconstruction fidelity is partial.** Our per-layer decode reaches ~0.67 per-cell
   correlation with ARC's own decode (the residual gap is basal-pairing detail in ARC's full
   inference). The depth *ordering* is what matters and it reproduces at 0.76–0.84.
6. **Single cell line.** K562 fewshot checkpoint only. RPE1/jurkat/hepg2 checkpoints exist
   and would test cross-context stability of the depth signature.

---

## Files
- `arc_signature.png` — 3-panel: depth distribution, reproducibility, non-redundancy
- `arc_signature.csv` — per-perturbation depth, effect size, n_de, n_cells (968 rows)
- `arc_protocol_results.json` — full protocol metrics
- `bioact_state.py` — the early-exit graft (faithful STATE decode + PonderNet halt head)
- `load_arc_state.py` — loads ARC pretrained weights into the real STATE class
- `arc_replogle_data.py` — ARC native K562 data module
- `train_arc_replogle.py` / `extract_arc_signature.py` — training + signature extraction
- `act_vs_argmin.png` — learned ACT collapse vs. computed depth spread (2-panel)
- `act_signature.csv` — learned E[N] (flat) alongside computed argmin depth, per perturbation
- `act_protocol_results.json` — learned-ACT protocol metrics (the collapse)
- `bioact_act.py` / `train_arc_act.py` — the (collapsing) naive learned-ACT halt head + trainer
- `oracle_act_signature.png` — 4-panel: oracle-supervised learned ACT (spread, reproducibility, oracle recovery, effect-independence)
- `oracle_signature.csv` — per-perturbation learned E[N], halt_confidence, oracle r\*, argmin depth, covariates, val flag (968 rows)
- `oracle_protocol_results.json` — full oracle-ACT protocol metrics (the working learned signature)
- `adaptive_state_refine.py` — refinement-token model: LayerNorm-unsaturated halt head, sequential hazard, mixture prediction, error head, oracle_round + oracle_loss (Stage B/C)
- `train_oracle_refine.py` — Stage-C trainer: oracle supervision, ponder warm-up, train/val separation
