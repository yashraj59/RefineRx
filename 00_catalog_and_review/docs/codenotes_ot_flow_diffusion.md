# Code-Grounded Notes — Family: Optimal-Transport / Flow-Matching / Diffusion / World-Model Generative Models

**Track:** OT / flow-matching / diffusion / world-model generative perturbation-response models (~2023–2026).
**Reviewer focus:** These families ALREADY ITERATE at inference (OT map application, flow-matching ODE integration steps, diffusion denoising steps, world-model rollout). The number of inference iterations is a natural **refinement-round analog** for the user's ACT (Adaptive Computation Time) halting thesis. For each model I locate the iterative core in code, name the file/class/function, and reason about grafting a learned-halting/ponder loop on top.

**Verification discipline:** every GitHub link hit via authenticated REST API (size/pushed_at/stars recorded). Code read via shallow blobless clone or raw.githubusercontent.com. Where no repo exists, I say NOT_FOUND and base claims on the paper (fetched PDF).

---

## 1. CellOT (ANCHOR) — 2023, Nature Methods
**Repo:** https://github.com/bunnech/cellot — VERIFIED_ACTIVE (176★, 592 KB, pushed 2024-10-31, branch `main`).

### Files read
- `cellot/networks/icnns.py` — `class ICNN(nn.Module)` (Input-Convex Neural Net) + `NonNegativeLinear`.
- `cellot/models/cellot.py` — `load_cellot_model`, `compute_loss_f/g`, `compute_w2_distance`.
- `cellot/train/train.py` — `train_cellot()` min-max loop.
- `cellot/transport.py` — `transport_cellot()` inference.

### Architecture (code-grounded)
Neural optimal transport via a **dual pair of ICNNs** `(f, g)` (Makkuva/Amos ICNN, Brenier theorem). The OT map is the **gradient of the convex potential**: `ICNN.transport(x)` (icnns.py) returns `autograd.grad(self.forward(x), x)` — i.e. `T(x) = ∇g(x)`. `ICNN.forward` is a fixed stack of non-negative-weight layers `z_{l+1}=σ(W_l z_l + A_l x)` guaranteeing convexity (`test_icnn_convexity`). Training (`train_cellot`) alternates `n_inner_iters` g-updates then one f-update, minimizing the dual OT objective; `f.clamp_w()` enforces convexity constraints.

### Perturbation handling / unseen strategy
**One OT map is trained PER perturbation/condition** (see overview fig: `T_k = ∇g_k`, "Trained for each perturbation"). There is **no perturbation featurization** — a held-out *perturbation* cannot be predicted (needs its own trained map). Generalization is over **unseen CONTROL CELLS** of a *seen* condition (transport applies to any new source cell) and unseen donors/patients (lupus IFN-β task). So `unseen_pert_strategy = "none — needs a trained map per condition; generalizes across cells/donors, not across unseen perturbations."`

### Iterative structure (KEY for ACT)
Inference is **single-shot**: `T(x)=∇g(x)` — one gradient evaluation, no loop. The ONLY step-like knob is in `transport.py`: `outputs = (1-dosage)*inputs + dosage*outputs` — a scalar dosage INTERPOLATION along the displacement, not an iterated solver. There is no unrolled T-step trajectory in code.

### ACT graftability — **MEDIUM**
No native loop, but neural-OT has a natural refinement axis: the McCann displacement interpolation `x_t = (1-t)x + t·∇g(x)` can be discretized into T sub-steps, OR one can iterate a *residual* correction. **Insertion point:** wrap `transport_cellot()` (cellot/transport.py) — replace the one-shot `g.transport(x)` with a K-step displacement/residual loop, add a halting MLP reading the per-step W2-residual (`compute_w2_distance`) to decide when the pushforward has converged. Effort MEDIUM (the map is convex/single-shot; you are *inventing* the refinement axis, but the OT geometry supplies a principled one). ACT halting here ≈ "how many displacement sub-steps until the transported cell stabilizes."

---

## 2. CellFlow — 2025 (theislab)
**Repo:** https://github.com/theislab/cellflow — VERIFIED_ACTIVE (149★, 37 MB, pushed 2026-07-10, branch `main`; actively developed).

### Files read
- `src/cellflow/solvers/_otfm.py` — `class OTFlowMatching`, `predict/_predict_jit/_get_predict_fn`, `ClassifierFreeGuidance`.
- `src/cellflow/networks/_velocity_field.py` — `class ConditionalVelocityField(nn.Module)` (`__call__`).
- `src/cellflow/networks/_set_encoders.py` — `class ConditionEncoder` (attention-pooled set encoder).
- `src/cellflow/model/_cellflow.py` — `class CellFlow` (`prepare_data`, `predict`), solver registry (`otfm`, `genot`).

### Architecture (code-grounded)
**Conditional flow matching** (JAX/Flax + `diffrax` + OTT-JAX). A `ConditionalVelocityField` MLP/FiLM/ResNet network `v_θ(t, x_t, condition)` is trained by OT-flow-matching (`OTFlowMatching.step_fn` samples an OT coupling `solver_utils.sample_joint(tmat)` between source/target minibatches, then regresses the velocity along the straight-line interpolant). The condition embedding comes from `ConditionEncoder`: perturbation covariates (drug/gene/dose…) are embedded and **attention-pooled over a SET** (`TokenAttentionPooling` / `SeedAttentionPooling`), so arbitrary combinations of covariates map to one condition vector.

### Perturbation handling / unseen strategy
`prepare_data(perturbation_covariates=..., perturbation_covariate_reps=...)`: each covariate carries a **representation stored in `adata.uns`** (e.g. drug embeddings, gene embeddings). Because conditions are built from these reps and pooled by attention, **unseen perturbations / unseen COMBINATIONS generalize** as long as a representation exists for the new covariate (molecular/gene embedding), plus classifier-free guidance (`ClassifierFreeGuidance`, learned unconditional velocity via `condition_dropout_prob`). `unseen_pert_strategy = "covariate representation embeddings (drug/gene reps in adata.uns) + attention set-pooling → new perturbations/combos generalize; CFG unconditional path."`

### Iterative structure (KEY for ACT)
**Inference solves an ODE** `dx/dt = v_θ(t,x,cond)` from t0=0 → t1=1 via `diffrax.diffeqsolve` inside `_otfm.py::_get_predict_fn.solve_ode`. Defaults (`_predict_jit`): `solver=diffrax.Tsit5()`, adaptive `stepsize_controller=PIDController(rtol=1e-5,atol=1e-5)`, `dt0=None`. **The step count IS an inference knob** — pass `dt0` / a fixed-step solver (Euler) / step controller through `predict(**kwargs)` → `diffeqsolve`. So refinement rounds = number of ODE solver steps; already first-class.

### ACT graftability — **HIGH**
Textbook adaptive-compute target: an integrator already varies step count. **Insertion point:** in `_otfm.py::_get_predict_fn` replace `diffrax.diffeqsolve(...)` with a custom Euler/step loop over `solve_ode`, and add a halting head reading the per-step velocity norm ‖v_θ(t,x_t,cond)‖ (already computed each step) — halt when velocity/displacement drops below a learned threshold (adaptive ODE step ≈ learned ponder). Even simpler: the adaptive `PIDController` already emits a data-dependent step count you can *record per perturbation* with zero code change (a ready-made "refinement rounds" readout). Effort LOW–MEDIUM. This is the strongest ACT host in the family alongside scDiff.

---

## 3. scDiff — 2023/2024 (OmicsML)
**Repo:** https://github.com/OmicsML/scDiff — VERIFIED_ACTIVE (34★, 702 KB, pushed 2024-08-13, branch `master`).

### Files read
- `scdiff/model.py` — `class ScDiff(pl.LightningModule)`, `class DiffusionModel(nn.Module)`, `p_sample`, `p_sample_loop`, `sample`, `register_schedule`.
- `scdiff/utils/diffusion.py` — `make_beta_schedule`, `timestep_embedding`, `MaskedEncoderConditioner`.
- `scdiff/modules/diffusion_model/embedder.py` — `class Embedder`.
- `scdiff/data/gene_pert.py` — `GenePerturbationBase` (GEARS/GO wiring).
- `configs/eval_genepert.yaml` — perturbation-conditioning config.

### Architecture (code-grounded)
**Conditional DDPM** over single-cell expression. `register_schedule` builds a 1000-step (`timesteps=1000`) β-schedule with the full alphas_cumprod/posterior buffers. The denoiser `DiffusionModel` is a **masked-autoencoder Transformer** (`BasicTransformerBlock`, self- + cross-attention): condition signals (cell type, **perturbation**, text) enter as cross-attention context (`forward_decoder(..., context_list, conditions=...)`, `MaskedEncoderConditioner`). Reverse process: `p_sample_loop` (model.py:728) runs `for i in reversed(range(0, t_start+1)): x = p_sample(...)` — the canonical iterative denoising chain `x_T→…→x_0`.

### Perturbation handling / unseen strategy
Gene-perturbation conditioning is GEARS-style (`gene_pert.py` uses `PertData`, `go_essential_all.csv`, `gene2go_all.pkl`, `num_similar_genes_go_graph=20`): a perturbed gene is represented via its **GO-graph neighbors / co-essential genes**, so **unseen perturbation genes borrow embeddings from GO neighbors** → few/zero-shot. `unseen_pert_strategy = "GEARS-style GO-graph / co-essential neighbor genes supply embeddings for unseen perturbation genes (few/zero-shot)."`

### Iterative structure (KEY for ACT)
Reverse diffusion loop in `p_sample_loop`, step count = `t_start` (≤ `num_timesteps=1000`). Config exposes `recon_sample: false → one-step generation`; `true → full multi-step sampling`, and `denoise_t_sample`/`t_sample` control how many steps. So the refinement-round count is explicit and already tunable per run.

### ACT graftability — **HIGH**
Diffusion reverse chain is a canonical unrollable loop. **Insertion point:** wrap the `for i in reversed(...)` loop in `ScDiff.p_sample_loop` (scdiff/model.py) — add a halting head that reads the per-step predicted-x0 change or posterior variance and early-exits the denoising chain (adaptive number of denoising steps ≈ ponder). The model already computes a VLB per step (`return_vlb`), giving a natural convergence signal. Effort LOW–MEDIUM. Per-perturbation halting = "how many denoising steps until the conditioned x0 estimate stabilizes."

---

## 4. X-Cell — 2026, "Diffusion Language Models" (bioRxiv 2026.03.18.712807; Xaira Therapeutics)
**Repo:** https://github.com/xaira-therapeutics/x-cell (canonical `Xaira-Therapeutics/X-Cell`) — VERIFIED_NO_MODEL_CODE (105★, 1138 KB, pushed 2026-03-16, branch `main`). README: "Model weights and inference code coming soon." Package is an **API stub**.

### Files read
- `src/xcell/model.py` — `class XCell` (`from_pretrained`, `predict`).
- `README.md`, `MODEL_CARD.md` (skimmed).

### Architecture (code-grounded, from the stub API + README)
**Masked diffusion language model** operating on **SETS of cells** (not single cells), initialized from scGPT (X-Cell Mini = 55M params). README: multi-modal priors (ESM-2, STRING, GenePT, DepMap, JUMP Cell Painting, scGPT) integrated via **cross-attention**; "iterative inference-time refinement" via a masked diffusion process. Both `XCell.from_pretrained` and `XCell.predict` currently `raise NotImplementedError` — so architecture claims are **paper/README-grounded, not runtime-verified**.

### Perturbation handling / unseen strategy
Predicts CRISPRi-knockdown transcriptional response from control cells; README claims **zero-shot generalization to unseen cell types and perturbations** via the multi-modal biological priors (a perturbation is a gene, represented through ESM-2/STRING/GenePT embeddings). `unseen_pert_strategy = "multi-modal gene priors (ESM-2/STRING/GenePT/DepMap) via cross-attention → zero-shot to unseen genes & cell types (README claim; code not released)."`

### Iterative structure (KEY for ACT)
The stub `predict()` signature exposes **`n_diffusion_steps: int = 4`** — an explicit iterative-refinement step count (masked-diffusion decoding rounds over the cell set). This is the refinement-round analog, but the actual denoising loop is not in the released code.

### ACT graftability — **MEDIUM (paper/stub-only; would be HIGH if code matched the API)**
A masked-diffusion decoder with `n_diffusion_steps` is structurally a per-step loop and would host a halting head naturally. But because `predict` is unimplemented, there is **no code loop to attach to today**. **Insertion point (prospective):** the (unreleased) masked-diffusion decode loop implementing `n_diffusion_steps`; a halting head would read per-step token/expression change. `act_effort = HIGH` until code is released. Flag as promising-but-unverifiable.

---

## 5. AlphaCell — 2026, "World Model … perturbation-induced cellular dynamics" (bioRxiv 2026.03.02.709176)
**Repo:** NOT_FOUND. Authenticated GitHub repo + code search (AlphaCell / world model / flow matching / virtual cell) returned no matching repository; the fetched PDF contains **no code/data-availability URL** (only the CC-BY-NC-ND license watermark). Paper-only.

### Source read
- Full PDF `10.64898_2026.03.02.709176` (abstract + methods extracted via pypdfium2).

### Architecture (paper-grounded)
Generative **"Virtual Cell World Model"** with three parts: (1) **Latent Manifold Rectification** — an encoder processing the *full protein-coding transcriptome* into a differentiable "Virtual Cell Space" (with an ArcFace metric head on a Level-2 semantic embedding to shape the manifold); (2) **Biological Reality Reconstruction** — a large decoder mapping latent states back to genome-wide expression; (3) **Universal State Transition** — **Optimal-Transport Conditional Flow Matching**: a "heavily customized Flow Matching Transformer backbone" with **Shared+Routed Mixture-of-Experts** predicts a velocity field `v(z_t, t, pert)`, where the perturbation is a **learnable lookup embedding**. Inference integrates the velocity field with an **ODE solver (Euler) from t=0→1** in latent space, then decodes.

### Perturbation handling / unseen strategy
Perturbation = learnable lookup embedding fed to the flow transformer; the paper claims **compositional generalization** and **zero-shot prediction in entirely unseen cellular contexts** by abstracting perturbations into "generalized dynamic laws" (context enters the flow field, so a trained perturbation vector field transfers to new contexts). `unseen_pert_strategy = "per-perturbation lookup embedding + context-conditioned universal velocity field → compositional + zero-shot to unseen contexts (paper claim; no code)."`

### Iterative structure (KEY for ACT)
Latent flow-matching ODE integrated by **Euler from t=0→1** — an explicit multi-step integrator; step count = number of Euler steps (world-model "rollout" of cellular dynamics). Refinement-round analog is the Euler step budget.

### ACT graftability — **HIGH in principle (paper-only; no code to attach to)**
Same structure as CellFlow/AlphaCell flow ODE → a step loop that would host a halting head reading velocity-field magnitude. But **NOT_FOUND repo** ⇒ no insertion point in code; `act_effort = HIGH` (would require reimplementation). Record as architecturally ACT-friendly, verification blocked by absent code.

---

## 6. IMPA — 2025, Nature Communications (morphology) (theislab)
**Repo:** https://github.com/theislab/IMPA — VERIFIED_ACTIVE (26★, 739 MB incl. data, pushed 2025-01-10, branch `main`). Read via raw URLs (repo too large to clone under disk budget).

### Files read (raw.githubusercontent.com)
- `IMPA/model.py` — `ResBlk`, `AdaIN`, `AdainResBlk`, `class Generator`, `MappingNetwork(SingleStyle/MultiStyle)`, `StyleEncoder`, `Discriminator`, `build_model`.
- `IMPA/solver.py` — `class IMPAmodule(LightningModule)` (`training_step`, `_compute_d_loss`, `_compute_g_loss`).

### Architecture (code-grounded)
**StarGAN-v2-style image-to-image GAN** on Cell-Painting microscopy. `Generator` is an encoder→decoder CNN where the decoder blocks (`AdainResBlk`) inject a **style vector via AdaIN** (`AdaIN.fc: Linear(style_dim, 2*num_features)` → per-channel scale/shift). The style comes from `MappingNetwork.forward(z, mol=None, y=None)` (noise + perturbation → style) or `StyleEncoder`. Adversarial training in `solver.py` (`_compute_d_loss`/`_compute_g_loss`, `_adv_loss`, R1 reg, style-diversity + cycle losses). `Generator.forward(x, s, basal=False)` is one encode→decode pass (`for block in self.encode … for block in self.decode`).

### Perturbation handling / unseen strategy
The mapping network takes a **`mol`** argument (molecular/compound identity/embedding), i.e. perturbation = drug conditioning that produces the AdaIN style. Generalization to unseen compounds is via the **molecular embedding** feeding the style (multimodal `MappingNetwork`). `unseen_pert_strategy = "molecular/compound embedding → AdaIN style vector; unseen drugs generalize through the mol embedding space."`

### Iterative structure (KEY for ACT)
**None.** `Generator.forward` is a single-shot encoder→decoder pass; the encode/decode `for block in …` loops are a FIXED-DEPTH CNN stack, not an iterable refinement whose step count varies (changing it changes the architecture/param count). GANs generate in one shot. No diffusion/ODE/recurrence.

### ACT graftability — **LOW**
Single feed-forward generator, no natural refinement axis. Could in principle iterate the generator (feed output back as input, image-space fixed-point) but that is not how it is trained and image round-trips degrade. **Insertion point (weak):** none clean; the only unrollable object is the fixed decoder block stack in `Generator.forward` (model.py) whose depth is structural, not a compute-time variable. `act_effort = HIGH`. Include as a LOW-graftability morphological OT-adjacent baseline.

---

## 7. Cell Painting CNN (DeepProfiler) — 2024, Nature Communications (Broad / Carpenter-Singh)
**Repo:** https://github.com/cytomining/DeepProfiler — VERIFIED_ACTIVE (128★, 17 MB, pushed 2026-07-02, branch `master`). Paper: "Learning representations for image-based profiling of perturbations," Nat Commun 2024 (s41467-024-45999-1).

### Files read (raw URLs)
- `deepprofiler/profiling.py` — `_EFFICIENTNET_MODELS` map, `build_model`, `class Profile`.
- `tests/deepprofiler/profiling/test_cell_painting_cnn_v1.py` — architecture/checkpoint spec.

### Architecture (code-grounded)
**EfficientNet CNN feature extractor** (Cell Painting CNN v1 = EfficientNet-B0). `build_model` (profiling.py): `efficientnet.tfkeras.EfficientNetB0` base on a `(h, w, c)` input (c = 5 Cell-Painting channels), `GlobalAveragePooling2D` → **1280-d embedding** → `Dense(num_classes, softmax)` "ClassProb" head. Trained **weakly-supervised** by classifying compound/treatment identity (490 compound classes for v1); the penultimate pooled layer is used as the morphological profile. Weights hosted on Zenodo (`Cell_Painting_CNN_v1.hdf5`).

### Perturbation handling / unseen strategy
This is a **representation learner, not a response predictor** — it maps an image of perturbed cells to an embedding; it does not *predict* the response to an unseen perturbation. New perturbations are simply *embedded* (forward pass on their images) and compared. `unseen_pert_strategy = "n/a — image→embedding profiler; new perturbations are embedded from their images, not predicted from perturbation identity."`

### Iterative structure (KEY for ACT)
**None.** Single forward pass through a fixed EfficientNet (`GlobalAveragePooling2D` → Dense). No recurrence, no diffusion, no solver. Depth is architectural.

### ACT graftability — **LOW**
Pure feed-forward classifier/encoder; no refinement axis. **Insertion point:** none natural. `act_effort = HIGH` (would require replacing the model with an iterative one). Included because it is the canonical "Cell Painting CNN" morphological baseline, but it is out of scope for adaptive-refinement halting.

---

## Lineage / benchmark note

### CINEMA-OT — 2023 (OT lineage; vandijklab)
**Repo:** https://github.com/vandijklab/CINEMA-OT — VERIFIED_ACTIVE (39★, 22 MB, pushed 2025-04-16, branch `main`).
Files read: `cinemaot/cinemaot.py` (`cinemaot_unweighted/weighted`, `synergy`), `cinemaot/sinkhorn_knopp.py`.
**Not a predictive generative model** — it is a **causal confounder-disentangling OT matcher**: FastICA separates confounder vs treatment-associated components (Chatterjee-coefficient threshold), then **entropic OT via Sinkhorn-Knopp** (`SinkhornKnopp(epsilon=eps)`) computes a counterfactual cell-cell matching between control and treated to estimate individualized treatment effects (+ a `synergy` function for combinations). Iterative structure = **Sinkhorn iterations** (`eps` stop-condition) — an OT-solver loop, but for *matching observed* cells, not for *predicting unseen* perturbations. ACT relevance: only as OT-solver-iteration lineage; **not a perturbation-response predictor** (unseen_pert_strategy = none — requires observed treated cells). Noted for OT-family completeness per instructions.

---

## Cross-family ACT summary (refinement-round analog = inference iteration count)

| Model | Iterative core (file:function) | Step-count knob | ACT graftability | Effort |
|---|---|---|---|---|
| CellFlow | `diffrax.diffeqsolve` in `_otfm.py::_get_predict_fn.solve_ode` | ODE solver steps (`dt0`/solver/PIDController) | **HIGH** | LOW–MED |
| scDiff | `p_sample_loop` reverse chain, `model.py:728` | `t_start` ≤ 1000 denoising steps | **HIGH** | LOW–MED |
| X-Cell | masked-diffusion decode (unreleased); API `n_diffusion_steps=4` | diffusion refinement steps | MEDIUM (stub) | HIGH |
| AlphaCell | latent Euler ODE t=0→1 (paper; no code) | Euler steps | HIGH (paper-only) | HIGH |
| CellOT | `T(x)=∇g(x)` one-shot; `transport.py` dosage interp | none native (invent displacement sub-steps) | MEDIUM | MEDIUM |
| IMPA | `Generator.forward` fixed encode→decode | none (structural depth) | LOW | HIGH |
| Cell Painting CNN | EfficientNet forward pass | none (structural depth) | LOW | HIGH |
| CINEMA-OT | Sinkhorn iterations (matching, not prediction) | Sinkhorn `eps` | n/a (not a predictor) | — |

**Best ACT hosts in this family: CellFlow and scDiff** — both have a first-class, already-tunable inference loop (ODE solver steps / diffusion denoising steps) with a per-step convergence signal (velocity norm / VLB / x0-change) ready to feed a learned halting head. CellOT is a MEDIUM (must invent the displacement-refinement axis). X-Cell & AlphaCell are architecturally HIGH but verification-blocked (stub / no repo). IMPA & Cell Painting CNN are LOW (single-shot feed-forward).
