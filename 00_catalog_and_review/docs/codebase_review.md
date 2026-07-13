# Codebase Review — Single-Cell Perturbation Models (2024–2026)

**Purpose.** Per-model, *code-grounded* notes from cloning/reading the actual repositories:
model class, forward pass, training loop, inference API, and — for the adaptive-depth thesis —
the iterative structure and where a learned-halting (ACT) refinement loop would attach.

**Method.** Each GitHub link was verified live against the GitHub REST API (authenticated).
Repos were shallow + blobless cloned (weights/checkpoints never fetched) or read via raw blobs.
Architecture / iterative-structure / graftability claims cite the file + class/function actually read.
Verification date: 2026-07-11.

**Families:** (1) Foundation-model & condition-adaptation, (2) Knowledge-graph / GRN,
(3) Chemical / drug, (4) Optimal-transport / flow / diffusion / world-model, (5) Transfer / spatial + benchmarks.

---

# Code-grounded notes — Foundation-model-based & condition-adaptation perturbation models

Track: **Foundation-model-based & condition-adaptation perturbation models** (~2024–2026).
Verification date: 2026-07-11. All GitHub facts pulled live from the GitHub REST API
(authenticated). Architecture/iterative-structure claims grounded in files actually read
(raw.githubusercontent.com blobs + git trees). Disk kept tight: no clones retained, blobs
read individually, weights/checkpoints never fetched.

**ACT lens (why the "iterative_structure" field matters):** the user wants backbones whose
forward pass contains a repeatable/unrollable computation whose *step count* could be turned
into an Adaptive-Computation-Time (Graves 2016) halting loop — a learned halting head + ponder
cost that lets *the number of refinement rounds* vary per perturbation. So for each model I
report (a) is there already an iterative core, (b) where the step-count is set, (c) how hard it
is to wrap that core in a halting loop, and (d) the concrete file/class insertion point.

---

## 1. STATE (State Transition + State Embedding) — Arc Institute
- **Repo:** https://github.com/ArcInstitute/state — VERIFIED_ACTIVE. pushed 2026-07-06, 614★,
  size ~129 MB, default branch `main`. Also `ArcInstitute/cell-eval` (eval), `state-reproduce`,
  `cell-load` (dataloaders). PyPI `arc-state`.
- **Files read:** `src/state/tx/models/state_transition.py` (796 lines; classes
  `StateTransitionPerturbationModel`, `ConfidenceToken`, `CombinedLoss`),
  `src/state/tx/models/base.py` (`PerturbationModel` ABC), `src/state/emb/nn/model.py`
  (`StateEmbeddingModel`, `SkipBlock`, uses `FlashTransformerEncoder`).
- **Architecture (grounded):** Two components.
  - **State Transition (ST)** = a *set-level* transformer. `forward(batch)` reshapes a batch into
    a "cell sentence" of `cell_sentence_len` cells; `encode_perturbation(pert)` +
    `encode_basal_expression(basal)` are summed → `seq_input [B,S,H]` → run through a HuggingFace
    transformer backbone (`transformer_backbone_key="GPT2"` by default, via
    `get_transformer_backbone`; LoRA optionally applied via `apply_lora`). Output is a **residual**:
    `out_pred = self.project_out(res_pred) + basal` when `predict_residual=True` (default). So the
    model predicts Δ(control→perturbed) added to the basal cell — a native "response shift" head.
  - **State Embedding (SE)** = `StateEmbeddingModel`: token/count MLP encoder →
    `FlashTransformerEncoderLayer × nlayers` (fixed stack) → cell embedding; produces the 
    embedding space ST operates in.
- **ConfidenceToken (KEY for the user):** `class ConfidenceToken(nn.Module)` — a learnable token
  appended to the cell-sentence; after the transformer, `extract_confidence_prediction()` maps its
  output through a 3-layer MLP (`hidden→h/2→h/4→1`, ReLU) to a **scalar predicted loss/confidence**
  per set. `forward` returns `(output, confidence_pred)` when enabled. This is an amortized
  "how hard/uncertain is this prediction" scalar — directly reusable as a halting signal.
- **unseen_pert_strategy:** perturbations enter as embeddings (`pert_emb`, e.g. ESM2/gene
  embeddings for genetic; learned featurization). A perturbation unseen in training is representable
  if it has an embedding → generalizes by *perturbation-embedding similarity*, not by retraining.
  Control-cell context is provided as the basal set (the "sentence"), so ST is *context-conditioned*
  at inference (many control cells in → distribution out).
- **adaptive_refinement_train:** none per-perturbation beyond optional LoRA fine-tuning of the
  backbone; standard amortized training (`training_step`). Curriculum/meta-learning: none found.
- **adaptive_refinement_infer:** **none iterative** — `predict_step` is a single forward pass
  (optionally returns confidence). No test-time optimization, no few-shot fine-tune loop in `tx/`.
  It *is* context-adaptive in the amortized sense (conditions on the provided control set).
- **iterative_structure:** the transformer backbone is a fixed L-layer stack (GPT2 config
  `n_layer`); one forward pass, no repeated block exposed as a variable-count loop. Step-count is
  the transformer depth (config), not a runtime-varied refinement loop. So **no native unrollable
  refinement axis** in the model forward — but the residual-prediction form (`pred = basal + Δ`)
  is the natural thing to iterate (feed prediction back as new basal).
- **act_graftability: MEDIUM–HIGH.** Two independent hooks: (1) the **ConfidenceToken already is a
  halting head** — wrap `forward` in a loop that re-feeds `out_pred` as the next `basal` and halts
  when `confidence_pred` crosses a threshold or the residual norm converges; add a ponder penalty on
  step count. (2) Alternatively unroll the GPT2 layer stack with per-layer early-exit. Route (1) is
  cleaner because the residual form makes iteration semantically meaningful (successive Δ refinement).
- **act_insertion_point:** `StateTransitionPerturbationModel.forward` in
  `src/state/tx/models/state_transition.py` — wrap the body (lines ~396–540) in a
  `for step in range(max_steps)` loop, re-assigning `basal ← out_pred` each round, and use the
  existing `ConfidenceToken.extract_confidence_prediction` output as the halting logit +
  ponder cost. `predict_step` (~764) would return the halt-step count per set.
- **act_effort: MEDIUM** (Lightning module; residual + confidence scaffolding already present;
  main work is the refinement loop + ponder loss + re-training the halting head).

## 2. Stack — "In-context learning of single-cell biology" — Arc Institute
- **Repo:** https://github.com/ArcInstitute/stack — VERIFIED_ACTIVE. pushed 2026-04-28, 141★,
  size ~0.9 MB (code only; weights external), branch `main`. PyPI `arc-stack`. Paper: bioRxiv
  2026.01.09.698608 (in-context learning).
- **Files read:** `src/stack/models/core/base.py` (`StateICLModelBase`),
  `src/stack/modules/attention.py` (`MultiHeadAttention`, `TabularAttentionLayer`), README.
- **Architecture (grounded):** encoder-decoder FM over **cell-by-gene matrix chunks** as the basic
  input unit ("tabular attention"). `StateICLModelBase(n_layers=6)` holds
  `self.layers = ModuleList([TabularAttentionLayer × n_layers])`. Each `TabularAttentionLayer.forward`
  does **two attention passes**: (1) *intra-cell* gene attention (`cell_attn` over genes within a
  cell, + gene positional emb), then (2) *inter-cell* attention (`gene_attn` over the cell axis of
  the chunk) + MLP. `forward(features)`: log1p → random gene masking (`apply_mask`) →
  `_reduce_and_tokenize` → `_run_attention_layers` (the loop over the 6 layers) → per-cell embeddings
  → `_compute_nb_parameters` produces a **negative-binomial** head (`nb_mean, nb_dispersion, px_scale`)
  → masked-reconstruction loss. In-context = the *other cells in the chunk* attend to each other, so
  a novel context/perturbation is handled by placing example cells in the input chunk (no weight update).
- **unseen_pert_strategy:** **in-context learning** — provide context cells (e.g. a few cells from
  the new condition/context) in the same chunk; inter-cell attention transfers the effect to query
  cells. Zero-shot generation of unseen profiles is the headline capability (README: "generation of
  unseen cell profiles in novel contexts"). No per-perturbation retraining.
- **adaptive_refinement_train:** none per-perturbation (masked-reconstruction pretraining;
  fine-tuning module exists under `finetune/`).
- **adaptive_refinement_infer:** **in-context adaptation** (amortized, single forward) — the context
  cells condition the prediction; no gradient step, no iterative solver. One pass through 6 layers.
- **iterative_structure:** `_run_attention_layers` loops over `n_layers=6` `TabularAttentionLayer`s —
  a homogeneous repeated block. Step-count = `n_layers` (constructor arg). This is the classic
  "repeatable transformer block" that ACT-on-transformers (Universal-Transformer style) targets.
- **act_graftability: MEDIUM.** The 6 tabular-attention layers are a uniform stack; make them a
  **weight-shared recurrent block** and add a per-step halting head (ACT / Universal Transformer).
  Not HIGH because the current layers are *not weight-shared* (each is a distinct `TabularAttentionLayer`)
  so straightforward halting needs either weight-tying + retrain or per-depth early-exit.
- **act_insertion_point:** `StateICLModelBase._run_attention_layers` in
  `src/stack/models/core/base.py` — replace the `for layer in self.layers` loop with a shared-layer
  `while not halted and step<max` loop, adding a halting MLP on the running cell embedding `x` and a
  ponder cost; `forward` returns the halt distribution.
- **act_effort: MEDIUM–HIGH** (weight-tying the tabular block + retraining is the real cost).

## 3. PertAdapt — Condition-Sensitive Adaptation on frozen scFMs (2025/2026)
- **Repo:** https://github.com/BaiDing1234/PertAdapt — VERIFIED_ACTIVE. pushed 2026-03-23, 6★,
  size ~3.4 MB, branch `main`. Vendors two backbone variants: `AIDOCell/PertAdapter/…` and
  `scFoundation/PertAdapter/…`, each carrying a copy of GEARS + (for AIDO) a slimmed `modelgenerator`.
- **Files read:** `AIDOCell/PertAdapter/gears/model_new.py` (classes
  `GEARS_Model_Pert_Adapter_New_aido`, `PertAdapterNew`, `MLP`),
  `AIDOCell/PertAdapter/gears/gears_ddp.py` (training/freeze logic),
  `modelgenerator/adapters/adapters.py` (adapter library), README.
- **Architecture (grounded):** GEARS-style perturbation model *on top of a frozen scFM*.
  `GEARS_Model_Pert_Adapter_New_aido.forward(data)`:
  1. **Frozen FM base embedding:** `emb = self.singlecell_model(pre_in)` — `singlecell_model` is the
     scFM (AIDO.Cell via `modelgenerator` `Embed`, or scFoundation), instantiated `.eval()` and
     `.to(bfloat16)`; in `gears_ddp.py` all `singlecell_model` params get `requires_grad=False`
     (`finetune_method=='frozen'`, and the module is `.eval()`'d each epoch).
  2. **GO-GNN perturbation embedding:** `self.sim_layers = ModuleList([SGConv × num_go_gnn_layers])`;
     the loop `for idx, layer in enumerate(self.sim_layers): pert_global_emb = layer(pert_global_emb,
     G_sim, G_sim_weight)` runs message passing over the **GO similarity graph** — this is how
     unseen perturbations borrow signal from GO-neighbors.
  3. **Condition-sensitive adapter:** `self.pert_adapter = PertAdapterNew(d_model, nhead=8)`.
     `PertAdapterNew.forward(exp_encodings, pert_encodings)`: adds sum-pooled perturbation encoding to
     each gene's FM embedding, LayerNorm, then a **gene-similarity-masked multi-head self-attention**
     — `attn_mask = self.Adjacency` where the adjacency is a GO-mask set to 0 (allowed) / −inf
     (blocked), so genes only attend to GO-connected genes; residual + FFN. This injects the
     perturbation into the frozen gene representations in a *condition-aware* way.
  4. Gene-specific decoder (`indv_w1/indv_b1`) → per-gene output, added to input expression
     (`out = w + b + x`, a residual/delta form). Optional uncertainty head (`uncertainty_w`).
- **unseen_pert_strategy:** GO-graph message passing (`sim_layers` over `G_sim`) shares embeddings
  between GO-neighboring genes → an unseen perturbed gene inherits neighbors' embedding; plus the
  GO-masked adapter attention. Same GEARS generalization mechanism, but on a frozen-FM base.
- **adaptive_refinement_train:** **YES — this is the whole point.** Only the *adapter + GNN + decoder*
  are trained; the scFM is frozen. `finetune_method=='frozen'` selectively sets `requires_grad`. This
  is parameter-efficient *condition adaptation* on top of a foundation model (adapter-tuning), not
  per-perturbation meta-learning, but it is explicitly "condition-sensitive adaptation."
- **adaptive_refinement_infer:** none iterative — single forward; no test-time gradient step.
  (Adaptation is at *train* time via the adapter.)
- **iterative_structure:** the GO-GNN loop `for layer in self.sim_layers` runs `num_go_gnn_layers`
  message-passing rounds — an **unrollable message-passing depth**. `PertAdapterNew` is a single
  attention block (could be stacked/unrolled). Step-count set by `args['num_go_gnn_layers']`.
- **act_graftability: MEDIUM–HIGH.** The GO-GNN message-passing rounds are exactly the kind of
  iterative block ACT targets (vary #rounds per perturbation). Add a halting head on `pert_global_emb`
  after each `sim_layers` round; halt when the GO-embedding converges. Alternatively unroll
  `PertAdapterNew` into N refinement blocks with halting.
- **act_insertion_point:** `GEARS_Model_Pert_Adapter_New_aido.forward` in
  `AIDOCell/PertAdapter/gears/model_new.py` — wrap the `for idx, layer in enumerate(self.sim_layers)`
  loop (GO message passing) with a per-step halting MLP + ponder cost; or stack `self.pert_adapter`
  into a variable-count refinement loop over gene embeddings.
- **act_effort: MEDIUM.**

## 4. Scouter (2025, Nat. Comput. Sci.)
- **Repo:** https://github.com/PancakeZoy/scouter — VERIFIED_ACTIVE. pushed 2025-02-01, 14★,
  size ~1.6 MB, branch `master`.
- **Files read:** `scouter/_model.py` (`ScouterModel`, full), `scouter/Scouter.py` (`Scouter` API,
  `train` loop).
- **Architecture (grounded):** simple two-MLP feed-forward. `ScouterModel.forward(pert_idx, ctrl_exp)`:
  `input_gene = self.embd[pert_idx].sum(axis=1)` (sum of **frozen LLM gene embeddings** —
  `self.embd = nn.Parameter(..., requires_grad=False)`) concatenated with `self.encoder(ctrl_exp)`
  (a control-expression MLP), fed to `self.generator` MLP → predicted expression. Encoder/generator are
  SELU-MLPs (`_build_mlp` with optional BN/LN/AlphaDropout). Training is a plain `for epoch in range`
  loop (`Scouter.train`).
- **unseen_pert_strategy:** LLM gene embeddings — any perturbed gene with an embedding vector is
  representable, so unseen single/combo perturbations are handled by summing their frozen embeddings.
- **adaptive_refinement_train:** none. **adaptive_refinement_infer:** none (single forward MLP).
- **iterative_structure:** none — pure feed-forward MLP, no repeated/recurrent block.
- **act_graftability: LOW.** Single-shot; no natural refinement axis. Would require inventing a
  recurrent generator (predict Δ, re-feed) — an architectural change, not a graft.
- **act_insertion_point:** would have to make `generator` recurrent in `_model.py`
  (`ScouterModel.forward`); not a natural fit.
- **act_effort: HIGH** (no iterative core to attach to).

## 5. GenePert (2024/2025)
- **Repo:** https://github.com/zou-group/GenePert — VERIFIED_ACTIVE. pushed 2024-10-29, 22★,
  size ~2.2 MB, branch `main`.
- **Files read:** `GenePertExperiment.py` (`GenePertExperiment`, `TrainConditionModel`, `MLP`),
  `utils.py`, README.
- **Architecture (grounded):** the simplest baseline — **regularized regression on GenePT
  embeddings**. `sklearn.linear_model.Ridge` (primary), plus `KNeighborsRegressor` and a small torch
  `MLP` as alternatives. Fits `X (perturbation GenePT embedding) → Y (mean expression change)`.
  For an unseen perturbation, applies the fitted Ridge coefficients to its GenePT embedding
  (README panel d). GenePT embeddings = OpenAI text-embeddings of NCBI/UniProt gene summaries.
- **unseen_pert_strategy:** linear map from LLM gene-embedding space → expression-change space; any
  gene with a GenePT embedding is predictable (README: generalizes with limited data).
- **adaptive_refinement_train:** none (closed-form ridge / KNN / short MLP).
- **adaptive_refinement_infer:** none (matrix-multiply / neighbor lookup).
- **iterative_structure:** none (linear model; MLP variant is 2-layer feed-forward).
- **act_graftability: LOW.** No iterative core. Useful only as a baseline anchor for the ACT study
  (its predicted Δ / #DE genes are the "effect size" covariates to regress out).
- **act_insertion_point:** n/a.
- **act_effort: HIGH / not applicable.**

## 6. scGenePT (2024/2025) — CZI
- **Repo:** https://github.com/czi-ai/scGenePT — VERIFIED_ACTIVE. pushed 2025-01-22, 31★,
  size ~11 MB, branch `main`.
- **Files read:** `models/scGenePT.py` (`class scGenePT(nn.Module)`, `GOPTEncoder` usage,
  `forward`, `pred_perturb_from_ctrl`). Imports `scgpt.model.TransformerGenerator` and reuses the
  scGPT stack.
- **Architecture (grounded):** scGPT's perturbation model augmented with **language/GO gene
  embeddings**. Builds on `TransformerGenerator`/`TransformerEncoder(encoder_layers, nlayers)`.
  Gene tokens are encoded three ways and summed: scGPT token embeddings (`self.encoder`), a learned
  perturbation embedding (`self.pert_encoder = nn.Embedding`), and **GPT-derived GO/text embeddings**
  via `GOPTEncoder` (`GO_token_embs_gpt_avg`/`_concat`, NCBI/GO text). `forward(...)` /
  `_encode` feed the summed embeddings into `transformer_encoder`, then `self.decoder` (MLM head)
  predicts post-perturbation expression. `pred_perturb_from_ctrl` predicts from control cells (mean
  over a pool).
- **unseen_pert_strategy:** GO/text embeddings + scGPT gene embeddings give every gene a
  representation → unseen perturbations handled via language-derived gene semantics (the paper's
  claim: adding textual knowledge improves generalization).
- **adaptive_refinement_train:** none per-perturbation (fine-tunes scGPT on perturbation data).
- **adaptive_refinement_infer:** none iterative (single encode→decode pass over control cells).
- **iterative_structure:** the scGPT `TransformerEncoder` is a fixed `nlayers` stack — one forward
  pass; step-count = transformer depth (config), not a runtime loop.
- **act_graftability: MEDIUM.** Same story as scGPT: uniform transformer stack could be made a
  weight-shared ACT block, but not weight-shared today. Also supports iterating the MLM head
  (re-feed predicted expression), but no native refinement loop.
- **act_insertion_point:** the `transformer_encoder` call inside `scGenePT._encode`/`forward`
  (`models/scGenePT.py`) — wrap encoder layers in a shared-block halting loop; or iterate
  `pred_perturb_from_ctrl` with a convergence/halting check.
- **act_effort: MEDIUM–HIGH** (rides scGPT; weight-tying + retrain).

## 7. scLAMBDA (2025)
- **Repo:** https://github.com/gefeiwang/scLAMBDA — VERIFIED_ACTIVE. pushed 2026-03-20, 26★,
  size ~2.8 MB, branch `main`.
- **Files read:** `sclambda/networks.py` (`Net`, `Net_context`, `Encoder`, `Decoder`, `MINE`),
  `sclambda/model.py` (`Model`, `Model_context`, `train`, `predict`, `generate`).
- **Architecture (grounded):** **disentangled latent-variable model with latent arithmetic**.
  `Net.forward(x, p)`: `Encoder_x` (a VAE encoder → `mean_z, log_var_z`, reparameterized `z`),
  `Encoder_p` (deterministic, maps a **gene/perturbation embedding** `p` → salient code `s`);
  decoder reconstructs `x_hat = Decoder_x(z + s)` — i.e. perturbation is *added in latent space*
  (latent arithmetic). Multi-gene combos: sum salient codes of each gene embedding
  (`p_tg` list → `s_tg` summed). `MINE` mutual-information term disentangles `z` (basal) from `s`
  (perturbation). Training uses **adversarial examples on the embedding** (`p_ae = p + eps*…*sign(grad)`)
  for robustness. `Model_context` variant adds a context encoder `c` (`x_hat = Decoder(z+s+c)`).
- **unseen_pert_strategy:** **latent arithmetic on gene embeddings** — a novel perturbation's
  embedding maps through `Encoder_p` to a salient shift added to basal latent; combos = sum of shifts.
  Uses precomputed gene embeddings (e.g. GenePT/foundation-model), so unseen genes/combos are
  representable without retraining.
- **adaptive_refinement_train:** none per-perturbation (adversarial-robust VAE/GAN training over all
  perts). **adaptive_refinement_infer:** none iterative — `generate`/`predict` sample `z`, add `s`,
  decode in one pass (a VAE sampling step, not an iterative solver).
- **iterative_structure:** none unrollable — encoders/decoders are fixed MLPs; the only "steps" are
  training epochs. No recurrent/diffusion/message-passing loop in the forward.
- **act_graftability: LOW–MEDIUM.** The latent-arithmetic form (`x = Decoder(z + s)`) invites an
  *iterative* refinement of `s` (successive latent corrections with halting), but nothing iterative
  exists today; adding it is an architectural change. Could graft a latent-refinement loop that
  repeatedly re-encodes the decoded output and updates `s` until convergence.
- **act_insertion_point:** wrap the `z + s → Decoder_x` step in `Net.forward` (`sclambda/networks.py`)
  in a refinement loop with a halting MLP on the latent; retrain.
- **act_effort: MEDIUM–HIGH.**

## 8. scPert (2026, "Multi-modal LLM-Knowledge Fusion") — NOT_FOUND
- **Repo:** NOT_FOUND. bioRxiv 2026.04.24.720560. GitHub repo/code search (`scPert`, DOI,
  "multimodal knowledge fusion perturbation") returned no matching public repository as of
  2026-07-11; the only `*/scPert` hit (`GCS-ZHN/scPert`) is an unrelated 2022 project. No code to
  ground. Paper-only; all ACT fields "paper-only: no public code found."

## 9. scELMo (2024) — embedding provider, not a new perturbation architecture
- **Repo:** https://github.com/HelloWorldLTY/scELMo — VERIFIED_ACTIVE (as a codebase), pushed
  2026-01-31, 24★, size ~3.5 MB, branch `main`. **VERIFIED_NO_MODEL_CODE for a novel perturbation
  model** — the "Perturbation Analysis" folder vendors existing methods (CINEMA-OT, CPA, GEARS)
  and applies **LLM (GPT) gene/drug embeddings** to them; scELMo itself is an embedding-generation +
  zero-shot-analysis toolkit (`Get outputs from LLMs`, `seq2emb`), not a new nn.Module for
  perturbation prediction. Files present: `Perturbation Analysis/{CINEMAOT,CPA,gears}/…`,
  `readme.md`. So scELMo's contribution to this family is *embeddings* consumed by other models
  (like GenePert/Scouter/scGenePT consume GenePT/text embeddings).
- **unseen_pert_strategy:** provides LLM embeddings that *other* models use for unseen genes/drugs.
- **iterative_structure / act_graftability: LOW / n/a** for scELMo itself (it is not the predictor).

## 10. AIDO.Cell perturbation ("Foundation Models Improve Perturbation Response Prediction") — GenBio
- **Repo:** https://github.com/genbio-ai/foundation-models-perturbation — VERIFIED_ACTIVE.
  pushed 2026-04-16, 25★, size ~25 MB, branch `main`. (Discovered — not in seed list; directly on
  topic. The AIDO.Cell backbone lives in `genbio-ai/ModelGenerator`,
  `modelgenerator/huggingface_models/cellfoundation/modeling_cellfoundation.py`, VERIFIED_ACTIVE.)
- **Files read:** `train_generative/diffusion.py` (`GaussianDiffusion1D`),
  `train_generative/flow.py` (`FlowMatching1D`), `train_generative/schrodinger.py`
  (`SchrodingerBridge1D`), `train_generative/models.py`, `train_gnn/models.py` (`GNN`, `GNNLayer`),
  `benchmark/.../fusion/model/fusion/fusion_model.py`.
- **Architecture (grounded):** two families of perturbation heads on top of AIDO.Cell embeddings.
  - **Generative (control→perturbed distribution transport):**
    - `GaussianDiffusion1D` — DDPM/DDIM over gene-expression vectors, `timesteps=1000`,
      `sampling_timesteps` configurable. `ddim_sample` loop:
      `times = linspace(-1, T-1, sampling_timesteps+1)`, `for time, time_next in time_pairs:` predict
      noise, DDIM update. `p_sample_loop`: `for t in reversed(range(0, num_timesteps))`. Conditioned on
      `source_cells` (control) + `z_emb` (perturbation/context embedding).
    - `FlowMatching1D` — conditional OT flow matching; `sample` calls `ODESolver(velocity_model=…)`
      (torchdiffeq/flow_matching), **number of ODE integration steps is the solver knob**.
    - `SchrodingerBridge1D` — SDE/ODE bridge (`sampling_timesteps>=100`, torchsde/torchdyn),
      transports control distribution → perturbed distribution over N steps.
  - **GNN:** `GNN` with `self.mps = ModuleList([GNNLayer × num_layers])` (GCN/GIN message passing
    over a gene/perturbation graph), `for i in range(num_layers)`; `filter_by_score_adaptive.py`
    hints at adaptive filtering.
- **unseen_pert_strategy:** conditioning embedding `z_emb` (perturbation/context) — unseen
  perturbations enter as new conditioning vectors (foundation-model or graph embeddings); GNN variant
  shares signal across graph neighbors.
- **adaptive_refinement_train:** none per-perturbation (amortized generative training).
- **adaptive_refinement_infer:** **YES — iterative sampling.** Every generative head refines a sample
  over many steps (diffusion denoising / flow ODE / bridge SDE). This is per-instance iterative
  computation (though step-count is currently fixed, not learned/halted).
- **iterative_structure:** **the strongest in this family.** Diffusion `sampling_timesteps` (DDIM loop),
  flow-matching ODE solver steps, Schrödinger-bridge SDE steps, and GNN `num_layers` — all are explicit
  loops whose step-count is a set parameter. These are exactly the axes ACT varies.
- **act_graftability: HIGH.** A learned-halting / convergence-based refinement loop is a natural fit:
  add a halting head on the running sample `img`/`x_t` inside the DDIM/ODE/SDE loop and stop when the
  update norm converges (or a learned halt probability fires), with a ponder cost on step count. This
  is essentially "adaptive number of denoising steps per perturbation" — precisely the user's
  "refinement rounds" signature, and semantically clean (harder perturbations → more steps).
- **act_insertion_point:** `GaussianDiffusion1D.ddim_sample` / `p_sample_loop` in
  `train_generative/diffusion.py` (wrap the `for time, time_next in time_pairs` loop with a halting
  head on `img`/`x_start` + ponder cost); equivalently the `ODESolver` call in `FlowMatching1D.sample`
  (adaptive step controller) or the SDE loop in `SchrodingerBridge1D`. GNN route:
  `GNN.forward` `for` loop in `train_gnn/models.py`.
- **act_effort: LOW–MEDIUM** (the iterative core already exists; main work is the learned halting head
  + ponder loss + defining the convergence signal). **Best single ACT host in this family.**

---

## FM backbones (context rows — light touch, github_status only)
- **scGPT** — https://github.com/bowang-lab/scGPT — VERIFIED_ACTIVE. pushed 2026-04-29, 1600★,
  ~36 MB. Model code present: `scgpt/model/{model.py, generation_model.py, multiomic_model.py}`
  (`TransformerGenerator`, transformer encoder over gene tokens). Backbone for scGenePT and for
  STATE's `tx/models/scgpt/*` baseline. Fixed L-layer transformer → MEDIUM ACT-graftability (needs
  weight-tying for a shared ACT block).
- **scFoundation** — https://github.com/biomap-research/scFoundation — VERIFIED_ACTIVE. pushed
  2025-11-23, 419★, ~110 MB. `model/` present (MAE + Performer autobin encoder-decoder). Backbone
  for PertAdapt (scFoundation variant) and GEARS-enhanced pipelines. Performer/MAE stack, single
  forward → MEDIUM.
- **Geneformer** — HuggingFace `ctheodoris/Geneformer` (not GitHub; HF domain not fetched here —
  light-touch row). BERT-style masked gene-token transformer; embeddings consumed by GenePert and
  others. Fixed transformer depth → MEDIUM.
- **AIDO.Cell** — https://github.com/genbio-ai/ModelGenerator (`modelgenerator/huggingface_models/
  cellfoundation/modeling_cellfoundation.py`, ~76 KB) — VERIFIED_ACTIVE. pushed 2026-02-25, 117★.
  The `genbio-ai/AIDO` umbrella repo (167★, 38 KB) is a pointer/readme only (VERIFIED_NO_MODEL_CODE
  itself; real code in ModelGenerator). AIDO.Cell is the frozen backbone under PertAdapt (AIDO variant)
  and the AIDO.Cell perturbation repo. Transformer FM → MEDIUM alone; HIGH when paired with the
  generative heads above.

## Benchmarks / eval harnesses (verified)
- **cell-eval** — https://github.com/ArcInstitute/cell-eval — VERIFIED_ACTIVE (2026-07-01, 144★).
  Arc's eval suite for perturbation prediction (used by STATE / Virtual Cell Challenge).
- **scPerturBench** — https://github.com/bm2-lab/scPerturBench — VERIFIED_ACTIVE (2026-05-06, 92★,
  ~53 MB). Benchmark of single-cell perturbation-effect prediction methods.
- **scPerturbBench** — https://github.com/TianGzlab/scPerturbBench — VERIFIED_ACTIVE (2026-06-12, 9★,
  ~103 MB). Comparison scripts across perturbation-response models.
- **scPerturb (resource)** — https://github.com/sanderlab/scPerturb — VERIFIED_ACTIVE (2025-02-25).
  Harmonized perturbation datasets + E-distance tooling (data resource, not a predictor).

## ACT-graftability ranking within this family (for the user's backbone-selection)
1. **AIDO.Cell perturbation generative heads (diffusion/flow/Schrödinger-bridge) — HIGH, LOW–MED effort.**
   Native iterative sampler; `sampling_timesteps` IS the refinement-round axis. Add halting head in the
   DDIM/ODE/SDE loop. Best host.
2. **STATE — MEDIUM–HIGH, MED effort.** Residual form (`pred=basal+Δ`) + an *already-present*
   `ConfidenceToken` scalar head = re-feed loop with halting almost for free. Cleanest "graft on an
   existing model" story.
3. **PertAdapt — MEDIUM–HIGH, MED effort.** GO-GNN message-passing rounds are unrollable; halt on
   GO-embedding convergence. Also a frozen-FM + adapter design the user may want for the adaptation axis.
4. **Stack / scGenePT / scGPT — MEDIUM, MED–HIGH effort.** Uniform transformer stacks → Universal-
   Transformer-style ACT after weight-tying + retrain.
5. **scLAMBDA — LOW–MEDIUM.** Latent-arithmetic invites iterative latent correction but nothing
   iterative exists today.
6. **Scouter / GenePert — LOW.** Single-shot MLP / regression; no refinement axis. Serve as effect-size
   baselines/covariates in the reproducibility→non-redundancy analysis.

---

# Code-grounded notes: Knowledge-graph / GRN genetic-perturbation models

Family track: KG/GRN genetic-perturbation models. Verification via GitHub REST API
(pushed_at/stars/size) + shallow blobless clones, core files read directly.

---

## GEARS (2023, Nature Biotechnology) — ANCHOR
Repo: https://github.com/snap-stanford/GEARS — VERIFIED_ACTIVE
Evidence: pushed_at 2025-02-01, stars 381, size 6587 KB, default branch `master`.
Files read: `gears/model.py` (full), `gears/gears.py` (defaults, train, predict), `gears/utils.py` (loss_fct, GO graph build).

### Model class + forward pass
- `GEARS_Model(torch.nn.Module)` in **gears/model.py**.
- TWO GNN stacks, each an `nn.ModuleList` of `torch_geometric.nn.SGConv` layers:
  - `self.layers_emb_pos` — SGConv on gene **co-expression** graph (`G_coexpress`), depth `num_gene_gnn_layers`.
  - `self.sim_layers` — SGConv on **GO** perturbation-similarity graph (`G_sim`/`G_go`), depth `num_go_gnn_layers`.
- forward(data): base gene embedding (`gene_emb`) + positional embedding refined by co-expr GNN (`for idx,layer in enumerate(self.layers_emb_pos): pos_emb = layer(pos_emb, G_coexpress, weight)`), then per-perturbation global embedding refined by GO GNN (`for idx,layer in enumerate(self.sim_layers): pert_global_emb = layer(...)`), fused into perturbed genes via `pert_fuse` MLP; decoded by a shared MLP (`recovery_w`) + per-gene weights (`indv_w1/indv_w2`) + a `cross_gene_state` MLP; output added residually to input expression `x`.
- Optional `uncertainty_w` head → (pred, logvar).

### Iterative structure (KEY)
- Message-passing depth is EXPLICIT and unrollable: two `for` loops over ModuleLists of SGConv. Step-count set by `num_go_gnn_layers` / `num_gene_gnn_layers` (both **default = 1** in gears.py:123-124). SGConv also has internal K-hop (`SGConv(hidden,hidden,1)` → K=1).
- So the natural ACT axis exists but ships shallow (1 layer). Grafting ACT = deepen the GO-GNN stack and wrap the `sim_layers` loop with a per-step halting head + ponder cost.

### Training / inference
- `GEARS.train(epochs=20,...)` gears.py:478; loss = `loss_fct` (MSE with gamma=2 exponent + direction/sign loss) or `uncertainty_loss_fct`, utils.py:388/339.
- `GEARS.predict(pert_list)` gears.py:300 — amortized feed-forward; **raises ValueError if a gene not in pert_list/GO graph**. No per-pert test-time adaptation.

### Unseen-perturbation strategy
- GO graph built from `gene2go.pkl` Jaccard similarity (utils.py:131-143, `get_GO_edge_list`). The GO-GNN (`sim_layers`) propagates embeddings from GO-neighbor perturbations to the query perturbation → generalizes to unseen single genes and combos (combos = sum of single-gene embeddings, `pert_track`). Still limited to genes present in the GO/pert graph vocabulary.

### ACT graftability: MEDIUM-HIGH
- Insertion point: `gears/model.py` GEARS_Model.forward, the `for idx, layer in enumerate(self.sim_layers)` loop (and/or `layers_emb_pos`). Add a halting MLP reading `pert_global_emb` after each SGConv step, accumulate halting prob + ponder loss (Graves ACT). Effort MEDIUM: loop already exists and is unrolled by hand; main work = deepen default depth, add halting head + ponder term to loss_fct, thread step budget through gears.py config.
- Caveat: default depth 1 means "as shipped" there is nothing to halt over; the axis is real but must be deepened first.


---

## TxPert (2026, Nature Biotechnology) — headline OOD/unseen perturbation
Repo: https://github.com/valence-labs/TxPert — VERIFIED_ACTIVE
Evidence: pushed_at 2026-03-25, stars 40, size 38607 KB, default branch `main`.
Files read: `gspp/models/txpert.py`, `gspp/models/pert_models/basic_gnn.py`, `gspp/models/pert_models/multi_graph.py`, `gspp/data/datamodule.py` (splits), README.

### Model class + forward pass
- `TxPert(nn.Module)` in **gspp/models/txpert.py**. Two-part decomposition:
  1. **Basal/control model** (`BASAL_STATE_MODEL_DICT`: vae/mlp/moe/moa) encodes control cell → `z_intrinsic`.
  2. **Perturbation model** (`PERT_MODEL_DICT`: mlp / `GNN` / `HybridWrapper` / `MultiGraph` / `ExphormerModel`) produces a per-perturbation latent `z_p` on the knowledge graph(s).
  - Decoder: `nn.Sequential` MLP on `cat([z_intrinsic + pert_z, cntr_covariates])`, residual add to control expression.
- forward(cntr, pert_idxs, p_emb) (txpert.py:172): runs basal model, runs pert model on the graph, sums the graph node embeddings of the perturbed genes into `pert_z`, decodes. Loss = recon (hybrid MSE) + KL (if VAE basal), `TxPert.loss`.

### Iterative structure (KEY) — TWO explicit message-passing loops
- **`GNN` pert model** (basic_gnn.py): `self.gnn_layers = nn.ModuleList()` of GAT/GCN layers, **default `num_layers=4`**, forward does `for idx, layer in enumerate(self.gnn_layers): x = layer(x, edge_index, ...)` with optional learned skip connections. Depth = `num_layers`.
- **`MultiGraph`/`MGAT`** (multi_graph.py): stacks MULTIPLE knowledge graphs (`n_layers = len(self.graphs)`, one per KG) + `MGAT` with `num_hidden` GATv2 message-passing layers (`for i, layer in enumerate(self.layers): h = layer(h, edge_index)`), a parallel structural encoder (`MLStruct`), fused per-layer by `GatedCombiner` (a learned sigmoid gate — already a soft, per-layer mixing that is close in spirit to a halting gate).
- Step-count knobs: `num_layers` (GNN), `num_hidden` (MGAT). Both are explicit unrolled Python loops.

### Unseen-perturbation strategy (headline)
- Multiple biological **knowledge graphs** (STRING/GO/etc.) give every perturbation gene a node; GNN message passing propagates embeddings from graph neighbors, so an unseen perturbation gene inherits a representation from its KG neighborhood. OOD is evaluated via predefined held-out splits (`datamodule.py:_load_predefined_splits`, `splits/train_test_split.pkl`, `subgroup.pkl`). This is exactly "GO/KG neighbors share embeddings," extended to several graphs + a structural graph encoder.

### ACT graftability: HIGH
- Insertion point: `gspp/models/pert_models/basic_gnn.py` `GNN.forward` (`for idx, layer in enumerate(self.gnn_layers)`) and/or `gspp/models/pert_models/multi_graph.py` `MGAT.forward` (`for i, layer in enumerate(self.layers)`). Add a per-step halting MLP reading `x`/`h`, accumulate halting probability + ponder cost. The `GatedCombiner` already provides a per-layer learned gate that could be repurposed as the halting head.
- Effort: MEDIUM. Loops are already explicit and unrolled; `num_layers`/`num_hidden` are config-driven. Main work = add halting head + ponder term to `TxPert.loss`, and make the layer loop early-exit on the halting signal.


---

## biolord (2024, Nature Biotechnology) — disentanglement
Repo: https://github.com/nitzanlab/biolord — VERIFIED_ACTIVE
Evidence: pushed_at 2024-08-12, stars 98, size 1644 KB, default branch `main`.
Files read: `src/biolord/_module.py`, `src/biolord/_model.py`.

### Model class + forward pass
- `BiolordModule(BaseModuleClass)` in **src/biolord/_module.py** (scvi-tools module). Disentangled generative model:
  - Per-sample **latent_unknown_attributes** via `RegularizedEmbedding` (nn.Embedding + noise/L2) — the "residual" unexplained variation.
  - Known attributes split into **ordered** (`self.ordered_networks` = `nn.ModuleDict` of MLPs, one per ordered/continuous attribute e.g. dose/time) and **categorical** (`self.categorical_embeddings` = `nn.ModuleDict` of `nn.Embedding`).
  - `inference()` (line 332): concatenate latent_unknown + all attribute latents → `latent`. `generative()` (line 394): `Decoder`/`DecoderSCVI` (scvi) maps latent → expression mean/var. Single encode→decode, no recurrence.
  - Losses: `GaussianNLLLoss` + MSE reconstruction; the disentanglement comes from regularizing the unknown-attribute embedding, not from adversarial classifiers.
- `Biolord.predict()` (model.py:419): amortized feed-forward — loops minibatches, calls `module.get_expression`. No test-time optimization.

### Unseen-perturbation strategy
- Disentangled attribute latent space: for a NEW perturbation you supply its attribute value and read its embedding through the (continuous) `ordered_networks[attr](val)` (model.py `get_ordered_attribute_embedding`) or a categorical embedding, then decode combined with any cell's basal state. Generalization is via the smooth ordered-attribute networks + compositional latent, **not** a graph. (No KG/GRN — included as a disentanglement contrast to the graph models.)

### Iterative structure
- NONE that varies a step count. Decoder has `decoder_depth=4` MLP layers but that is a fixed feed-forward stack, not a repeatable message-passing/refinement block over which halting is meaningful.

### ACT graftability: LOW
- Single-shot amortized encode→decode. No natural refinement axis: the only "depth" is the decoder MLP, and halting over generic MLP layers is not the intended ACT use (no fixed-point / iterative-refinement semantics). Insertion would require ADDING an iterative refinement loop (e.g. iterative latent correction) that does not exist today → effort HIGH, low fidelity to the design.
- act_insertion_point: none native; would have to wrap `generative()` in an iterative latent-refinement loop (architectural change, not a graft).


---

## PDGrapher (2024/2025, Nature Biomedical Engineering) — causally-inspired, combinatorial
Repo: https://github.com/mims-harvard/PDGrapher — VERIFIED_ACTIVE
Evidence: pushed_at 2025-10-03, stars 128, size 65094 KB, default branch `main`.
Files read: `src/pdgrapher/_models.py`, `src/pdgrapher/pdgrapher.py`, `src/pdgrapher/train.py`, `experiments_resubmission_bme/genetic.py`.

### Model class + forward pass
- `GCNBase(nn.Module)` (_models.py:65) with `self.convs = nn.ModuleList()` of custom `GCNConv` (from `_torch_geometric`) over a PPI/GRN `edge_index`, depth `n_layers_gnn`, plus an MLP head (`n_layers_nn`).
- Two subclasses:
  - `ResponsePredictionModel` (_models.py:172): predicts treated/perturbed expression given (expression, intervention). forward → `_get_embeddings` → `from_node_to_out`.
  - `PerturbationDiscoveryModel` (_models.py:225): predicts WHICH intervention (perturbation set) converts diseased→treated — the inverse/causal problem.
- `PDGrapher` wrapper (pdgrapher.py:14) holds BOTH models + separate optimizers.
- Core GNN loop: `from_node_to_out` (_models.py:142): `for conv, bn in zip(self.convs, self.bns): x = F.elu(conv(x, self.edge_index, x_j_mask=..., ...)); x = cat([x1,x2,x]); x = bn(x)`. Message passing over the biological network.
- **Causal "mutilation"**: `mutilate_graph()` builds `x_j_mask` that ZEROES incoming messages to intervened nodes (do-operator on the GRN) — this is the causally-inspired inductive bias.

### Iterative structure (KEY)
- GNN message-passing depth `n_layers_gnn` — explicit `for conv,bn in zip(self.convs,self.bns)` loop. Paper/experiments sweep `n_layers_gnn ∈ [1,2,3]` (genetic.py:38). So depth is a real, already-swept axis.
- PLUS a train-time forward↔backward **cycle**: model_2 predicts intervention → model_1 predicts treated state from that intervention (train.py `_train_one_pass`, backward pass), an outer 2-step consistency loop (not variable-length though).

### Unseen-perturbation strategy
- GNN over the PPI/GRN propagates from the intervened node to the rest of the graph; the **perturbation-discovery** model can output NEW intervention sets (combinations) not seen in training by scoring every gene node → generalizes to unseen combinatorial interventions. Node identity comes from graph position + positional embeddings, so unseen target genes still have a graph node.

### ACT graftability: HIGH (message-passing depth) / MEDIUM (cycle)
- Insertion point: `src/pdgrapher/_models.py` `GCNBase.from_node_to_out`, the `for conv, bn in zip(self.convs, self.bns)` loop — add a per-step halting head on the pooled node state + ponder cost. `n_layers_gnn` already varied {1,2,3}, so a learned adaptive depth is a natural drop-in.
- Alternative: make the forward↔backward cycle in train.py iterate to convergence (fixed-point) with a halting criterion.
- Effort: MEDIUM. Custom GCNConv + `x_j_mask` plumbing means each step re-concatenates `[x1,x2,x]`; a halting head must respect that residual-cat shape.


---

## SAMS-VAE (Sparse Additive Mechanism Shift VAE) (NeurIPS 2023) — latent-arithmetic VAE
Repo: https://github.com/insitro/sams-vae — VERIFIED_ACTIVE
Evidence: pushed_at 2023-10-20, stars 15, size 24829 KB, default branch `main`. NOTE: paper is
Bereket & Karaletsos, **NeurIPS 2023** (candidate said 2024 — corrected from README).
Files read: `sams_vae/models/sams_vae/model.py`, `guides/correlated_normal_guide.py`, `predictor.py`, `loss_module.py`, `sams_vae/models/utils/perturbation_lightning_module.py`, `models/utils/predictor.py`, README.

### Model class + forward pass
- `SAMSVAEModel(torch.nn.Module)` in **sams_vae/models/sams_vae/model.py** — a Pyro-style probabilistic VAE.
- Generative model (forward, line 51): latent basal state `z_basal ~ N(0,I)` (per cell); per-treatment latent shift `E ~ N(0,σ)` and sparsity `mask ~ Bernoulli` (both **perturbation-plated** = one global latent per treatment). Core equation:
  `z = z_basal + D @ (E * mask)` — a **sparse additive latent shift** (D = dosage/one-hot of perturbations). Decoder MLP (`get_likelihood_mlp`, `decoder_n_layers`) maps z → expression likelihood (NB / normal).
- Guide (`SAMSVAECorrelatedNormalGuide`): amortized `z_basal_encoder` MLP over expression; variational params `q_mask_logits`, embedding encoder for E; Gumbel-softmax relaxation for the Bernoulli mask (`gs_temperature`).
- Loss: ELBO / IWELBO (`loss_module.py`, `PerturbationPlatedELBOLossModule`), optimized with Adam, `n_particles=5` MC samples (`perturbation_lightning_module.py`).

### Iterative structure
- NONE with a variable step-count. "n_particles" is Monte-Carlo sampling breadth, not sequential refinement. The decoder is a fixed MLP. Inference is **amortized** (single encoder forward), not iterative SVI per instance.

### Unseen-perturbation strategy
- **Latent arithmetic / compositional mechanism shifts**: because a combination's effect is the ADDITIVE sum of its constituent sparse shifts `Σ E_i*mask_i`, unseen COMBINATIONS of seen single perturbations are predicted by adding their learned mechanism vectors. But it needs each single perturbation to have been seen to estimate its `(E, mask)` — a genuinely novel single treatment has no learned shift (no molecular/graph encoder). So: generalizes to unseen combos, NOT to unseen singletons.

### ACT graftability: LOW
- Amortized single-shot VAE; the only repeatable axis is the decoder MLP depth, which has no fixed-point/refinement semantics. Could add iterative-amortized-inference (e.g. iterative refinement of q(z_basal)) — but that is an inference-scheme change, not present today. act_insertion_point: none native; a refinement loop would wrap the guide's `z_basal_encoder` as iterative amortized VI. Effort HIGH.


---

## STAMP (2024, Nature Computational Science) — subtask decomposition
Repo: https://github.com/bm2-lab/STAMP — VERIFIED_ACTIVE
Evidence: pushed_at 2025-05-23, stars 19, size 2233 KB, default branch `main`.
Paper: "Toward subtask-decomposition-based learning..." Nat Comput Sci 2024 (s43588-024-00698-1).
Files read: `stamp/Modules.py`, `stamp/STAMP.py`.

### Model class + forward pass
- `STAMP` wrapper (STAMP.py:26) builds a `TaskCombineLayer_multi_task` (Modules.py:134). The prediction is **decomposed into 3 chained subtasks**, each an MLP "level layer":
  1. `First_level_layer` — per-gene binary DE classifier (is this gene a DEG?). MLP 1024→2048, Sigmoid.
  2. `Second_level_layer` — direction (up/down) of change, conditioned on DEG mask + gene embeddings (`cat` of perturbed-gene + target-gene embeddings). Sigmoid.
  3. `Third_level_layer` — magnitude of change (regression), on the hidden features of stage 2, masked by DEG mask. LeakyReLU.
- forward (Modules.py:148): `output_1 = first_level(X)` → `output_2, mask, hids = second_level(level_1_output, X)` → `output_3, mask = third_level(hids, mask)`; returns `(loss_1 BCE, loss_2 BCE, loss_3 MSE)`. Explicit **3-stage chain with masking** (DEG mask gates stages 2/3).
- There is also a Bayesian variant (`Bayes_first_level_layer`) and a `_concate` variant.

### Iterative structure
- The forward pass is a **fixed 3-stage cascade** (subtask-1 → subtask-2 → subtask-3), NOT a repeated/recurrent block. Step count is fixed at 3 (the semantic decomposition), not a tunable message-passing/refinement depth. Each stage is a plain MLP.

### Unseen-perturbation strategy
- Uses **precomputed gene embeddings** (`Gene_embeddings`, joblib-loaded, STAMP.py:42-45) for every gene incl. perturbation targets; a new perturbation gene generalizes if it has an embedding. Cross-cell-line prediction swaps in a new cell line's gene-embedding notebook (`prediction_cross_cell_line`, STAMP.py:402). So generalization is via shared gene-embedding space, not a graph GNN.

### ACT graftability: LOW-MEDIUM
- The 3-stage cascade is a *semantic* decomposition (each stage a different task), so you cannot simply "run more steps" — extra steps have no defined target. There is no repeatable homogeneous block. act_insertion_point: none native for a per-perturbation halting loop; the closest would be to make each MLP stage internally iterative, which is not the design.
- Effort HIGH to add a meaningful ACT axis (would require re-architecting the cascade into a recurrent refinement). LOW-MEDIUM only if one repurposes the DEG-mask cascade as an "early-exit after subtask k" which is not per-perturbation refinement of a single endpoint.


---

## CellCap (2024) — interpretable response-program VAE (attention)
Repo: https://github.com/broadinstitute/CellCap — VERIFIED_ACTIVE
Evidence: pushed_at 2025-03-04, stars 16, size 12617 KB, default branch `main`.
Files read: `cellcap/model.py`, `cellcap/scvi_module.py`, `cellcap/nn/attention.py`, `cellcap/mixins.py`, `cellcap/nn/decoder.py`.

### Model class + forward pass
- `CellCapModel(BaseModuleClass, CellCapMixin)` in **cellcap/model.py** (scvi-tools). A linear-decoder VAE that decomposes the perturbation response into **latent "response programs"**.
  - Learnable params: `H_pq` (drug→program usage), `w_qk` (program→latent basis), `H_key` (per-head attention keys), `w_covar_dk` (covariate shift), `alpha_q`.
  - inference (model.py:169): encode basal `z_basal = z_encoder(x)`; program usage `h = p @ softplus(H_pq)`; **attention**: for each head, key = `p @ H_key`, `score = z_basal·keyᵀ`, `attn = softmax(score)`; multi-head combined by `torch.max`; `H_attn = attn * h`; perturbation shift `delta_z = H_attn @ tanh(w_qk)`.
  - generative (model.py:232): `z = z_basal + delta_z + delta_z_covar` → `LinearDecoderSCVI` → NB/Poisson likelihood.
  - Adversarial `discriminator`/`covarclassifier` with gradient reversal (`nn/gradient_reversal.py`, `advclassifier.py`) to disentangle basal from perturbation. Loss = recon + KL + `lamda * adv_loss` (BCE).

### Iterative structure
- NONE with variable step count. The multi-head attention loop `for i in range(self.n_head)` is parallel heads, combined by max — a single attention pass, not sequential refinement. Decoder is linear. No message passing, no diffusion/flow steps.

### Unseen-perturbation strategy
- Perturbations enter as a one-hot/dosage `p` mapped through `H_pq`/`H_key`; there is **no molecular/graph encoder for genes**, so a genuinely unseen perturbation has no learned column in `H_pq`. CellCap targets INTERPRETATION of seen perturbations (which response programs each drug activates), not zero-shot prediction. unseen_pert_strategy ≈ 'none — needs seen perts' (new perts need their `p` column learned).

### ACT graftability: LOW
- Single attention pass + linear decoder; no repeatable refinement block. act_insertion_point: none native. One could iterate the attention→delta_z→re-encode loop as a recurrent refinement of `z`, but that fixed-point interpretation is not in the code. Effort HIGH.


---

## CausalGRN (2025, bioRxiv preprint) — statistical causal SEM (NOT a neural net)
Repo: https://github.com/yub-hutch/CausalGRN — VERIFIED_ACTIVE
Evidence: pushed_at 2026-06-11, stars 3, size 2789 KB, default branch `main`. **R package** (0 Python files; 18 .R files). Authors Bo Yu & Wei Sun (Fred Hutch). Paper: bioRxiv 10.64898/2025.12.30.692369.
Files read: `R/expression_prediction.R`, `R/infer_causalgrn.R`, `R/infer_skeleton.R`, `R/partial_correlation.R`, `DESCRIPTION`, `NAMESPACE`, README.

### Method (no nn.Module — this is a statistics package)
- Three-stage causal pipeline:
  1. **Skeleton inference** (`infer_skeleton.R`, `perform_ci_test`): undirected graph via partial-correlation CI tests with an **adaptive thresholding correction** for spurious partial correlations in sparse scRNA-seq (a `repeat{}` loop over conditioning thresholds — a numerical search, not NN).
  2. **Orientation** (`infer_causalgrn.R`, `infer_causalgrn`): orient edges using observed CRISPR KO outcomes, `for (ko in kos)` first- and second-order orientation rules.
  3. **Prediction** (`expression_prediction.R`): fit linear structural equations per gene (`fit_expression_model`, lm/lasso/ridge over graph parents `igraph::neighbors(graph, gene, mode='in')`), giving coefficient matrix `B`.
- **Unseen-perturbation prediction = network propagation via steady-state linear solve**: `.impute_deltas` solves `Δx = B Δx` → `(I − B_UU)⁻¹ · driving_force` with `base::solve` (closed-form, `expression_prediction.R:199`). A nonlinear **Gaussian-process variant** exists (`fit_expression_model_with_gp`, `infer_causalgrn_with_gp`).

### Unseen-perturbation strategy
- The directed causal GRN + structural equations propagate a knockout's direct effect to downstream genes → predicts the transcriptome shift for a perturbation never seen in training, as long as the KO'd gene is a node in the GRN. This is the "orient with perturbations, then propagate on the directed graph" strategy.

### Iterative structure
- The perturbation prediction is a **single linear-system SOLVE** (`solve(I − B_UU, driving_force)`), i.e. the fixed point is obtained in closed form, NOT by unrolled iteration. Skeleton inference has a `repeat{}` threshold search but that is model-selection, not a differentiable refinement step-count.

### ACT graftability: NOT APPLICABLE / LOW
- No neural network, no differentiable forward pass, no gradient training → ACT (a learned halting head + ponder loss on an iterative *neural* core) does not apply. One COULD replace the closed-form `solve()` with an iterative fixed-point solver (Jacobi/Neumann series `Δx_{t+1} = B Δx_t + f`) and count iterations-to-convergence — that iteration count is a genuine "response-complexity" proxy, but it is a numerical solver step, not a learned ACT halting mechanism.
- act_insertion_point: `R/expression_prediction.R` `.impute_deltas` — swap `base::solve` for an iterative Neumann-series/power-iteration loop and record steps-to-convergence (numerical, not learned). Effort: LOW to add convergence-step counting; but this is NOT the neural ACT the user is prototyping.

---

## Cross-model synthesis (ACT-relevant)

| Model | Iterative core | Step knob | ACT graft |
|---|---|---|---|
| GEARS | 2× SGConv GNN stacks (GO + co-expr), hand-unrolled | `num_go_gnn_layers`/`num_gene_gnn_layers` (default 1) | MEDIUM-HIGH (deepen + halt on `sim_layers` loop) |
| TxPert | GAT/GATv2 message passing (GNN & MGAT), multi-KG + GatedCombiner | `num_layers`=4 / `num_hidden` | HIGH (loops explicit; GatedCombiner ≈ halting gate) |
| PDGrapher | GCNConv stack on PPI/GRN + fwd↔bwd cycle | `n_layers_gnn` (swept 1-3) | HIGH depth / MEDIUM cycle |
| biolord | none (disentangled AE, feed-forward decoder) | decoder_depth (not refinement) | LOW |
| STAMP | fixed 3-stage semantic cascade (DEG→dir→mag) | 3 (fixed, heterogeneous) | LOW-MEDIUM |
| SAMS-VAE | none (amortized VI; n_particles = MC breadth) | — | LOW |
| CellCap | single attention pass (multi-head parallel) | n_head (parallel, not sequential) | LOW |
| CausalGRN | closed-form linear solve (I−B)⁻¹ | — (or Neumann iters) | N/A (non-neural) / LOW |

**Best ACT hosts in this family: TxPert (HIGH) and PDGrapher (HIGH), then GEARS (anchor, MEDIUM-HIGH after deepening).** All three have an EXPLICIT, hand-unrolled graph-message-passing loop whose depth is already a config knob (and PDGrapher already sweeps it 1-3), so a per-step halting head + ponder cost drops onto the existing `for`-loop with minimal surgery. The GNN message-passing round IS the natural adaptive-computation axis. biolord / SAMS-VAE / CellCap are single-shot amortized models with no native refinement axis (LOW). CausalGRN is non-neural (ACT not applicable, though solver-iteration-count is a loose numerical analogue).

---

# Code-Grounded Notes — Chemical / Drug Transcriptional-Response Models

**Family:** Chemical / small-molecule perturbation → transcriptional response prediction (single-cell + bulk L1000), ~2023–2026.
**Reviewer focus:** For the ACT thesis, the key question per model is: *does the forward pass contain an iterative/recurrent/multi-step computation whose step-count could be varied, onto which a learned-halting (ACT) or convergence-based refinement loop could be grafted?*

**Verification method:** Every repo was verified live via the GitHub REST API (existence, size, `pushed_at`, stars, default branch). All source was read via `raw.githubusercontent.com` (blobless, no clones) to keep disk footprint at zero — no weights/datasets fetched. File paths + class/function names cited below were actually read.

**Headline finding for this family:** These are overwhelmingly **single-shot feed-forward encoder→(+drug latent)→decoder** models (autoencoders, conditional VAEs, CycleGAN, counterfactual AEs). Unseen-drug generalization is achieved by a **molecular-structure/fingerprint encoder** (RDKit ECFP, GAT on molecular graph, Uni-Mol 3D, GROVER/JT-VAE pretrained embeddings) that maps any SMILES → a drug latent added to the cell latent — *not* by any iterative refinement. **Consequently ACT-graftability is LOW for most.** The two exceptions with a genuine unrollable core are **XPert** (a configurable stack of cross/self-attention transformer blocks — MEDIUM/HIGH) and, weakly, any model whose molecular branch is a message-passing GNN (GAT rounds — cycleCDR/XPert drug encoder).

---

## Lineage anchors (brief)

### CPA / ComPert (Lotfollahi et al. 2023, Mol Syst Biol) — `facebookresearch/CPA` (archived) & `theislab/cpa`
- **Verified:** `facebookresearch/CPA` size 51804 KB, 186★, pushed 2023-09-07, **archived**. `theislab/cpa` (scvi-tools port) 56677 KB, 149★, pushed 2024-08-14, active.
- Autoencoder that disentangles a **basal cell latent** from additive **perturbation + covariate embeddings**, trained with **adversarial classifiers** so the basal latent carries no drug/covariate info. Dose handled by a learned `GeneralizedSigmoid` doser. Prediction = `decoder(encoder(x) + drug_emb·dose + cov_emb)`. Single-shot. This is the template the whole family builds on.

### MultiCPA (Inecik et al. 2022) — `theislab/multicpa`
- **Verified:** 278 KB, 15★, pushed 2022-07-08.
- **Read `MultiCPA/model.py`:** `class ComPert(BaseModel)` (line 487) — CPA autoencoder extended to **multimodal (RNA+protein) totalVI-style** output via `NBMixture`/`GaussianMixture` heads (lines 57, 210) and a Transformer/Marginal aggregation option. Same additive-latent + adversary scheme; single-shot forward. Prediction API `BaseModel.predict` (line 407).

---

## chemCPA (2023, NeurIPS) — ANCHOR of this family
- **Repo:** https://github.com/theislab/chemCPA — **VERIFIED_ACTIVE.** 245541 KB, 156★, pushed 2025-02-06, default `main`.
- **Files read:** `chemCPA/model.py` (full `ComPert`, `MLP`, `GeneralizedSigmoid`, `compute_drug_embeddings_`, `predict`, `update`), tree of `embeddings/{rdkit,grover,jtvae,seq2seq}/`.
- **Architecture (code-grounded):** `class ComPert(torch.nn.Module)` in `chemCPA/model.py`. Encoder `self.encoder = MLP([num_genes]+[width]*depth+[dim])`; decoder `self.decoder = MLP([dim]+...+[num_genes*2])` emitting mean+variance (Gaussian/NB NLL loss). Drug handling: `self.drug_embeddings` (nn.Embedding OR injected pretrained molecular embedding) → `self.drug_embedding_encoder` (MLP) → scaled by learned doser (`GeneralizedSigmoid`, `doser_type∈{sigm,logsigm,mlp,amortized}`). `predict()`: `latent_basal = encoder(genes); latent_treated = latent_basal + compute_drug_embeddings_(...) + Σ covariate_emb; return decoder(latent_treated)`. Adversarial drug/covariate classifiers (`adversary_drugs`, `adversary_covariates`) enforce basal disentanglement.
- **Unseen-pert strategy:** **Molecular-structure encoder generalizes to new SMILES.** The `drug_embeddings` can be initialized from **pretrained molecule encoders** (RDKit descriptors, GROVER GNN, JT-VAE, seq2seq — see `embeddings/` subdirs); `drug_embedding_encoder` maps that fixed chemical embedding into perturbation-latent space, so a never-seen molecule gets an embedding from its structure. This is chemCPA's central contribution over CPA (which used a free nn.Embedding per seen drug).
- **Adaptive refinement (train):** none per-perturbation, BUT the repo ships an explicit **fine-tuning workflow** (`config/finetune*.yaml`, `MLP(append_layer_width=..., append_layer_position='first'/'last')` adds "henc"/"hdec" layers to adapt a LINCS-pretrained model onto a new gene set / sci-Plex) — this is transfer-learning of the whole model, not per-drug adaptation.
- **Adaptive refinement (infer):** none — amortized single forward pass.
- **Iterative structure:** **None.** Encoder/decoder are fixed-depth MLPs; the only "depth" is `autoencoder_depth` (a static hyperparameter, not an unrolled-per-input loop). No recurrence, diffusion, or ODE steps.
- **ACT graftability:** **LOW.** Single-shot feed-forward; no natural refinement axis. To add ACT you'd have to *invent* an iterative core (e.g., loop the `latent_treated` through a shared residual block N times) — the additive-latent design gives no convergence signal to halt on.
- **ACT insertion point (if forced):** wrap the single `latent_treated = latent_basal + drug_embedding` update in `ComPert.predict` (`chemCPA/model.py`) as a recurrent residual refinement `z_{t+1}=z_t+f(z_t,drug)` with a halting MLP on `z_t`; requires a new shared-weight block + ponder loss in `chemCPA/lightning_module.py`.
- **ACT effort:** HIGH.

---

## PRnet (2024, Nat Commun) — `Perturbation-Response-Prediction/PRnet`
- **Repo:** https://github.com/Perturbation-Response-Prediction/PRnet — **VERIFIED_ACTIVE.** 36422 KB, 82★, pushed 2024-12-13, default `main`.
- **Files read:** `models/PRnet.py` (full: `PRnet`, `PGM`, `PEncoder`, `PDecoder`, `PAdaptor`), tree incl. `trainer/PRnetTrainer.py`, `train_lincs.py`, `test_sciplex.py`.
- **Architecture (code-grounded):** `class PRnet(nn.Module)` wraps `class PGM(nn.Module)` — a **perturbation-conditioned generative model** with three parts: **Perturb-adaptor** `PAdaptor` (maps a `comb_num*drug_dimension`=2×1024 drug fingerprint vector → `comb_dimension`=50 latent `c`), **Perturb-encoder** `PEncoder` (`[x_dim + c_dim]→hidden→z`, z_dim=10), **Perturb-decoder** `PDecoder` (`[z + c + n]→...→x_dim*2`, emits mean+var, ReLU on mean). `PGM.forward`: `c=CombAdaptor(c); z=encoder(cat(x,c)); x_hat=decoder(cat(z,c,n))`. Single-shot conditional-VAE-style.
- **Unseen-pert strategy:** **Molecular-structure fingerprint encoder** (1024-dim per drug, up to `comb_num` drugs → supports combinations) fed through the Perturb-adaptor; a new molecule's fingerprint produces a new `c`, so unseen drugs/combos generalize. Scales to large libraries (paper screens ~ thousands of drugs).
- **Adaptive refinement (train):** none per-perturbation (standard amortized training via `trainer/PRnetTrainer.py`).
- **Adaptive refinement (infer):** none — single forward pass (`test_*.py` call encoder→decoder once).
- **Iterative structure:** **None.** Fixed MLP encoder/decoder; no recurrence/diffusion/ODE.
- **ACT graftability:** **LOW.** Single-shot CVAE; no unrollable core. Latent `z`/`c` are computed once.
- **ACT insertion point (if forced):** iterate the `z→decoder` mapping in `PGM.forward` (`models/PRnet.py`) as a refinement loop over the latent, with a halting head on `z`; net-new machinery.
- **ACT effort:** HIGH.

---

## TranSiGen (2024, Nat Commun) — `myzhengSIMM/TranSiGen`
- **Repo:** https://github.com/myzhengSIMM/TranSiGen — **VERIFIED_ACTIVE.** 56655 KB, 36★, pushed 2025-01-21, default `main`.
- **Files read:** `src/model.py` (full `TranSiGen` class: `__init__`, `encode_x1/x2`, `decode_x1/x2`, `forward`, `loss`, `train_model`), tree incl. `src/vae_x1.py`, `src/vae_x2.py`, `src/prediction.py`.
- **Architecture (code-grounded):** `class TranSiGen(torch.nn.Module)` = **two coupled VAEs**. VAE-x1 encodes the **control/basal profile** (`encoder_x1`→`mu_z1,logvar_z1`→`z1`), VAE-x2 encodes the **perturbed profile** (`encoder_x2`→`z2`). A **molecular-feature bridge** predicts the perturbed latent from the control latent + drug features: `z1_feat = cat(z1, feat_embeddings(features)); mu_pred,logvar_pred = mu_z2Fz1(z1_feat); z2_pred = sample; x2_pred = decoder_x2(z2_pred)`. Loss (`loss()`) = MSE on x1 recon + x2 recon + **perturbation delta** (`mse(x2_pred - x1, x2 - x1)`) + KL terms bridging predicted→true z2. Single-shot forward.
- **Unseen-pert strategy:** **Molecular-structure encoder** — `features` are per-compound molecular descriptors (KPGT/ECFP fingerprints per README); `feat_embeddings` MLP maps them into latent space, so a new SMILES yields a new `z2_pred`. Designed for phenotype-based drug repurposing / virtual screening over unseen compounds.
- **Adaptive refinement (train):** none per-perturbation; note a two-stage init (pretrain the x1/x2 VAEs, then train the bridge) but that's global, not per-drug.
- **Adaptive refinement (infer):** none — single amortized pass (`src/prediction.py`).
- **Iterative structure:** **None.** Two parallel fixed-depth VAEs + one linear bridge; no recurrence/diffusion.
- **ACT graftability:** **LOW.** The z1→z2 bridge is a single linear map — no iteration to unroll.
- **ACT insertion point (if forced):** replace the single `mu_z2Fz1` map in `TranSiGen.forward` (`src/model.py`) with a recurrent latent-refinement `z2^{(t+1)}=z2^{(t)}+g(z2^{(t)},feat)` + halting head; net-new.
- **ACT effort:** HIGH.

---

## CODEX (2024, Bioinformatics/ISMB) — `sschrod/CODEX`
- **Repo:** https://github.com/sschrod/CODEX — **VERIFIED_ACTIVE.** 29 KB (tiny; source-only), 5★, pushed 2024-10-15, default `master`.
- **Files read:** `codex/Network_base.py` (`single_layer`, `network_block`), `codex/CODEX_reconstruction.py` (`CODEXReconstruction.forward/predict/predict_with_weighted_perturbations`), tree incl. `codex/CODEX_Dose.py`, `codex/CODEX_Synergy.py`.
- **Architecture (code-grounded):** `class CODEXReconstruction(nn.Module)` — a **counterfactual autoencoder**. `encoder = network_block(...)` → shared latent; **one `single_layer` per treatment** in `self.T_rep = ModuleList([single_layer(...) for i in range(num_treatments)])`; `decoder` emits `in_features*2` (mean+softplus var). `forward(input, treatment)`: `embedding=encoder(input); for t: latent_rep[mask_t] += T_rep[t](embedding[mask_t]); return decoder(latent_rep)`. Additive per-treatment latent transforms (CPA-like but with a dedicated small net per treatment). `predict_with_weighted_perturbations` linearly weights treatment reps → enables **dose/synergy extrapolation** (`CODEX_Dose.py`, `CODEX_Synergy.py`).
- **Unseen-pert strategy:** **Partial / "needs seen perturbations."** Each treatment gets its own `T_rep[t]` learned from data, so a genuinely novel drug has no head. Generalization is to **unseen doses and unseen combinations** of *seen* treatments via weighted latent addition (`predict_with_weighted_perturbations`), not to new molecular structures. No SMILES encoder.
- **Adaptive refinement (train):** none (standard AE training).
- **Adaptive refinement (infer):** none — single forward pass; dose/synergy handled by closed-form weighting of latent reps, not optimization.
- **Iterative structure:** **None.** The `for t in range(num_treatments)` loop is over treatment *channels* (a masked scatter-add), not a repeatable computation over the same input — step count = #treatments, fixed by the data, and each iteration uses a *different* weight matrix. Not unrollable in the ACT sense.
- **ACT graftability:** **LOW.** Single-shot; the per-treatment loop is not a refinement axis.
- **ACT insertion point (if forced):** wrap `encoder→latent_rep→decoder` in `CODEXReconstruction.forward` (`codex/CODEX_reconstruction.py`) in a recurrent block with halting on `latent_rep`; net-new.
- **ACT effort:** HIGH.

---

## cycleCDR (2024, Bioinformatics/ISMB) — `hliulab/cycleCDR`
- **Repo:** https://github.com/hliulab/cycleCDR — **VERIFIED_ACTIVE.** 193 KB, 3★, pushed 2024-01-24, default `main`.
- **Files read:** `cycleCDR/model/model.py` (full `cycleCDR`, `GANLoss`, generators `netG_A/netG_B`, `update_G/update_D`, `predict`), `cycleCDR/model/gat.py` (present), tree incl. `preprocessing/drug_embedding/generate_embedding_rdkit_*.py`, `configs/*gat*.yaml`.
- **Architecture (code-grounded):** `class cycleCDR(nn.Module)` — a **CycleGAN between control and treated transcriptomes**. `drug_encoder` = **`GATNet` (graph-attention on molecular graph)** or MLP on RDKit fingerprint; optional `dose_encoder`. Two generators: `netG_A(control, drug_emb) = decoderG_A(encoderG_A(control) + drug_emb)` (control→treat); `netG_B(treat, drug_emb) = decoderG_B(encoderG_B(treat) − drug_emb)` (treat→control). Trained with **cycle-consistency** (`rec_control=netG_B(netG_A(...))`), identity, and **GAN discriminator** losses (`update_G`, `update_D`, `discriminator_A/B`). `predict()` = single `netG_A` pass.
- **Unseen-pert strategy:** **Molecular-structure encoder generalizes to new SMILES** — the GAT (`gat.py`) encodes the drug molecular graph, so an unseen molecule produces a drug embedding added in latent space; RDKit-fingerprint MLP variant likewise.
- **Adaptive refinement (train):** none per-perturbation; the "cycle" is a training-time consistency constraint (A→B→A), not a per-input iterative refinement.
- **Adaptive refinement (infer):** none — `predict()` runs one generator pass. (The cycle A→B→A exists but is fixed 2-step and used for training/reconstruction, not test-time refinement.)
- **Iterative structure:** **Weak/borderline.** (i) The cycle is a fixed 2-hop A→B→A composition, not a variable-length loop. (ii) The **GAT drug encoder** does a fixed number of message-passing rounds — that *is* an unrollable graph-iteration axis, but it lives in the molecule branch, not the cell-response path the thesis targets.
- **ACT graftability:** **LOW–MEDIUM.** As a cell-response refiner: LOW (single generator pass). If one is willing to unroll the GAT message-passing rounds in `gat.py`, MEDIUM but off-target (halting would describe molecular-graph convergence, not response complexity).
- **ACT insertion point:** either (a) iterate `encoderG_A→(+drug)→decoderG_A` in `netG_A` (`cycleCDR/model/model.py`) as a refinement loop with a halting head on the latent, or (b) add a halting head over GAT layers in `cycleCDR/model/gat.py`.
- **ACT effort:** MEDIUM.

---

## XPert (2026, Nat Mach Intell) — `GSanShui/XPert`  ← KEY MODEL FOR ACT
- **Repo:** https://github.com/GSanShui/XPert — **VERIFIED_ACTIVE.** 11468 KB, 23★, pushed 2025-11-22 (most recently active in family), default `main`.
- **Files read:** `models/model_XPert.py` (full: `XPertNet`, `AttnEncoder`, `get_unimol_drug_feat`), `configs/config_l1000.yaml` (structure strings), tree incl. `models/model_utils.py` (`Embeddings`, `Encoder`, `crossEncoder`, `cell_Embeddings`, `unimol_Embeddings`), `train_xpert.py`, `pretrain_hg.py`.
- **Architecture (code-grounded):** `class XPertNet(torch.nn.Module)` = **biologically-informed dual-branch transformer**. Cell branch: `cell_Embeddings` (binned expression + **pretrained PPI gene vectors**). Drug branch: `unimol_Embeddings` (**Uni-Mol 3D atom features** + a **heterogeneous-graph pretrained drug embedding** `drug_HG_embed`, optional dose/time embeddings). Two **`AttnEncoder`** stacks — `attnEncoder_trt` and `attnEncoder_ctl` — each built from a **configurable string of cross-attention (`CA`) and self-attention (`SA`) transformer blocks**. In `configs/config_l1000.yaml`: `trt_structure: CA+SA+SA+CA`, `ctl_structure: SA+SA+SA+SA`, `n_heads: 8`, `hidden_size: 256`. `AttnEncoder.forward` **loops over `self.layers = structure.split('+')`**, dispatching each token to `self.crossEncoders[i]` (drug↔cell `crossEncoder`) or `self.selfEncoders[i]` (`Encoder`), optionally adding a `drug_specific_gene_embedding` residual between layers. Heads `trt_fc/ctl_fc/deg_fc` regress treated / control / **differential** expression per gene.
- **Unseen-pert strategy:** **Molecular-structure (Uni-Mol 3D) encoder + knowledge-graph drug embedding generalize to new SMILES**, and PPI-informed gene embeddings support the cell side. Drug-specific attention lets it attend molecule↔gene.
- **Adaptive refinement (train):** none per-perturbation, but a **pretraining stage** (`pretrain_hg.py`, heterogeneous-graph drug/gene pretraining) precedes fine-tuning — global, not per-drug.
- **Adaptive refinement (infer):** none — amortized single forward pass through the fixed transformer stack.
- **Iterative structure:** **YES — a stacked transformer whose block sequence is set by a config string.** `AttnEncoder.forward` iterates `for layer_type in self.layers:` over `crossEncoders`/`selfEncoders`. Step-count = length of `trt_structure`/`ctl_structure` (config `config['model']['ATTN']['*_structure']`). Blocks are **separately-parameterized** (a `ModuleList`, BERT-style — not weight-tied), so the natural ACT move is to make the *number of applied blocks* input-dependent, or add extra weight-tied refinement blocks. There is already an inter-layer residual injection (`cell_embed = cell_embed + λ·drug_specific_gene_embedding`) — a convergence signal is available on `cell_embed`.
- **ACT graftability:** **MEDIUM–HIGH.** It has a real, explicitly-looped block stack with a config-controlled step count and a running `cell_embed` state — the closest thing in this family to an unrollable iterative core. Two routes: (a) **halt over existing CA/SA blocks** (learned early-exit — MEDIUM, blocks are untied so a halting head reads `cell_embed` after each and a ponder cost penalizes depth); (b) **add a weight-tied refinement block** looped to convergence with an ACT halting head (HIGH fidelity to the thesis, more engineering).
- **ACT insertion point:** in `AttnEncoder.forward` (`models/model_XPert.py`), after each block updates `cell_embed`, insert a halting MLP `h_t = σ(W·pool(cell_embed))`, accumulate halting mass and stop when Σh_t≥1−ε (Graves ACT), adding a ponder-cost term in `train_xpert.py`'s loss. For a weight-tied variant, wrap a single `crossEncoder`+`Encoder` pair in the loop instead of the `ModuleList`.
- **ACT effort:** MEDIUM (early-exit over existing blocks) to HIGH (weight-tied refinement core).

---

## PS — Perturbation-response Score (2025, Nat Cell Biol) — `davidliwei/PS`
- **Paper:** "Decoding heterogeneous single-cell perturbation responses," Nat Cell Biol 2025 (s41556-025-01626-9). Code-availability statement (read from the PDF) points to **https://github.com/davidliwei/PS**, implemented as part of the **scMAGeCK** pipeline.
- **Repo:** https://github.com/davidliwei/PS — **VERIFIED_ACTIVE** (but not a neural predictor). 33900 KB, 17★, pushed 2024-04-20, default `main`. Contents: `README.md`, `demo/demo1/ps_demo.R`, R Markdown demos, example datasets — **no nn.Module / no Python model**.
- **Files read:** `README.md`, `demo/demo1/ps_demo.R`.
- **What it actually is (code-grounded):** An **R statistical method**, not a deep-learning perturbation-*prediction* model. PS quantifies **per-cell perturbation-response scores** for observed perturbations via `scmageck_eff_estimate(rds_object, bc_frame, perturb_gene=..., non_target_ctrl=...)` in the scMAGeCK R package (Seurat-based). It measures **heterogeneity of response among cells that received a known perturbation** (e.g., TP53 efficiency score per cell), used to discover response subpopulations — it does **not** predict transcriptomes for unseen drugs/genes.
- **Unseen-pert strategy:** **N/A — needs observed perturbed cells.** PS scores cells that were actually perturbed; there is no mechanism (and no intent) to predict a novel perturbation's profile.
- **Adaptive refinement (train/infer):** N/A (no trained neural network; it's a per-cell scoring/regression estimator, `scmageck_eff_estimate`).
- **Iterative structure:** **N/A / none** — a statistical scoring pipeline, no neural forward pass to unroll.
- **ACT graftability:** **N/A (out of scope).** No neural core, no prediction of unseen perturbations. Its per-cell PS score is, however, conceptually adjacent to the thesis's "response shift" signature — it could serve as an *external effect-size/response covariate* to regress against, not as an ACT host.
- **ACT effort:** N/A.
- **Note:** Included because it was on the assignment list, but flagged as a **modality/task mismatch** (scoring method, not a predictive model). The user's "response shift" signature could reuse PS as a ground-truth heterogeneity measure.

---

## scVIDR (2023, Patterns) — `BhattacharyaLab/scVIDR`
- **Repo:** https://github.com/BhattacharyaLab/scVIDR — **VERIFIED_ACTIVE.** 98641 KB, 8★, pushed 2025-05-23, default `main`.
- **Files read:** `vidr/vidr.py` (`class VIDR`, `predict` incl. dose-continuous branch), `vidr/modules.py` (`VIDREncoder`, `VIDRDecoder`), tree incl. `bin/scvidr_train.py`, `bin/scvidr_predict.py`.
- **Architecture (code-grounded):** `class VIDR(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass)` — a **scGen-style VAE** (scvi-tools base). `VIDREncoder`/`VIDRDecoder` are plain MLP encoder (`mean`,`log_var` heads) / decoder. Prediction is by **latent vector arithmetic**: `predict()` computes `delta = latent_treat_centroid − latent_ctrl_centroid`; treated cells = `decoder(latent_ctrl + delta)`. Optional **regression mode**: `LinearRegression().fit(latent_centroids, deltas)` predicts the delta for a held-out cell type. **Dose-continuous** branch: `treat_pred = latent_cd + delta·(log1p(d)/log1p(max_dose))` — an explicit dose-scaled latent shift.
- **Unseen-pert strategy:** **Latent arithmetic + delta-regression.** For an unseen *cell type* it regresses the perturbation delta from seen cell types' deltas; for unseen *doses* it log-scales the delta. It does **not** encode molecular structure — it generalizes across cell types/doses of a *single studied perturbation* (e.g., TCDD, IFN), not across new chemicals.
- **Adaptive refinement (train):** none per-perturbation.
- **Adaptive refinement (infer):** **Borderline-YES (lightweight, non-iterative).** At inference it *fits a small `LinearRegression`* over training-set latent deltas to produce the delta for the query (regression mode) — a per-query closed-form fit, but a single least-squares solve, not iterative optimization or gradient refinement.
- **Iterative structure:** **None.** VAE encode→(latent add)→decode is single-shot; the dose loop (`for d in doses`) just evaluates the same closed-form shift at several dose values (a sweep, not a refinement of one prediction).
- **ACT graftability:** **LOW.** Single encode/decode + algebraic latent shift; no unrollable core. The dose axis is a parameter sweep, not iteration to convergence.
- **ACT insertion point (if forced):** iterate the `latent_ctrl + delta → decoder` step in `VIDR.predict` (`vidr/vidr.py`) as a refinement loop with a halting head on the latent; net-new machinery.
- **ACT effort:** HIGH.

---

## Cross-family summary for the ACT thesis

| Model | Core computation | Iterative/unrollable? | ACT graftability | Where a halting loop attaches |
|---|---|---|---|---|
| chemCPA | AE + additive drug latent (pretrained mol emb) | No | LOW | `ComPert.predict` (invent recurrent latent block) |
| PRnet | Conditional VAE + Perturb-adaptor | No | LOW | `PGM.forward` |
| TranSiGen | Dual VAE + linear latent bridge | No | LOW | `TranSiGen.forward` (mu_z2Fz1) |
| CODEX | Counterfactual AE + per-treatment latent heads | No (loop is over treatments) | LOW | `CODEXReconstruction.forward` |
| cycleCDR | CycleGAN generators + GAT drug encoder | Weak (fixed 2-hop cycle; GAT rounds off-target) | LOW–MED | `netG_A` latent loop OR `gat.py` message-passing |
| **XPert** | **Dual-branch transformer, config-length CA/SA stack** | **YES (block sequence set by config string)** | **MED–HIGH** | **`AttnEncoder.forward` — halting head after each block** |
| PS | R statistical per-cell scoring (scMAGeCK) | N/A | N/A (out of scope) | — |
| scVIDR | scGen VAE + latent delta arithmetic/regression | No | LOW | `VIDR.predict` latent-shift loop |

**Bottom line:** Only **XPert** offers a genuine, already-present iterative block stack (with a running `cell_embed` state and a config-controlled step count) suitable for a learned-halting / early-exit ACT graft with modest effort. Every other model in the chemical/drug family is single-shot feed-forward; grafting ACT there means *introducing* a recurrent refinement core rather than instrumenting an existing one. The molecular-structure encoders (RDKit/GAT/Uni-Mol/GROVER/JT-VAE) that give these models unseen-drug generalization are orthogonal to any refinement axis — they map SMILES→latent in one shot.

---

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

---

# Code-Grounded Notes — Family: Cell-type-transfer + Spatial models + Benchmark/critique context

Track: **Transfer / spatial + benchmarks**. Reviewed 2024–2026. Every GitHub link below was
queried live via the GitHub REST API; every architecture / iterative-structure / ACT claim is
grounded in source files actually read via `raw.githubusercontent.com` (no weight/data blobs
downloaded; total disk footprint 336 KB). ACT = Adaptive Computation Time (Graves 2016): an
iterative core that runs a *variable* number of refinement steps under a learned halting head +
ponder cost. The key question per model: **does the forward pass contain an iterative/recurrent/
multi-step computation whose step-count could be varied and gated by a halting head?**

---

## 1. PrePR-CT  (Cell-Type-Specific-Graphs)

- **Repo:** https://github.com/reem12345/Cell-Type-Specific-Graphs — VERIFIED_ACTIVE.
  stars 5, size 215,842 KB (dominated by `.png`/`.pdf`/graphs, NOT weights), pushed 2025-09-10,
  branch `main`, license GPL-3.0, lang Jupyter Notebook. Zenodo DOI 10.5281/zenodo.15241234.
- **Venue/year:** bioRxiv 2024 → Nature Machine Intelligence 2026 (per task; repo is the
  reference implementation).
- **Files read:** `model.py` (127 lines), `utils.py` (545 lines, head), `README.md`,
  contents of `training/` (config_train_*.yaml, training_testing_demo.ipynb).

### Architecture (code-grounded)
- Core class **`GNN(torch.nn.Module)`** in `model.py`. Layers instantiated in `__init__`:
  - `self.conv1 = TransformerConv(-1, hidden_channels, heads=in_head)`
  - `self.conv2 = GATConv(-1, hidden_channels, heads=in_head, add_self_loops=True, concat=False)`
    (imported from `torch_geometric.nn`; GATv2Conv also imported)
  - `self.lin1`, `self.lin2` = `Linear` projections
  - `self.embd_pert = MLP([124,124], act)` — a fixed 124-dim perturbation embedding network
  - `self.lin_predict = MLP([total_genes + hidden_channels + 124, 1024, total_genes], act)`
    (multi_pert=True path) — final gene-expression predictor.
- **`MLP(torch.nn.Module)`** helper (same file): stacked `Linear`+optional `BatchNorm1d`+
  `Sigmoid`/`Softplus`.
- **`forward(x, edge_index, cell_line, cell_type_keys, ctrl, pert, pos)`**: for each cell type
  key it applies `conv1(x[key],edge_index[key]) + lin1(x[key])`, then `torch.max(...,dim=0)` to
  pool the per-cell-type gene graph into ONE cell-type embedding. It concatenates
  `[ctrl, cell_type_features]`, appends `embd_pert(pert)`, and runs `lin_predict` → predicted
  expression. **Single forward pass, no loop over refinement steps.**
- **Graphs**: `utils.py::Correlation_matrix` builds per-cell-type gene–gene correlation graphs
  (Pearson over HVGs); these are the fixed input graphs. README: "GAT layers to encode
  cell-type graphs … integrated with control gene expression and predefined perturbation
  embeddings … processed through MLPs."
- **Loss**: `utils.py::loss_fct` = **Earth Mover's Distance** (`ot.lp.emd2`, POT library) between
  predicted and true cells, grouped per perturbation (distributional, not per-cell MSE).

### Unseen-perturbation / unseen-cell-type strategy
- The model's headline generalization is to **unseen CELL TYPES** (inductive): a new cell type is
  handled by building its cell-type-specific correlation graph and passing it through the shared
  GAT encoder — no retraining. Perturbations use a **predefined perturbation embedding**
  (`embd_pert`, 124-dim), so genuinely unseen *perturbations* are limited to the embedding space
  provided; the design emphasizes small-data inductive priors across cell types rather than novel
  chemistry. Chemical perturbations (drugs) with predefined embeddings.

### Adaptive refinement / iterative structure / ACT
- **adaptive_refinement_train:** none (standard full-batch/minibatch training; StepLR).
- **adaptive_refinement_infer:** none — amortized single feed-forward inference per (cell-type,
  perturbation).
- **iterative_structure:** NONE that is step-count-varyable. The only loop in `forward` is over
  cell-type keys (data parallelism), not depth. GNN is fixed 2 conv layers.
- **act_graftability: LOW.** Feed-forward encoder+MLP with a fixed 2-layer GNN. There is no
  natural refinement axis; you would have to invent an unrolled predictor (e.g., iterate
  `lin_predict` residually) that the architecture does not currently contain.
- **act_insertion_point:** would require *adding* a recurrent refinement block around
  `GNN.lin_predict` in `model.py` — e.g., wrap the final MLP prediction in a
  `for step in range(T)` residual-update loop with a per-step halting MLP reading the pooled
  cell-type + pert embedding. Not a natural fit.
- **act_effort: HIGH** (must introduce the iterative core from scratch).

---

## 2. CONCERT  (niche-aware spatial counterfactual perturbation)

- **Repo:** https://github.com/mims-harvard/CONCERT — VERIFIED_ACTIVE. stars 24, size 17,233 KB,
  pushed 2025-11-13, branch `main`, license MIT, lang Jupyter Notebook. (Zitnik lab / HMS.)
- **Venue/year:** bioRxiv 2025 (2025.11.08.686890). Lin, Kong, Ghosh, Kellis, Zitnik.
- **Files read:** `README.md`, `src/concert_map.py` (1104 lines — the `CONCERT(nn.Module)` class,
  full `forward`, batching helpers), `src/run_concert_map.py` (623 lines, head), tree of
  `src/`, `src/CONCERT-3D/`, `src/CONCERT-long/`, `src/single_kernel/`.

### Architecture (code-grounded)
- Core class **`CONCERT(nn.Module)`** in `src/concert_map.py` (docstring: "GP-VAE for spatial
  counterfactual perturbation modeling"). Components built in `__init__`:
  - **`self.svgp = SVGP(...)`** (from `SVGP_Batch.py`) — a stochastic variational Gaussian
    process prior over spatially structured latent dims; `multi_kernel_mode=True` learns
    **perturbation-specific kernels** (README: "learns perturbation-specific kernels to capture
    various propagation patterns").
  - **`self.lord_encoder = LordEncoder(...)`** — a Lord-style attribute-disentangling encoder
    (per-sample latent + basal latent + attribute labels).
  - **`self.encoder = DenseEncoder(...)`** → posterior params for `GP_dim` GP latents +
    `Normal_dim` Gaussian latents.
  - **`self.decoder = buildNetwork(...)`** + `self.dec_mean` (MeanAct) + `self.dec_disp`
    (per-gene dispersion) → **Negative-Binomial** count likelihood (`NBLoss`).
  - **`self.PID = PIDControl(Kp=0.01, Ki=-0.005, ...)`** — a PID controller that dynamically
    anneals the KL weight β each step (`dynamicVAE`).
- **`forward(...)`**: LORD encode → `encoder` posterior → split into GP block + Gaussian block →
  `for l in range(self.GP_dim): svgp.approximate_posterior_params(...) ; svgp.variational_loss`
  (loops over latent GP DIMENSIONS, not refinement steps) → reparameterized sample →
  `decoder` → NB recon loss + β·(gp_KL + gauss_KL) + LORD penalty. **Single encode→decode pass.**
- **Counterfactual prediction (the "perturbation prediction")**: done by *switching cell
  attributes* at eval (`--target_cell_tissue`, `--target_cell_perturbation` in
  `run_concert_map.py`) and re-decoding, using the spatial GP to propagate niche effects across
  neighboring spots. `batching_latent_samples` / `batching_denoise_counts` are the eval-time
  encoders (`@torch.no_grad`).

### Unseen-perturbation strategy
- **Counterfactual attribute-switching in a disentangled latent space**: CONCERT does NOT learn a
  transferable molecular/structure encoder for novel perturbagens; instead it models perturbation
  as a switchable cell attribute (e.g., Jak2-KO vs WT) whose spatial propagation is captured by a
  perturbation-specific GP kernel. Generalization is over spatial *niches/positions* (predict
  response of unobserved spots, impute missing cells) rather than over chemically novel perts.
  Perturbation type: genetic KO (Perturb-map) + attribute counterfactuals.

### Adaptive refinement / iterative structure / ACT
- **adaptive_refinement_train:** the PID controller adaptively tunes β (KL weight) per training
  step — this is adaptive *optimization control*, not per-perturbation model refinement. Also
  `EarlyStopping`. Not per-perturbation adaptation.
- **adaptive_refinement_infer:** none per-instance; eval is amortized encode/decode. (num_samples
  Monte-Carlo latent draws exist but are averaged, not refined.)
- **iterative_structure:** the two `for l in range(GP_dim)` loops are over latent dimensions;
  `num_samples` loop is MC sampling. **Neither is a refinement recurrence whose step count
  controls prediction quality of a single perturbation.** No diffusion/ODE/message-passing
  depth axis.
- **act_graftability: LOW–MEDIUM.** As a VAE the encode→decode is single-shot. HOWEVER the SVGP
  posterior over a spatial graph is conceptually iterable (belief propagation / repeated GP
  conditioning over neighbors), and the decoder could be unrolled as an iterative refinement of
  spot expression given neighbor states. This is a *possible* but non-native refinement axis.
- **act_insertion_point:** wrap the decode step in `CONCERT.forward`
  (`src/concert_map.py`, after `z = latent_dist.rsample(); h = self.decoder(z)`) in a
  `for t in range(T)` loop that re-conditions the SVGP posterior on updated neighbor
  predictions, with a per-step halting MLP on the latent mean; requires exposing the SVGP
  `approximate_posterior_params` as a repeatable operator.
- **act_effort: HIGH** (must convert amortized GP-VAE into an unrolled iterative refiner).

---

## 3. scRank  (target-perturbed GRN ranking)

- **Repo:** https://github.com/ZJUFanLab/scRank — VERIFIED_ACTIVE. stars 75, size 24,652 KB,
  pushed 2025-10-16, branch `main`, license GPL-3.0, lang **R**. (ZJUFanLab.)
- **Venue/year:** Cell Reports Medicine 2024.
- **Files read:** `R/method.R` (588 lines), `R/class.R` (118 lines), `R/utils.R` (fetched;
  `.get_downstream_genes`, `.manifold_setup`, `.cal_module_score`, `.cluster_gene`), `README.md`.

### Architecture / method (code-grounded)
- **NOT a neural network** — a classical R pipeline over an S4 `scRank` object (`class.R`). Two
  stages (README + `method.R`):
  1. **`Constr_net()`** — builds a cell-type-specific gene regulatory network (gene×gene
     adjacency) per cell type via a regression/ensemble strategy with random cell subsampling
     (`n_selection` networks, CP tensor decomposition via `rTensor::cp` or Python `tensorly`).
  2. **`rank_celltype()`** — creates a **target-perturbed GRN (dpGRN)** by zeroing the drug
     target's outgoing edges (antagonist: `dpGRN[perturbed_target,] <- 0`) or clamping
     (agonist), then measures the **distance between the original GRN and the perturbed dpGRN in
     a shared low-dimensional manifold** (`.align_net` → `.manifold_setup` builds a graph
     Laplacian for manifold alignment, `n_dim` dims). Perturbation score per cell type =
     `.calculate_score(...)`; cell types ranked by how much the drug perturbs their network.
- **In-silico perturbation** is thus a **network edit + manifold-distance** computation, not a
  learned generative prediction of an expression vector.

### Unseen-perturbation strategy
- Handles any drug whose **direct target gene(s)** are known (`target` argument, ≤2 genes);
  generalizes to new drugs by editing the corresponding node in the GRN — **no seen-perturbation
  training required at all** (it is unsupervised/inference-only over the user's dataset). Does not
  predict a full transcriptome; outputs a *ranking* of drug-responsive cell types.

### Adaptive refinement / iterative structure / ACT
- **adaptive_refinement_train:** none (no gradient training of a predictor; GRN construction uses
  fixed random-subsampling ensembling).
- **adaptive_refinement_infer:** none per-perturbation beyond the fixed manifold alignment.
- **iterative_structure:** there IS a network-propagation hyperparameter **`n_hop` (default 2)**
  in `rank_celltype()`, BUT the code **hard-caps it at 1–3** (`if (n_hop > 3 | n_hop < 1)
  stop(...)`) and it merely toggles flags `simple_sum`/`multi_layer` passed to
  `.calculate_score`; it is not a genuine unrollable iterative solver. `.get_downstream_genes`
  is a single-hop neighbor lookup. So the "depth" axis exists but is discrete, tiny, and
  non-differentiable.
- **act_graftability: LOW.** Non-neural, R, discrete 1–3 hop propagation; there is no continuous
  differentiable iterative core to attach a learned halting head to. ACT is a poor conceptual
  fit (halting would be over a graph-propagation radius, but the method caps at 3 and has no
  learnable per-step gate).
- **act_insertion_point:** conceptually, generalize the `n_hop`/`simple_sum`/`multi_layer`
  branch in `R/method.R::rank_celltype` into a random-walk-with-restart propagation with a
  convergence/halting criterion — but this is a re-write into a different (differentiable)
  framework, not an insertion.
- **act_effort: HIGH** (language + paradigm change).

---

## 4. UNAGI  (deep generative cellular dynamics + in-silico perturbation)

- **Repo:** https://github.com/mcgilldinglab/UNAGI — VERIFIED_ACTIVE. stars 58, size 746,115 KB
  (large — tutorials/data; source read via raw, NO weights fetched), pushed 2026-07-10,
  branch `main`, license NOASSERTION, lang Jupyter Notebook. pip `scUNAGI`.
- **Venue/year:** Nature Biomedical Engineering 2025 (Zheng, Schupp, Adams et al.,
  10.1038/s41551-025-01423-7).
- **Files read:** `UNAGI/model/models.py` (397 lines), `UNAGI/train/runner.py` (286 lines),
  `UNAGI/train/trainer.py` (315 lines), `UNAGI/UNAGI_tool.py` (top-level driver),
  `UNAGI/perturbations/new_perturbation_strategy.py` (302 lines), `README.md`.

### Architecture (code-grounded)
- A **VAE-GAN** over single-cell expression with a graph encoder. In `models.py`:
  - **`GCNLayer(nn.Module)`** — `x @ weight (+bias)` graph conv (adjacency `adj`).
  - **`Graph_encoder(nn.Module)`** — `GCNLayer(input→graph_dim)` → `Linear`→hidden →
    (`fc21`,`fc22`) = latent μ, logvar. Also a `Plain_encoder` variant (no graph).
  - **`Discriminator(nn.Module)`** — 3-layer MLP with sigmoid (the GAN/adversarial critic on the
    latent/reconstruction, `adversarial=True`).
  - A `VAE` class (referenced by the perturbator) with `get_latent_representation`.
- **Iterative training (this is the model's defining loop):** `UNAGI_tool.py::run_UNAGI` runs
  **`for iteration in range(start_iteration, self.max_iter)`** (default `max_iter=10`). Each
  iteration = an EM-like round: `runner.py::load_stage_data` reads the **previous** iteration's
  stage data (`f'{self.iteration-1}'/'stagedata'/...`), the `UNAGI_runner.run()` retrains the
  VAE-GAN (`trainer.py`), re-embeds cells, re-clusters per disease stage (Leiden), and
  **rebuilds the temporal dynamic graph** linking cell clusters across stages
  (`getandUpadateEdges`, `buildGraph`) + reconstructs GRNs via **iDREM**. README:
  "Learning disease-specific cell embeddings through **iterative training processes**."
- **In-silico perturbation (`new_perturbation_strategy.py`):** class **`perturbator`** loads the
  trained VAE. `_BFS` does a **`while len(queue) != 0`** breadth-first propagation over a PPI
  network (`getPPINetworkDict`, HIPPIE/STRING) to spread a perturbation from target genes to
  neighbors; `perturb_input_data(mode='direct'|'GIN')` edits the input, `get_perturbed_embedding`
  re-encodes, and `calculate_similarity` scores the latent shift toward a "healthy" stage
  (CMAP-informed, unsupervised).

### Unseen-perturbation strategy
- **Unsupervised in-silico perturbation via latent-space manipulation + PPI propagation**: any
  single gene, gene combination, pathway, or CMAP drug can be perturbed by editing input genes
  (direct targets or BFS-propagated neighbors over the PPI graph) and re-encoding — no need for
  the perturbation to appear in training. Efficacy = movement of the cell embedding toward a
  healthier disease-stage centroid. Modality: genetic + pathway + drug/compound.

### Adaptive refinement / iterative structure / ACT — **the strongest generative candidate**
- **adaptive_refinement_train:** YES (structurally) — the outer `for iteration in range(max_iter)`
  EM loop iteratively refines embeddings→graph→GRN→embeddings. This is a *global* iterative
  refinement of the whole model/data state across disease stages, not per-perturbation, and step
  count is a fixed `max_iter` (default 10), not learned/halted.
- **adaptive_refinement_infer:** the `_BFS` `while` loop propagates a perturbation over the PPI
  graph until the queue empties — a variable-length propagation, but graph-topology-driven (not a
  learned halting head).
- **iterative_structure:** TWO genuine multi-step loops whose step counts are meaningful:
  (a) the training EM loop `for iteration in range(max_iter)` in `UNAGI_tool.py`;
  (b) the inference PPI-BFS `while` loop in `new_perturbation_strategy.py::_BFS`.
- **act_graftability: MEDIUM.** The EM iteration loop is a real unrollable core — one could add a
  convergence/halting criterion so `max_iter` becomes data-adaptive (halt when embeddings/graph
  stabilize) with a ponder cost. Mapping this to a *per-perturbation* halting signal (the user's
  goal) is indirect: the EM loop is over the whole dataset, and the per-perturbation BFS halts on
  graph topology, not learned complexity. So it hosts an adaptive-depth loop, but not cleanly a
  per-perturbation one.
- **act_insertion_point:** replace the fixed `for iteration in range(start_iteration, self.max_iter)`
  in `UNAGI/UNAGI_tool.py::run_UNAGI` with a convergence-gated `while not halt` loop + a halting
  head reading the inter-iteration embedding/graph delta (ponder cost on iteration count); OR add
  a learned stopping criterion to `_BFS` in `new_perturbation_strategy.py`. The training loop is
  the more natural adaptive-depth host.
- **act_effort: MEDIUM.**

---

## 5. SequenTx / AlphaTherapy  (RL for sequential drug treatment) — **most ACT-relevant**

- **Repo:** https://github.com/bm2-lab/SequenTx — VERIFIED_ACTIVE. stars 0, size 385,129 KB
  (bundles vendored benchmarks chemCPA/GROVER/JTVAE/tianshou + data; own source is small, read
  via raw, no weights fetched), pushed 2025-12-22, branch `main`, license MIT, lang Jupyter
  Notebook. Mirror: DELTA-TJ/SequenTx (same size, same push). Docker `xiaohanchen/alphatherapy:v1`.
- **Venue/year:** Nature Machine Intelligence 2026 (per task; described as a "theoretical
  proof-of-concept AI framework for rational design of sequential drug treatments").
- **Files read:** `scripts/AlphaTherapy/model/RL_agent.py` (135 lines),
  `scripts/AlphaTherapy/model/train_RL_agents.py`,
  `scripts/gym_cell_model/gym_cell_model/envs/ccl_env_cpd.py` (227 lines, the Gym env),
  `scripts/gym_cell_model/gym_cell_model/envs/model.py` (STATE_TRANSITION used by env),
  `reproducity/StateTransitionModel/Model/model.py` (STATE_TRANSITION training class),
  `README.md`.

### Architecture (code-grounded)
- **Two learned components + an RL loop:**
  1. **`STATE_TRANSITION(nn.Module)`** (`envs/model.py` / `StateTransitionModel/Model/model.py`):
     a small MLP predicting the **per-step transcriptomic delta** of a drug. `gene_number=978`
     (L1000), `drug_number=166`. `forward(drug, ccl)`: `sigmoid(enc1(drug))` and
     `sigmoid(enc2(ccl))` concatenated → `decoder_output_layer1`(sigmoid) →
     `decoder_output_layer2` → predicted expression vector. Trained with MSE + early stop
     (`fit()`), used at inference via `.predict(drug, ccl)`.
  2. **Gym environment `CCLEnvCPD(gym.Env)`** (`ccl_env_cpd.py`): holds the loaded
     `self.StateTransition` and a `CellViabModel` (sklearn). Key stepping code:
     - `single_action_step(action)`: `delta = StateTransition.predict(drug, cur_state)` →
       `next_state = cur_state + delta`; `self.env_step += 1`; append to `expression_ls`;
       reward `= cv - cv*delta_cv` from the cell-viability model.
     - `_reward()` sets `self.done = self.env_step >= self.max_step_number`.
     - `step(action)` calls the transition, returns `(obs, reward, done, info)`.
  3. **RL agent** (`RL_agent.py`): a **DQN** (`tianshou.policy.DQNPolicy`) over a `Net` MLP,
     trained by `offpolicy_trainer` with a `ReplayBuffer`, ε-greedy schedule, target network.
     The policy chooses which drug to apply at each step; an **episode is a sequence of drug
     applications** of length up to `max_step_number` (README: "terminal step" / "One step =
     24 h drug administration").

### Unseen-perturbation strategy
- Drugs are represented in a **drug pool** (395 drugs = 387 FDA-approved anticancer + 8 other,
  per README) with SMILES/target/pathway; the
  STATE_TRANSITION model maps a drug (166-dim descriptor) + current cell state → expression
  delta, so **new drugs generalize through the drug-descriptor input** to the transition model.
  The RL policy then composes *sequences* of these drugs. Modality: chemical (small molecule),
  sequential/multi-step; cell-line context as state.

### Adaptive refinement / iterative structure / ACT — **HIGH**
- **adaptive_refinement_train:** the DQN is trained by RL (off-policy) — the policy itself is a
  sequential decision maker; not per-perturbation fine-tuning but inherently multi-step planning.
- **adaptive_refinement_infer:** at inference the agent **rolls out a variable-length sequence of
  drug-application steps** (`env_step` from 0 up to `max_step_number`), each step re-applying the
  STATE_TRANSITION predictor and updating the cell state. This is a genuine per-instance,
  step-by-step refinement of the predicted response trajectory.
- **iterative_structure:** YES — an explicit episodic loop. The step count is set by
  `self.max_step_number` (config `max_step_number`) and terminated by
  `self.done = self.env_step >= self.max_step_number` in `ccl_env_cpd.py::_reward`. Each step is
  a `next_state = cur_state + StateTransition.predict(drug, cur_state)` recurrence — a residual
  state-transition unrolled over time. **This is exactly an unrollable iterative core.**
- **act_graftability: HIGH.** The env already runs a variable-length recurrence over a residual
  state-transition model with a learned policy choosing actions. A learned **halting head** +
  **ponder cost** attaches naturally: instead of the fixed `env_step >= max_step_number`
  termination, add a learned "stop treatment / converged" action or a halting probability read
  from `cur_state`, and penalize the number of steps (ponder). The DQN already emits per-step
  Q-values; a stop-action or ACT-style halting unit is a small extension.
- **act_insertion_point:** replace the fixed termination in
  `scripts/gym_cell_model/gym_cell_model/envs/ccl_env_cpd.py` (`CCLEnvCPD._reward` /
  `single_action_step`, the `self.done = self.env_step >= self.max_step_number` line) with a
  learned halting condition (a halting MLP on `self.cur_state`, accumulating halting probability
  ACT-style, or an explicit STOP action added to `env.action_space` in `__init__`); add a ponder
  penalty to the reward in `_reward()`. The DQN `Net`/policy in `RL_agent.py` would emit the
  halting logit.
- **act_effort: LOW–MEDIUM.** The iterative core, per-step state update, and step counter already
  exist; only the halting head + ponder cost + (optional) STOP action need to be added.
- **Caveat for the user's thesis:** SequenTx's "steps" are an explicit model of *sequential drug
  administration over time* (24h/step), i.e., a deliberate biological-time / treatment-sequence
  axis — NOT a pure computational-complexity proxy. Grafting ACT here would measure "how many
  treatment rounds to reach the endpoint," which conflates halting with treatment duration.
  This is the opposite of the user's requirement that halting be an effect-size-independent
  *computational* proxy decoupled from biological time. Flag: highest mechanical graftability,
  but semantically the step count is biological/temporal, so reproducibility-vs-effect-size
  analysis would need care.

---

# Benchmarks & Critiques (context)

| Name | Repo | Status | Scope | Headline finding |
|---|---|---|---|---|
| **PerturBench** | altoslabs/perturbench | VERIFIED_ACTIVE (88★, 1465KB, pushed 2026-02-17) | NeurIPS 2025 D&B; standardized framework, diverse datasets, metrics, model comparison for single-cell perturbation prediction | Provides a unified benchmarking platform; models struggle to generalize; simple/mean baselines are competitive on many metrics. |
| **Systema** | mlbio-epfl/systema | VERIFIED_ACTIVE (68★, 57382KB, pushed 2026-01-14) | Nature Biotechnology 2025 (s41587-025-02777-8); framework for genetic perturbation-response eval **beyond systematic variation** | Much of apparent predictive performance is explained by **systematic (technical/baseline) variation**; once you control for it, models add little over baselines — evaluation must go beyond systematic variation. |
| **scPerturBench** | bm2-lab/scPerturBench | VERIFIED_ACTIVE (92★, 53256KB, pushed 2026-05-06) | 27 methods, 29 datasets; unseen cellular contexts + unseen perturbations (genetic & chemical); 6 metrics (MSE, PCC-delta, E-dist, Wasserstein, KL, common-DEGs) | Models degrade sharply on unseen contexts/perturbations vs baselines; authors propose **bioLord-emCell** (cell-line embedding + disentanglement) to improve context generalization. |
| **PertEval-scFM** | aaronwtr/PertEval | VERIFIED_ACTIVE (38★, 20519KB, pushed 2025-07-29) | Evaluates single-cell **foundation models** for perturbation response via simple probes (PyTorch-Lightning/Hydra) | Foundation-model embeddings give **little to no benefit** over simple probes/baselines for perturbation prediction; questions "real knowledge capacity." |
| **Ahlmann-Eltze linear-baseline critique** | const-ae/linear_perturbation_prediction-Paper | VERIFIED_ACTIVE (87★, 63799KB, pushed 2025-07-18) | **Nature Methods 2025** (s41592-025-02772-6), Ahlmann-Eltze, Huber, Anders; R notebooks + benchmark | **"Deep-learning-based predictions of gene perturbation effects do not yet outperform simple linear baselines."** Central caveat of the field: for unseen (esp. combinatorial) perturbations, a linear/additive baseline matches or beats GEARS/scGPT/etc. |
| **Simple controls exceed best DL** | pfizer-opensource/perturb_seq | VERIFIED_ACTIVE (5★, 3846KB, pushed 2025-05-06) | **Bioinformatics 2025** (btaf317), Hill/Wong et al. (Pfizer); genetic perturbation prediction | **Simple controls (mean/baseline) exceed the best deep-learning algorithms**; also reports where foundation-model features *do* help, delineating the narrow regime of DL benefit. |
| **Foundation-cell-model benchmark** | turbine-ai/PerturbSeqPredBenchmark | VERIFIED_ACTIVE (7★, 41879KB, pushed 2026-03-06; fork of scGPT) | **BMC Genomics 2025** (s12864-025-11600-2); benchmarks scGPT/scFoundation/scELMo features vs RF/ElasticNet/KNN/TrainMean on Adamson/Norman/Replogle post-perturbation RNA-seq | Foundation-model embeddings do **not** consistently beat simple bulk regressors (RF/Elastic Net) or a train-mean baseline for post-perturbation prediction. |
| **TRADE** | ajaynadig/TRADEtools | VERIFIED_ACTIVE (38★, 50768KB, pushed 2026-06-08; code at ajaynadig/TRADE) | **Nat. Genetics 2025** (biorxiv 2024.07.03.601903); *Transcriptome-wide Analysis of Differential Expression* — an R method (uses `ashr`) | Not a predictor: estimates the **distribution of DE effect sizes** (transcriptome-wide) and a "transcriptome-wide impact" summary. Relevant as a rigorous DE-quantification tool for defining a perturbation's response magnitude / #DE genes (useful for the user's effect-size regressors). |

## Notes for the user's ACT thesis (family-level)
- **Best mechanical ACT host in this family: SequenTx** (already an unrolled, variable-length,
  residual state-transition recurrence with a learned policy — add a halting head + ponder cost).
  BUT its step axis is *biological treatment time* (24h/step), which conflicts with the user's
  requirement that halting be an effect-size-independent *computational* proxy. Use with the
  caveat above.
- **UNAGI** offers a real EM-style iterative training loop (`max_iter`) plus a graph-BFS
  inference propagation — MEDIUM graftability, but the loop is dataset-global, not
  per-perturbation.
- **PrePR-CT, CONCERT, scRank** are single-shot / non-iterative w.r.t. a per-perturbation depth
  axis (LOW–MEDIUM); grafting ACT means *introducing* an iterative core, not exposing an existing
  one. CONCERT's spatial GP is the most plausible place to *invent* one (repeated neighbor
  conditioning).
- **Benchmark takeaway relevant to the thesis:** the field's strongest, most consistent finding
  (Ahlmann-Eltze/Nature Methods, Systema/Nat Biotech, PertEval, Pfizer/Bioinformatics,
  turbine-ai/BMC Genomics) is that **simple linear/mean baselines match or beat deep models,
  especially on unseen perturbations, once systematic variation is controlled.** Any
  halting/complexity signature must therefore be validated to be **non-redundant with effect
  size / #DE genes / baseline-predictability** — exactly the regressing-out step the user plans.
  TRADE is a suitable tool for quantifying the DE-magnitude covariate to regress out.

---

