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
