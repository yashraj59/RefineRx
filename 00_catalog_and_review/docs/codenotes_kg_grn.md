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
