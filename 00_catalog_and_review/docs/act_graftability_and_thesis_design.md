# Adaptive-Computation-Time (ACT) Graftability & Thesis Design

*Companion to `perturbation_models_review.md` and `codebase_review.md`. Compiled 2026-07-11.*

**Thesis in one sentence.** Treat an adaptive-depth model's *halting behaviour* — how many iterative
refinement rounds it needs to predict a perturbation's control-vs-perturbed endpoint — as a candidate
per-perturbation signature, and test, in order, whether it is (i) **reproducible**, (ii) **non-redundant**
with effect size, and only then (iii) **biologically interpretable** as separating direct / context-dependent /
unstable perturbations. Halting is a *computational proxy for response complexity*, explicitly **not** biological
time or causal depth.

This document answers the engineering question the thesis rests on: **which existing perturbation model can host
a learned-halting refinement loop, where exactly does that loop attach in the code, and how do the model's internal
quantities map onto the per-perturbation signature you defined?** It then gives an honest, staged build plan with
the reproducibility and anti-confound guardrails the thesis demands.

> **Scope & honesty note.** This is a *design and feasibility* document grounded in the actual repositories, not a
> trained model or an empirical result. Every insertion point cites code that was read (see `codebase_review.md`).
> No performance is claimed. The hardest scientific risk — that halting collapses to an effect-size proxy — is
> treated as the primary hypothesis to *falsify*, per the thesis's own framing.

---

## 1. What ACT actually requires, and why most models can't host it

Adaptive Computation Time (Graves, 2016) needs three things in the forward pass:

1. **An unrollable core** — a computation step you can repeat a variable number of times (a recurrent update, a
   message-passing round, a denoising/ODE step), where "run it again" is *semantically meaningful* (each repeat
   refines the same quantity toward a fixed point or target).
2. **A halting head** — a small learned module that reads the running state after each step and emits a halting
   probability; cumulative halting decides when to stop.
3. **A ponder cost** — a penalty on the expected number of steps, so the model is pressured to stop early and the
   step count becomes an informative, learned quantity rather than a constant.

The **"semantically meaningful"** clause is what disqualifies most perturbation models. A feed-forward
autoencoder's decoder depth is not a refinement axis — layer 3 is not "a better version of" layer 2's output; it
is a different transformation. Running it "more" is undefined. This is why 20 of 43 models rank LOW or LOW–MEDIUM (16 LOW + 4 LOW–MEDIUM): adding ACT
would mean *inventing* an iterative core (an architecture change), not grafting a head onto an existing one.

Two model classes pass the clause cleanly:

- **Generative transport (diffusion / flow / neural-OT):** each step provably moves the sample toward the target
  distribution. "More steps → closer" is exactly the semantics ACT wants. **This is the cleanest fit for the
  thesis**, because "denoising/ODE steps to converge" is a per-perturbation scalar with an obvious complexity
  reading.
- **Graph message-passing:** each round propagates the perturbation signal one more hop across the regulatory/GO
  graph. "More hops → more of the network has responded" is a defensible complexity reading, though it conflates
  *graph distance* with *response complexity* (a caveat the thesis must handle).


---

## 2. Mapping your per-perturbation signature onto ACT mechanics

The thesis assigns each perturbation a five-part signature. Here is how each part is *computed* from an
adaptive-depth model, and the confound each one carries — this table is the analytical core of the thesis.

| Signature component | ACT / model quantity that realizes it | How to compute it per perturbation | Primary confound to regress out |
|---|---|---|---|
| **Response shift** | Magnitude of predicted Δ (perturbed − control) | L2 / L1 norm of the model's output delta, or E-distance(control, predicted) | This *is* effect size — it is the covariate, not a discovery |
| **Refinement rounds** | Expected halting step count *N* (ACT ponder count; or diffusion/ODE steps to convergence) | Mean halting step over cells of that perturbation (see per-model readout §3) | Must be shown ≠ monotone function of response shift |
| **Nonlinear-correction magnitude** | How much the iterative steps *change* the prediction vs a 1-step / linear readout | ‖prediction_N − prediction_1‖ (extra work beyond the first step) | Cell count / SNR (noisy perts may need more correction trivially) |
| **Halt confidence** | Sharpness/entropy of the halting distribution (or predicted-loss head, e.g. STATE's `ConfidenceToken`) | Entropy of per-step halt probabilities, or variance of step count across cells | Number of DE genes; library depth |
| **Stability** | Reproducibility of the above across guides / donors / seeds | SD or ICC of each component across guides, donors, random seeds, datasets | Technical batch; must be *high* for a usable signature |

**The non-redundancy test, made concrete.** For each candidate signature component *s* and external label *y*
(e.g. essentiality, pathway class), fit `y ~ effect_size + n_DE_genes + n_cells + essentiality + technical_covars`
first, take residuals, then test whether *s* adds predictive value for the residual. If it does not, *s* is an
effect-size proxy and the negative result is the finding (this is the thesis's own gate, and it is the correct one).

**A subtlety the thesis already anticipates and the code confirms.** "Refinement rounds" is only meaningful if the
iterative core refines *the response*, not something orthogonal. Two failure modes to guard against, both visible in
the code review:

- *Molecule-graph iteration ≠ response iteration.* In cycleCDR the unrollable loop is the **drug** GAT (molecular
  graph), not the cell-response path — halting there measures molecular-embedding convergence, which is off-target.
- *Graph-distance iteration ≠ complexity.* In GEARS/TxPert/PDGrapher the message-passing rounds refine along the
  GO/PPI/GRN graph. Halting later can mean "the perturbed gene is graph-distant from the responsive genes," which is
  partly a property of the *graph*, not the perturbation's biology. Diffusion/flow step-count does **not** have this
  problem, which is the main reason it ranks first below.


---

## 3. ACT-graftability matrix (all 43 models)

Ranked by graftability tier. "Insertion point" cites the file/class where a halting head + refinement loop would
attach (full reasoning in `codebase_review.md`). Effort is engineering effort assuming the repo's training setup.

| Model | Family | Tier | Iterative core (the axis to halt over) | Insertion point | Effort |
|---|---|---|---|---|---|
| **AIDO.Cell perturbation (Foundation** | Foundation | HIGH | STRONGEST in this family. Diffusion sampling_timesteps (DDIM loop), flow-matching ODE solver steps, Schrodinger-bridge SDE steps, and GNN num_layers - | GaussianDiffusion1D.ddim_sample / p_sample_loop in train_generative/diffusion.py (wrap 'for time,time_next in time_pairs' with a h | LOW-MEDIUM  |
| **AlphaCell** | OT | HIGH | paper-only: latent flow-matching ODE integrated by EULER from t=0->1 (explicit multi-step integrator; world-model 'rollout' of cellular dynamics). Ste | Prospective only (no code): the latent Euler ODE integration loop of the Universal State Transition flow field; a halting head wou | HIGH |
| **CellFlow** | OT | HIGH | YES -- inference solves dx/dt=v_theta(t,x,cond) from t0=0 to t1=1 via diffrax.diffeqsolve inside _otfm.py::_get_predict_fn.solve_ode. Defaults (_predi | In src/cellflow/solvers/_otfm.py::_get_predict_fn, replace diffrax.diffeqsolve with a custom Euler/step loop over solve_ode and ad | LOW-MEDIUM |
| **PDGrapher** | KnowledgeGraph | HIGH | YES (KEY): GNN message-passing depth n_layers_gnn -- explicit 'for conv, bn in zip(self.convs, self.bns)' loop in GCNBase.from_node_to_out. Paper/expe | src/pdgrapher/_models.py GCNBase.from_node_to_out -- add per-step halting head on pooled node state + ponder cost to the 'for conv | MEDIUM |
| **SequenTx (AlphaTherapy)** | Transfer | HIGH | YES -- explicit episodic loop. Step count set by self.max_step_number (config) and terminated by self.done=self.env_step>=self.max_step_number in ccl_ | Replace fixed termination in scripts/gym_cell_model/gym_cell_model/envs/ccl_env_cpd.py (CCLEnvCPD._reward / single_action_step, th | LOW-MEDIUM |
| **TxPert** | KnowledgeGraph | HIGH | YES (KEY): two explicit hand-unrolled message-passing loops. basic_gnn.py GNN.forward: 'for idx, layer in enumerate(self.gnn_layers): x = layer(x, edg | gspp/models/pert_models/basic_gnn.py GNN.forward ('for idx, layer in enumerate(self.gnn_layers)') and/or gspp/models/pert_models/m | MEDIUM |
| **scDiff** | OT | HIGH | YES -- canonical DDPM reverse chain in scdiff/model.py:728 p_sample_loop: for i in reversed(range(0,t_start+1)): x,vlb=p_sample(...). Step count = t_s | Wrap the for-i-in-reversed loop in ScDiff.p_sample_loop (scdiff/model.py ~line 748): add a halting head reading per-step predicted | LOW-MEDIUM |
| **GEARS** | KnowledgeGraph | MEDIUM-HIGH | YES (KEY): two explicit hand-unrolled message-passing loops in GEARS_Model.forward -- 'for idx,layer in enumerate(self.layers_emb_pos)' (co-expr) and  | gears/model.py GEARS_Model.forward -- wrap the 'for idx, layer in enumerate(self.sim_layers)' loop (and/or layers_emb_pos) with a  | MEDIUM |
| **PertAdapt (Condition-Sensitive Ada** | Foundation | MEDIUM-HIGH | GO-GNN loop 'for layer in self.sim_layers' runs num_go_gnn_layers message-passing rounds -- unrollable message-passing depth. PertAdapterNew is a sing | GEARS_Model_Pert_Adapter_New_aido.forward in AIDOCell/PertAdapter/gears/model_new.py -- wrap the 'for idx,layer in enumerate(self. | MEDIUM |
| **STATE (State Transition + State Em** | Foundation | MEDIUM-HIGH | Backbone is a fixed L-layer GPT2 stack (n_layer config); one forward, no runtime-varied loop. Step-count = transformer depth (config). BUT residual fo | StateTransitionPerturbationModel.forward in src/state/tx/models/state_transition.py -- wrap body (~lines 396-540) in for step in r | MEDIUM  |
| **XPert** | Chemical | MEDIUM-HIGH | YES — stacked transformer whose block sequence is set by a config string. AttnEncoder.forward iterates 'for layer_type in self.layers:' over crossEnco | In AttnEncoder.forward (models/model_XPert.py), after each block updates cell_embed insert a halting MLP h_t=sigmoid(W.pool(cell_e | MEDIUM  |
| **AIDO.Cell (FM backbone)** | Foundation | MEDIUM | transformer FM, fixed depth; no runtime loop ALONE. HIGH when paired with generative heads (AIDO.Cell perturbation repo). | n/a as backbone; graft on the diffusion/flow head or PertAdapt adapter. | MEDIUM |
| **CellOT** | OT | MEDIUM | Single-shot: T(x)=grad g(x), one autograd.grad call, no loop. Only step-like knob = scalar dosage interpolation along the displacement in transport.py | Wrap transport_cellot() in cellot/transport.py: replace one-shot g.transport(x) with a K-step displacement/residual loop; add a ha | MEDIUM |
| **Geneformer (FM backbone)** | Foundation | MEDIUM | fixed transformer depth; no runtime loop. | transformer encoder stack (HF model). | MEDIUM-HIGH |
| **Stack (In-Context Learning of Sing** | Foundation | MEDIUM | _run_attention_layers loops over n_layers=6 TabularAttentionLayers (homogeneous repeated block). Step-count = n_layers (constructor arg). Classic repe | StateICLModelBase._run_attention_layers in src/stack/models/core/base.py -- replace 'for layer in self.layers' with a shared-layer | MEDIUM-HIGH  |
| **UNAGI** | Transfer | MEDIUM | TWO genuine multi-step loops: (a) training EM loop `for iteration in range(max_iter)` in UNAGI_tool.py::run_UNAGI; (b) inference PPI-BFS `while` loop  | Replace `for iteration in range(start_iteration, self.max_iter)` in UNAGI/UNAGI_tool.py::run_UNAGI with a convergence-gated `while | MEDIUM |
| **X-Cell** | OT | MEDIUM | Stub API exposes n_diffusion_steps: int = 4 (explicit iterative masked-diffusion decode rounds over the cell set) in src/xcell/model.py predict() sign | Prospective: the (unreleased) masked-diffusion decode loop implementing n_diffusion_steps in src/xcell/model.py::XCell.predict; a  | HIGH |
| **scFoundation (FM backbone)** | Foundation | MEDIUM | Performer/MAE stack, fixed depth; no runtime loop. | n/a as backbone; graft on the adapter/head (see PertAdapt). | MEDIUM |
| **scGPT (FM backbone)** | Foundation | MEDIUM | fixed transformer depth (config); no runtime loop. | transformer encoder stack in scgpt/model/model.py / generation_model.py. | MEDIUM-HIGH |
| **scGenePT** | Foundation | MEDIUM | scGPT TransformerEncoder is a fixed nlayers stack -- one forward; step-count = transformer depth (config), not a runtime loop. | transformer_encoder call inside scGenePT._encode/forward (models/scGenePT.py) -- wrap encoder layers in a shared-block halting loo | MEDIUM-HIGH  |
| **CONCERT** | Transfer | LOW-MEDIUM | No refinement recurrence controlling single-perturbation prediction quality. The two `for l in range(GP_dim)` loops are over latent dimensions; the nu | Wrap the decode in CONCERT.forward (src/concert_map.py, after z=latent_dist.rsample(); h=self.decoder(z)) in a `for t in range(T)` | HIGH  |
| **STAMP** | KnowledgeGraph | LOW-MEDIUM | Fixed 3-stage semantic cascade (subtask-1 DEG -> subtask-2 direction -> subtask-3 magnitude), NOT a repeated/recurrent block. Step count fixed at 3 an | none native for a per-perturbation halting loop; would require re-architecting the heterogeneous cascade (stamp/Modules.py TaskCom | HIGH |
| **cycleCDR** | Chemical | LOW-MEDIUM | Weak/borderline. (i) Cycle is a fixed 2-hop A->B->A composition, not variable-length. (ii) The GAT drug encoder does a fixed number of message-passing | Either (a) iterate encoderG_A->(+drug)->decoderG_A in netG_A (cycleCDR/model/model.py) as a refinement loop with a halting head on | MEDIUM |
| **scLAMBDA** | Foundation | LOW-MEDIUM | none unrollable -- encoders/decoders are fixed MLPs; only 'steps' are training epochs. No recurrent/diffusion/message-passing loop in forward. | Wrap the z+s->Decoder_x step in Net.forward (sclambda/networks.py) in a refinement loop with a halting MLP on the latent; retrain. | MEDIUM-HIGH |
| **CODEX** | Chemical | LOW | None. The 'for t in range(num_treatments)' loop is a masked scatter-add over treatment CHANNELS (each iteration a different weight matrix), not a repe | Wrap encoder->latent_rep->decoder in CODEXReconstruction.forward (codex/CODEX_reconstruction.py) in a shared-weight recurrent bloc | HIGH |
| **Cell Painting CNN (DeepProfiler)** | OT | LOW | None -- single forward pass through a fixed EfficientNet (GlobalAveragePooling2D -> Dense). No recurrence, diffusion, or solver. Depth is architectura | None natural -- would require replacing the EfficientNet with an iterative model | HIGH |
| **CellCap** | KnowledgeGraph | LOW | none with variable step count. The multi-head attention loop 'for i in range(self.n_head)' is PARALLEL heads combined by torch.max -- a single attenti | none native; would wrap the inference() attention->delta_z step (cellcap/model.py:169) into an added recurrent refinement of z --  | HIGH |
| **Cradle-VAE (CRADLE-VAE)** | Chemical | LOW | None in the neural forward pass. z = z_basal + D@(E*mask) then decoder — single-shot generative + amortized guide. No recurrence/diffusion/unrolled so | Would need to make the guide's z_basal inference iterative (e.g., iterative amortized inference / unrolled SVI) in cradle_vae/mode | HIGH |
| **GenePert** | Foundation | LOW | none (linear model; MLP variant is 2-layer feed-forward). | n/a. | HIGH / not appli |
| **IMPA** | OT | LOW | None -- Generator.forward is a single-shot encoder->decoder pass; the encode/decode 'for block in self.encode/decode' loops are a FIXED-DEPTH CNN stac | No clean insertion point | HIGH |
| **PRnet** | Chemical | LOW | None. Fixed MLP encoder/decoder; no recurrence/diffusion/ODE. Latent z and c computed once. | Iterate the z->decoder mapping in PGM.forward (models/PRnet.py) as a latent refinement loop with a halting head on z; net-new mach | HIGH |
| **PrePR-CT (Cell-Type-Specific-Graph** | Transfer | LOW | NONE that is step-count-varyable. Only loop in forward() is over cell-type keys (data parallelism), not depth. Fixed 2 GNN conv layers -> single-shot. | Would wrap GNN.lin_predict in model.py in a `for step in range(T)` residual-update loop with a per-step halting MLP on the pooled  | HIGH  |
| **SAMS-VAE** | KnowledgeGraph | LOW | none with a variable step-count. Inference is amortized (single encoder forward); decoder is a fixed MLP; n_particles = Monte-Carlo particles, not seq | none native; a refinement loop would wrap the guide's z_basal_encoder (sams_vae/models/sams_vae/guides/correlated_normal_guide.py) | HIGH |
| **Scouter** | Foundation | LOW | none -- pure feed-forward MLP, no repeated/recurrent block. | Would have to make generator recurrent in scouter/_model.py (ScouterModel.forward); not a natural fit. | HIGH  |
| **TranSiGen** | Chemical | LOW | None. Two parallel fixed-depth VAEs + one linear latent bridge (mu_z2Fz1); no recurrence/diffusion. | Replace the single mu_z2Fz1 map in TranSiGen.forward (src/model.py) with a recurrent latent-refinement z2^{(t+1)}=z2^{(t)}+g(z2^{( | HIGH |
| **biolord** | KnowledgeGraph | LOW | none that varies a step count. Decoder is a fixed decoder_depth=4 feed-forward MLP stack (not a repeatable refinement/message-passing block); single e | none native; would have to wrap generative() (src/biolord/_module.py) in an added iterative latent-refinement loop -- an architect | HIGH |
| **chemCPA** | Chemical | LOW | None. Encoder/decoder are fixed-depth MLPs; autoencoder_depth is a static hyperparameter, not an unrolled-per-input loop. No recurrence/diffusion/ODE  | Wrap the single 'latent_treated = latent_basal + drug_embedding' update in ComPert.predict (chemCPA/model.py) as a shared-weight r | HIGH |
| **scELMo** | Foundation | LOW | none of its own (it is not the predictor). | n/a (embedding provider). | n/a |
| **scRank** | Transfer | LOW | A network-propagation hyperparameter n_hop (default 2) exists in rank_celltype(), BUT code hard-caps it 1-3 (if(n_hop>3 / n_hop<1) stop) and it only t | Conceptually generalize the n_hop/simple_sum/multi_layer branch in R/method.R::rank_celltype into a random-walk-with-restart propa | HIGH  |
| **scVIDR** | Chemical | LOW | None. VAE encode->(latent add)->decode is single-shot; the 'for d in doses' loop just evaluates the same closed-form dose-scaled shift at several dose | Iterate the 'latent_ctrl + delta -> decoder' step in VIDR.predict (vidr/vidr.py) as a refinement loop with a halting head on the l | HIGH |
| **CausalGRN** | KnowledgeGraph | NOT APPLICABLE | Perturbation prediction is a SINGLE closed-form linear SOLVE (solve(I - B_UU, driving_force)); the fixed point is obtained directly, NOT by unrolled i | R/expression_prediction.R .impute_deltas -- swap base::solve for an iterative Neumann-series/power-iteration loop and record steps | LOW  |
| **PS (Perturbation-response Score)** | Chemical | N/A | N/A / none — statistical scoring pipeline, no neural forward pass to unroll. | N/A — no neural forward pass | N/A |
| **scPert (Multi-modal LLM-Knowledge ** | Foundation | UNKNOWN | paper-only: unknown -- no code to inspect. | n/a (no code). | UNKNOWN |


---

## 4. Ranked backbone recommendations for the adaptive-depth thesis

The ranking weighs four criteria: (a) **semantic cleanliness** — does step-count mean "response complexity" without
a confound; (b) **effort** to add learned halting; (c) **harmonized multi-dataset training** feasibility (the thesis
trains on several harmonized datasets); (d) **maturity** of the public code.

### Tier A — recommended primary backbones

**A1. scDiff — conditional diffusion  ·  https://github.com/OmicsML/scDiff**
The single best fit. The reverse diffusion chain is the canonical unrollable loop, and it carries a *native
convergence signal* (change in predicted x₀, posterior variance, per-step VLB) with no graph-distance confound. "Steps
to converge" is a clean per-perturbation complexity scalar. Add a halting head on the running x_t and a ponder cost on
step count. General single-cell framework → tractable to train on harmonized datasets. **Recommended as the main
experimental backbone.**

**A2. CellFlow — flow matching  ·  https://github.com/theislab/cellflow**
Nearly tied with scDiff and arguably the *fastest path to a first result*: its adaptive ODE controller **already
emits a data-dependent per-instance step count**, so you can extract a "refinement rounds" readout *before writing any
halting code* — an ideal Step-0 diagnostic to check whether step-count correlates with effect size at all. Then add a
learned halting head for the full ACT version. Flow matching trains stably and scales.

**A3. AIDO.Cell perturbation heads — diffusion / flow / Schrödinger bridge  ·  https://github.com/genbio-ai/foundation-models-perturbation**
Three generative heads (`GaussianDiffusion1D`, `FlowMatching1D`, `SchrodingerBridge1D`) on a strong foundation-model
backbone, all with explicit step loops. Highest-capacity option and lets you test whether the halting signature is
*backbone-invariant* (same perturbation, different generative head → same refinement-rounds ranking?) — a built-in
reproducibility check. Heavier to train.

### Tier B — recommended for the graph-based cross-check

**B1. GEARS  ·  https://github.com/snap-stanford/GEARS** and **B2. TxPert  ·  https://github.com/valence-labs/TxPert**
Use these as a *mechanistically different* second backbone to test whether the signature is architecture-invariant
(a core reproducibility claim). Both hand-unroll GO/PPI message passing. **Caveat from the code:** GEARS ships at
message-passing **depth 1** (`num_go_gnn_layers=1`), so you must *deepen the stack first* before there is anything to
halt over. TxPert already runs depth-4 GAT stacks and its `GatedCombiner` computes per-layer sigmoid gates that are
close in spirit to a halting gate — the lightest graph graft. Remember the *graph-distance confound* from §2 when
interpreting graph halting.

**B3. PDGrapher  ·  https://github.com/mims-harvard/PDGrapher**
Message-passing depth already swept {1,2,3} in their experiments, so adaptive depth is a natural drop-in; but it
solves the *inverse* (which intervention reverses a state) problem, so it fits a different thesis question than
endpoint-response complexity. Use only if the inverse framing becomes relevant.

### Tier C — the "adaptation" reference, not an ACT host

**PertAdapt  ·  https://github.com/BaiDing1234/PertAdapt** — the only genuine per-condition adaptation model. Not your ACT backbone, but
the reference implementation for the *condition-adaptation* half of the landscape and a strong candidate if you later
want per-perturbation *parameter* adaptation to complement per-perturbation *depth* adaptation.

### Explicitly not recommended as backbones (but essential as baselines)

**GenePert (ridge on GenePT embeddings), Scouter, biolord, chemCPA** — single-shot, no iterative core. Their value to
the thesis is as **effect-size baselines and covariate generators**: their predicted Δ magnitude and #DE genes are
exactly the quantities you regress out in the non-redundancy test. Include at least GenePert (simplest) and a mean/linear
baseline per the benchmark critiques (§7 of the review).

### One-line decision

> **Start with CellFlow** (free step-count readout for the Step-0 confound check), **commit to scDiff** as the primary
> learned-halting backbone, and **cross-check on TxPert** (architecture-invariance). This trio gives generative-clean
> halting, a falsification-first diagnostic, and a mechanistically independent replication.


---

## 5. Exact code integration points

Pseudocode grounded in each repo's actual structure (see `codebase_review.md` for the class/file citations). These
are *illustrative graft sketches*, not drop-in patches — adapt to the current API.

### 5.1 scDiff — halting head on the reverse-diffusion loop (primary)

The reverse chain `for t in reversed(range(T)): x = p_sample(x, t, cond)` is the unrollable core. Add a halting head
that reads the running `x_t` (and `t`) and a ponder cost:

```python
# in the sampler loop (conceptually scDiff's reverse process)
halt_cum = 0.0; ponder = 0.0; x = x_T
for t in reversed(range(T)):
    x = p_sample(x, t, cond)                      # existing denoising step
    h = torch.sigmoid(halting_head(x, t_emb(t)))  # NEW: per-step halt prob in [0,1]
    halt_cum = halt_cum + h
    ponder = ponder + (1 - halt_cum).clamp(min=0) # Graves ponder cost
    if halt_cum >= 1 - eps:                        # ACT stop
        n_steps = t; break
# loss = recon_loss(x, target) + tau * ponder     # tau tunes eagerness to stop
# PER-PERTURBATION READOUTS: n_steps (refinement rounds),
#   ||x_final - x_after_1_step|| (nonlinear-correction magnitude),
#   entropy of per-step h (halt confidence)
```
*Convergence-based variant (no learned head, good Step-0 diagnostic):* halt when `||x_t - x_{t-1}|| < delta`; the
stopping index is "refinement rounds" with zero new parameters.

### 5.2 CellFlow — read the free step-count first, then add halting

CellFlow integrates a learned velocity field with an adaptive ODE solver. The adaptive controller **already**
returns a per-instance step count — extract it before writing any ACT code:

```python
# Step-0 diagnostic (no new params): does step-count even correlate with effect size?
traj = ode_solve(v_field, x0, cond, atol=a, rtol=r, return_stats=True)
n_func_evals = traj.stats.n_steps          # per-instance "refinement rounds", FREE
# ... aggregate per perturbation, then run the §6 reproducibility + non-redundancy tests.
# Full ACT version: replace fixed-grid integration with a halting head on the running state x_t.
```
This is why CellFlow is the fastest path to the *first* thesis result: you can test the central confound
(rounds vs effect size) on day one, before committing to a training run.

### 5.3 TxPert — repurpose the GatedCombiner as a halting gate (graph cross-check)

TxPert's `MGAT.forward` loops `for i, layer in enumerate(self.layers): h = layer(h, edge_index)` and fuses layers
with a learned `GatedCombiner` sigmoid gate. Convert fixed depth into learned adaptive depth:

```python
# in MGAT.forward (gspp multi_graph.py) — unrolled GAT stack
halt_cum = 0.0; ponder = 0.0
for i, layer in enumerate(self.layers):
    h = layer(h, edge_index)
    g_i = self.gated_combiner.gate(h)             # EXISTING per-layer sigmoid gate
    halt_cum = halt_cum + g_i.mean(...)           # reinterpret as halting mass
    ponder = ponder + (1 - halt_cum).clamp(min=0)
    if halt_cum >= 1 - eps: break                 # adaptive message-passing depth
# NB: deepen GEARS first (num_go_gnn_layers defaults to 1) before this is meaningful.
```

### 5.4 STATE — reuse the existing ConfidenceToken as a halting signal (lightest "graft on a released model")

STATE predicts a residual (`pred = basal + Δ`) and already has a `ConfidenceToken` that outputs a scalar predicted
loss. Wrap the forward in a re-feed loop and halt on confidence:

```python
# StateTransitionPerturbationModel.forward (src/state/tx/models/state_transition.py)
x = basal
for step in range(max_steps):
    out, conf = self.forward_once(x, pert)        # existing residual forward + ConfidenceToken
    x = out                                       # re-feed prediction as next basal (Δ refinement)
    if conf > thr: break                          # ConfidenceToken AS halting head
# ponder = step; n_steps is the per-perturbation refinement-rounds readout
```


---

## 6. Reproducibility, non-redundancy & leakage guardrails

These operationalize the thesis's own test order and import the protected-baseline discipline from `autoresearch-bio`.
They are not optional add-ons; the benchmark critiques (§7 of the review) show that skipping them is *the* failure mode
of this subfield.

### 6.1 Step 0 — protected baselines *before* any adaptive-depth model (non-negotiable)

Register, on every harmonized dataset and split, the performance of: (a) **predict-control** (no change), (b)
**predict-mean-perturbed**, (c) a **linear/ridge** model on perturbation embeddings (GenePert-style), and (d) a
**fixed-depth** version of your chosen backbone (scDiff at fixed T). No adaptive-depth result is interpretable except
relative to these. This is the direct answer to Ahlmann-Eltze et al. — you *lead* with the baseline the critics use.

### 6.2 Leakage pre-flight (before training)

- **Split manifest.** Freeze train/val/test splits per generalization axis and *name the axis*: unseen guide,
  unseen gene, unseen combination, unseen donor, unseen dataset. These are different difficulties; never pool them.
- **No perturbation on both sides.** A perturbation (and, for CRISPRi, all its guides) lives in exactly one split.
- **Donor/batch disjoint** where the claim is cross-donor stability.
- **The halting head must never see the test split during selection.** Ponder-weight (τ) and halt-threshold are
  hyperparameters — tune on validation only. Iterated tuning against test is exactly the "iterated selection" leak.

### 6.3 The three tests, in the thesis's order (a gate, not a menu)

1. **Reproducibility gate.** Each signature component must be stable across **seeds, guides, donors, and independent
   datasets** (report ICC / rank correlation). *If refinement-rounds is not reproducible across seeds, stop —* there
   is no signature. This gate comes first because an irreproducible signal cannot be non-redundant or biological.
2. **Non-redundancy gate.** Regress out effect size, #DE genes, cell count, essentiality, and technical covariates;
   test whether the residual signature predicts external labels. *If it collapses to an effect-size proxy, that
   negative result is the finding* (and a clean, publishable one under the §7 critique framing).
3. **Biological interpretation — only if 1 and 2 pass.** Test whether the signature separates putative direct /
   context-dependent / unstable perturbations against **independent** annotations (not derived from the model).

### 6.4 No-regression / anti-confound checks specific to halting

- **Effect-size decorrelation plot** is the headline diagnostic: refinement-rounds vs response-shift, per perturbation.
  A tight monotone line = the signature is just effect size. You *want* structured scatter with reproducible residuals.
- **Guard the off-target iteration axes** from §2: confirm halting reads the *response* path, not a molecular-graph
  (cycleCDR) or purely graph-distance (GEARS/TxPert) axis. The generative backbones (scDiff/CellFlow) are chosen
  precisely to avoid this.
- **Ponder-cost sensitivity.** Sweep τ; the *ranking* of perturbations by refinement-rounds should be stable across a
  reasonable τ range. If the ranking flips with τ, the signature is an artifact of the penalty, not the biology.

---

## 7. Honest, staged build plan

Each stage has an explicit **stop/kill condition**. The plan is designed so the cheapest, most likely-to-kill
experiment runs first (falsification-first), consistent with the thesis's negative-result-is-a-finding stance.

**Stage 0 — Free confound check on CellFlow (days, no training).**
Run CellFlow's adaptive ODE solver on an existing harmonized dataset; extract per-instance step counts; aggregate per
perturbation; plot refinement-rounds vs effect size and compute cross-seed stability. **Kill condition:** if step-count
is perfectly explained by effect size *and* unstable across seeds, the core hypothesis is already in doubt — reconsider
before investing in training. *(This stage exists because it can end the project cheaply, which is the point.)*

**Stage 1 — Fixed-depth baselines + leakage pre-flight (Step 0 above).**
Register protected baselines on all datasets/splits; produce `split_manifest.json` and `leakage_preflight.md`. **Gate:**
baselines reproduce published-order behaviour and splits pass leakage checks before any adaptive-depth training.

**Stage 2 — Learned halting on scDiff (primary experiment).**
Add the §5.1 halting head + ponder cost; train on harmonized datasets. Extract the full five-part signature per
perturbation. **Gate:** the *reproducibility* test (§6.3.1) across seeds/guides/donors/datasets. Stop if it fails.

**Stage 3 — Non-redundancy.**
Run the regress-out-then-test analysis (§6.3.2) against external labels. **Gate:** signature adds value beyond effect
size. A clean failure here is a complete, publishable negative result — do not p-hack past it.

**Stage 4 — Architecture-invariance replication on TxPert.**
Repeat Stages 2–3 on the deepened-GNN TxPert graft. **Gate:** the per-perturbation refinement-rounds *ranking*
correlates across the two mechanistically different backbones. This is the strongest evidence the signature is a
property of the perturbation, not the model.

**Stage 5 — Biological interpretation (only if Stages 2–4 pass).**
Test separation of direct / context-dependent / unstable perturbations against independent annotations.

**Stage 6 — Application: Marson/Pritchard primary human CD4+ T-cell CRISPRi Perturb-seq.**
Apply the validated framework to rank inhibitory targets that suppress stimulated inflammatory programs while sparing
the resting T-cell state, generating explainable per-target cards. **Precondition:** Stages 2–5 passed on the public
harmonized data first; the application is not a discovery venue for method validation.

> **Compute note.** Stages 2–4 need GPU training of diffusion/GNN models on multiple datasets — plan for a cluster,
> not a laptop. Stage 0 and the baselines (Stage 1) are cheap and where you should spend the first week.

---

## 8. Summary table — what to do with each candidate

| Role in the thesis | Model(s) | Why |
|---|---|---|
| **Primary ACT backbone** | scDiff | Clean generative halting semantics; native convergence signal; no graph-distance confound |
| **Fast Step-0 diagnostic** | CellFlow | Adaptive ODE solver already emits a free per-instance step count |
| **High-capacity / backbone-invariance** | AIDO.Cell heads | Three generative heads on an FM backbone; test signature invariance |
| **Architecture-invariance cross-check** | TxPert (deepen), GEARS | Mechanistically different (graph) halting; replicate the ranking |
| **Condition-adaptation reference** | PertAdapt | The only true per-condition adapter; complementary axis |
| **Effect-size baselines / covariates** | GenePert, mean, linear, biolord | Single-shot; supply the confounds to regress out and the benchmark-mandated controls |
| **Not applicable** | chemical single-shot AEs, image CNNs, non-neural (CausalGRN, scRank, PS) | No semantically meaningful refinement axis |

*Grounding: every model claim in this document is traceable to `codebase_review.md` (files/classes read) and the
verified rows in `perturbation_models_2024_2026.csv`. This is a design analysis; no model was trained or benchmarked
here, and no performance is claimed.*
