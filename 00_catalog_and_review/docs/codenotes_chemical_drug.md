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
