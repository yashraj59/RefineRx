# When Does a Perturbation Model Know Enough?

**Yash Raj**
*Built with Claude: Life Sciences 2026 — Researcher Track*
Project: **RefineRx**

---

## Abstract

Most Perturb-seq experiments measure a single endpoint — control versus perturbed — yet the biology underneath ranges from direct target effects to feedback, compensation, and context-specific regulatory cascades, so two perturbations of very different mechanistic complexity can land at responses of identical magnitude. Adaptive-computation halting (ACT/PonderNet) was invented to save compute — an easy input exits early, a hard one ponders longer. We repurpose it, to our knowledge for the first time on a perturbation-response model, as a measurement rather than a stop rule: on a frozen backbone whose eight transformer layers always execute, halting saves no compute, and the layer at which the model's endpoint estimate converges is read as a per-perturbation proxy for response complexity — not biological time or causal depth, which a single endpoint cannot recover. We ask whether this halting depth E[N] is reproducible, effect-size-independent, and informative beyond response magnitude. A magnitude-free, oracle-supervised halting head on ARC Institute's frozen STATE model clears both gates: on K562 (968 perturbations) E[N] is reproducible (split-half ρ = 0.76), effect-independent (partial ρ = −0.04 controlling #DE and cell count), and recovers its oracle target (ρ = 0.61), where a naive learned gate collapses to a per-seed constant. Across four Replogle lines E[N] reproduces within each line (mean ρ = 0.80) but does not port across cell types (ρ = 0.14). On the genome-wide Marson/Pritchard CD4+ T-cell CRISPRi screen at single-cell resolution, E[N] is reproducible within a donor and rises with stimulation (split-half ρ = 0.62 → 0.75), is donor-specific in the resting state (cross-donor ρ ≈ 0) but ports at the stimulated 48h endpoint where donors converge (ρ = 0.49), and there separates resting-sparing inflammatory-program suppressors from damaging and generic perturbations (Cliff's δ = −0.16 to −0.24). Clustered on its own, E[N] organizes the target landscape — approved and clinical drug targets concentrate in a translation/ribosome depth cluster in every line (per-cluster odds ratios 2.8–9.8) — and its per-perturbation ordering is reproduced by no network statistic (best |ρ| = 0.23), a genuinely novel descriptor; only for one incremental target-class ranking task, atop a STRING prior built from the same modules, does it add no further lift. Adaptive-depth halting is thus a reproducible, effect-independent, cell-type-specific property of a perturbation's response — recoverable from single-endpoint data and, clustered alone, a novel descriptor that organizes druggability, though not portable across contexts and non-additive with a network prior for incremental target ranking.

---

## 1. Introduction

Most Perturb-seq experiments measure a single endpoint: the transcriptome of control cells versus the transcriptome of perturbed cells. That endpoint collapses a rich temporal and mechanistic process into one contrast. A ribosomal knockdown shifts the transcriptome immediately and directly; a signalling regulator reaches a response of comparable magnitude only after feedback, compensation, and secondary regulatory cascades unfold. The endpoint cannot separate these two, so two perturbations of very different mechanistic complexity can land at responses of identical size. Response magnitude — the most common summary of a perturbation — is therefore blind to how much work the response took to produce.

We ask whether a computational correlate of that work can be recovered from the single endpoint we already collect. If a model predicts a perturbation's endpoint response by iterative refinement, how many rounds of refinement does each perturbation require before the prediction converges? Is that a stable property of the perturbation, and does it carry information beyond what response magnitude already tells us?

**From an efficiency mechanism to a measurement.** Adaptive computation was introduced to save compute: an easy input should exit early and a hard one should ponder longer [1, 2], and every subsequent adaptive-depth and early-exit method we are aware of — in language and in vision — uses halting to reduce inference cost. We invert that use. Our backbone is a frozen, pretrained model whose eight transformer layers always execute, so halting saves no compute at all; instead we read the depth at which the model's endpoint estimate converges as a per-perturbation measurement of response complexity. To our knowledge this is the first time a learned-halting mechanism has been applied to a perturbation-response model, and the first time halting depth is treated as a biological readout rather than a stop rule. The question is then whether that readout is a stable, non-redundant property of the perturbation, or merely a re-encoding of something we already measure.

**What the signature is, and is not.** The learned halting depth E[N] is a proxy for response complexity — specifically, the layer at which a fixed pretrained stack's endpoint estimate converges. It is explicitly *not* biological time: the data are a single endpoint, not a time course. It is *not* causal cascade length: a single endpoint cannot identify the causal graph. It is a model-internal quantity, and its value as a descriptor rests entirely on whether it survives the tests a new descriptor must pass — reproducibility, effect-size independence, and non-redundancy with what we already know.

**Contributions.** Beyond the specific results, the paper's framing contribution is to repurpose adaptive-depth halting from an inference-efficiency mechanism into a per-perturbation biological measurement, and to test — with the rigour a new descriptor demands — whether that measurement survives. This paper reports, in narrative order:

1. An **identifiability wall** (§3). Grafting learned halting onto models we trained ourselves fails in one of two ways: a free gate collapses to a constant, or a magnitude-anchored gate becomes an effect-size proxy. The two desiderata — reproducibility and effect-independence — pull against each other for any model whose depth is a transient of our own training.

2. A **working method** (§4), AdaptiveStateRefine, that clears both gates only on a frozen, well-fitted foundation model, using a magnitude-free per-round target and per-perturbation oracle stopping-round supervision.

3. A **reproducible, effect-independent depth signature on K562** (§5.1): split-half ρ = 0.76, partial ρ(effect | #DE, cells) = −0.04, oracle recovery ρ = 0.61.

4. A **cell-type-specificity result** (§5.2): E[N] reproduces within each of four Replogle lines (mean ρ = 0.80) but does not port across cell types (ρ = 0.14), with a three-tier decay that locates the dependence in the biology rather than in a training artifact.

5. A **single-cell T-cell application** (§5.8): in the genome-wide Marson/Pritchard CD4+ T-cell CRISPRi screen, E[N] is reproducible within a donor, ports across donors only at the fully stimulated endpoint where donors converge, and there separates resting-sparing inflammatory-program suppressors from damaging perturbations.

6. A **two-sided drug-target result** (§5.5–§5.6). Clustered on the depth signature, approved and clinical drug targets concentrate in a translation/ribosome depth cluster in every line (per-cluster odds ratios 2.8–9.8), and the descriptor is non-redundant with network topology — no network statistic reproduces its per-perturbation ordering (best |ρ| = 0.23). It does not, however, add incremental lift over a STRING-network prior built from the same functional modules for one target-class ranking task. Non-redundancy as a descriptor and non-additivity for that one task are different claims; only the latter is negative.

Where a design choice is a direct response to a negative result, we say so; the negatives motivated the method and they are reported first.

---

## 2. Related work

**Adaptive computation as efficiency.** Adaptive Computation Time [1] and PonderNet [2], along with early-exit and layer-skipping methods in language and vision, all use a learned halting signal to reduce inference cost — spending fewer steps on easier inputs. In every case the halting decision is a means to save compute, and the depth at which a model halts is an efficiency artifact, not a quantity of interest in its own right. We are not aware of prior work that reads a learned halting depth as a measurement of the input.

**Perturbation-response models.** A large and fast-moving family of models predicts single-cell perturbation responses, spanning foundation/adaptation models, knowledge-graph and gene-regulatory-network models, chemical and drug models, optimal-transport and flow/diffusion models, and transfer/spatial models. A survey of 43 such models from 2024–2026 (Figure 1) shows that most are single-shot: they map a perturbation to a predicted endpoint in one pass and would require architectural changes to host a halting loop. Eleven expose a semantically meaningful iterative core (rated HIGH or MEDIUM-HIGH graftability) — including ARC Institute's STATE, scDiff, TxPert, CellFlow, GEARS, PDGrapher, and AIDO.Cell prediction heads. We use STATE (the ST-SE-Replogle variant) as the frozen backbone because its iterative structure is meaningful, it is well fitted, and it is deployed frozen — the regime in which our reframe is well posed.

**Layer depth in single-cell foundation models.** Reading intermediate rather than final layers is an active question for single-cell foundation models. Concurrent work shows that the most informative layer is task- and context-dependent, shifting by up to 96% across T-cell activation states for perturbation modelling, with first-layer embeddings sometimes beating all deeper layers in quiescent cells [16]. Our aim differs in kind: rather than selecting one optimal extraction layer per task by sweeping downstream performance, we derive a per-perturbation *learned* halting depth and validate the depth signal's own properties — reproducibility, effect-independence, and portability. Where the two meet is independent corroboration: our finding that halting depth ports only where cell states converge (§5.8) recovers their context-dependence result from a different methodological angle.

---

## 3. The identifiability wall: negatives that shaped the method

Before the working method, we report the sequence of failures that produced it, because the failures are informative and they directly motivate each design choice in §4. We grafted learned halting onto several hosts — diffusion [10], graph message-passing [11, 13], and refiners we built ourselves — and, in every case, the learned depth landed in one of two failure modes. Table 1 summarises the outcomes against the two-corner test (reproducible *and* effect-independent).

**Three rulings.** Distilling the failures yields three rulings that constrain any method that hopes to clear both gates.

1. **A free learned gate over an expressive predictor collapses to constant depth.** A PonderNet halt head over pooled cell features, given every advantage (convergence-feature inputs, small-λ init, zero KL prior, entropy bonus, tens of epochs, multiple seeds), learns a *single global constant per seed*. On single-endpoint data the aggregate per-round loss is nearly flat, so the gradient toward any particular stopping round is weak, and the gate settles on whichever constant its seed drifts to. There is no per-perturbation signal to key on unless the gate is given one.

2. **Anchoring the halt loss to response magnitude makes depth reproducible but turns it into an effect-size proxy.** When the per-round target is tied to how large the response is, depth becomes highly reproducible — and highly correlated with effect size (ρ ≈ −0.7 to −0.84). Reproducibility is bought by making E[N] a restatement of magnitude, which adds nothing.

3. **Removing magnitude removes the proxy but makes depth irreproducible.** A magnitude-free refiner we trained from scratch is non-redundant with effect size but does not reproduce across seeds (ρ = 0.14). The signature exists, but it is a transient of the particular fit rather than a property of the perturbation.

The two desiderata therefore pull against each other for any model whose depth is a transient of our own training. The wall falls only when the depth is read off a model whose fit is fixed and well-behaved — the frozen foundation backbone of §4.

---

## 4. Method: AdaptiveStateRefine

### 4.1 The frozen backbone refines, then degrades

We read each of the eight layers of the frozen STATE backbone through the same frozen decoder — a logit-lens readout of the residual stream — and measure the per-layer direction error D_k = 1 − cos(û_k, u) between the layer-k endpoint estimate û_k and the true response direction u. Over 924 K562 perturbations (Figure 2a), the estimate refines monotonically through the early stack (D_k falls from 0.909 at layer 1 to 0.881 at layer 6), then the final two layers move the readout *away* from the response direction as the backbone specialises for its own pretraining objective. Refinement is real but not monotone across the full stack.

Critically, the layer of closest approach — arg min_k D_k — is per-perturbation and spans all eight layers (Figure 2b). E[N] therefore draws on the full range of the stack rather than collapsing to a single global round. This per-layer convergence depth is a property of the frozen model, not a learned decision; we report it as an independent, parameter-free reference against which the learned depth is validated.

### 4.2 Depth as a measurement, not compute-time

Because the backbone is frozen and all eight layers always execute, E[N] varies no compute and saves none. This is the load-bearing difference from adaptive computation as it is usually deployed: we are not exiting early to save FLOPs. E[N] is a *layerwise refinement-convergence depth*, not adaptive compute-time in the Graves/PonderNet sense. It is read as a per-perturbation proxy for response complexity, and its meaning is exactly the depth at which a fixed, gradually refining stack's endpoint estimate converges.

### 4.3 AdaptiveStateRefine: learned halting on the frozen backbone

The learned model adds a halting head on top of the frozen backbone. Three design choices — each a direct response to the identifiability wall — make learned halting work where the naive gate collapsed.

1. **A refinement token (per-perturbation probe).** Instead of pooling cell features, a dedicated learnable token is appended to the cell sentence; because the backbone is position-free, the token appends cleanly. Its hidden state at each layer feeds the halt head, so the halting representation is built per-perturbation by the transformer itself rather than averaged from cell features. Diagnostically, this token's hidden state varies across perturbations at layers 1–6 (pairwise cosine 0.80–0.83), confirming a per-perturbation signal exists for the gate to key on — the missing ingredient behind ruling 1.

2. **A LayerNorm-unsaturated gate (load-bearing).** The frozen refinement-token hidden states have large norm; without normalisation, the pre-sigmoid halt logits have standard deviation ∼100+, the sigmoid saturates to exactly 0/1, hazards become binary, stop-mass becomes one-hot, and depth collapses to a constant. A LayerNorm on the halt-head input keeps logits in a learnable range. This single change turns the halt confidence from a pinned 1.000 into genuine soft distributions (mean 0.28).

3. **Oracle stopping-round supervision.** The per-round target is the magnitude-free per-layer direction error of §4.1, and the halt head is supervised toward the per-perturbation oracle stopping round r\* derived from the model's own convergence. A magnitude-free target makes depth impossible to reduce to an effect-size proxy by construction; the oracle target makes a magnitude-free depth learnable reproducibly.

**Sequential-hazard halting and mixture prediction.** Let λ_k = σ(ℓ_k) be the halt hazard at round k from the halt head (ℓ_k the logit), with the last round forced to stop (λ_R = 1). The stop-mass at round k is the product of surviving to k times halting at k, and the model's prediction is the stop-mass-weighted mixture of per-round predictions. The expected halting depth E[N] is the stop-mass-weighted expected exit round, read as the per-perturbation depth signature.

---

## 5. Results

### 5.1 A reproducible, effect-independent signature on K562

On K562 (n = 968 perturbations, 4 seeds × 50 epochs, held-out validation), the oracle-supervised learned E[N] clears both corners of the identifiability test (Figure 3):

- **Learned depth spreads across perturbations** (mean 4.58, std 1.18 — 70% of the 8-round budget), rather than collapsing to a per-seed constant.
- **Reproducible across training seeds:** split-half ρ = 0.76 (validation 0.74).
- **Recovers the oracle target:** ρ(E[N], r\*) = 0.61, and ρ(E[N], arg min-depth) = 0.59.
- **Not an effect-size proxy:** the raw depth↔magnitude correlation is ρ(effect) = −0.35, but the partial correlation controlling #DE genes and cell count collapses to ρ = −0.04 (covariate R² = 0.15).

A naive learned gate, by contrast, collapses to a per-seed constant with no per-perturbation structure. The signature exists only with the oracle scaffold — a probe token, an unsaturated gate, and a magnitude-free target.

### 5.2 Reproducible within a context, not across

We retrain the halt head on each of four Replogle lines (K562, HepG2, Jurkat, RPE1; few-shot on the frozen pretrained backbone, single-cell, no from-scratch confound) and compare the model's E[N] across independent halves and across lines (Figures 4, 5).

- **Within a line, E[N] reproduces:** head-seed split-half ρ = 0.76–0.85 (mean 0.80), sitting below the effect-size noise ceiling (0.97) and above the model-free ceiling (0.011).
- **Across lines, the ranking does not port:** mean cross-line ρ = 0.14 (n = 194 shared perturbations).

A three-tier decomposition (Table 2) locates where the dependence lives by asking what varies between two fits:

| What varies between the two fits | E[N] reproducibility ρ |
|---|---|
| Only the halt-head seed (same backbone, same data) | 0.93 |
| The backbone seed (same data, retrained backbone) | 0.29 |
| The cell line | 0.14 |
| The donor subset (different donors) | 0.06 |

The reading: on a fixed pretrained backbone — the way the model is actually deployed — the depth signature is *cell-type-specific*, reproducible within a context but not shared across contexts. The backbone-seed tier (0.29) shows a secondary dependence on the particular fit when the backbone itself is retrained from scratch, which is a robustness caveat rather than the deployment case. Portability across donors is a decisive negative, and it is the expected behaviour of a context-specific property, not evidence that depth is a mere training artifact.

### 5.3 What depth tracks biologically, across four lines

To characterise what depth tracks — and to confirm effect-independence beyond K562 — we correlated E[N] against five covariates in every line: response magnitude, #DE genes, cell count, knockdown strength (the autologous target's own |Δ|), and target baseline expression, plus the partial correlation depth∼effect controlling {#DE, cells} (Table 3, Figure 7).

| Covariate | K562 | RPE1 | Jurkat | HepG2 |
|---|---|---|---|---|
| Response magnitude | −0.35\*\*\* | +0.32\*\*\* | +0.24\*\*\* | +0.36\*\*\* |
| #DE genes | −0.35\*\*\* | +0.31\*\*\* | +0.23\*\*\* | +0.32\*\*\* |
| Cell count | −0.08\* | −0.24\*\*\* | −0.01 | +0.14\*\*\* |
| Knockdown strength | −0.03 | +0.10 | −0.00 | +0.02 |
| Baseline expression | −0.03 | +0.09 | −0.00 | +0.03 |
| Partial (depth∼effect \| #DE, cells) | +0.03 | −0.02 | +0.07 | +0.16\* |

*(\*\*\*p < 10⁻³, \*p < 0.05; unmarked p > 0.05.)*

**Depth is not a reagent or abundance artifact.** Knockdown strength and target baseline expression are uncorrelated with E[N] in *every* line (|ρ| < 0.10, all n.s.). Halting depth is not measuring assay efficiency or gene abundance — it is a property of the *response*, not the *reagent*. This is the cleanest negative control in the panel and it passes in all four lines.

**Depth relates to complexity with a cell-type-specific sign.** E[N] correlates with response magnitude and #DE genes (ρ = 0.24–0.36) in HepG2, Jurkat, and RPE1 — larger, broader responses need more refinement — but K562 has the *opposite* sign (−0.35): there, larger responses halt earlier. This sign flip is itself a cell-type-specificity result: the depth↔complexity relationship is not universal, consistent with the cross-line ρ = 0.14 of §5.2.

**Effect-independence holds in all four lines.** After regressing out #DE and cell count, the partial correlation between depth and effect size collapses to +0.03 (K562), −0.02 (RPE1), +0.07 (Jurkat), and +0.16 (HepG2, the only weak residual, p = 0.03). The raw depth↔magnitude correlation is thus mediated by DE-gene count, not by a direct dependence on effect size — the K562 anchor (partial −0.04) reproduced across the panel.

### 5.4 Halting-depth clusters organize druggability and toxicity

§5.1–§5.3 treat E[N] as a scalar and test it *linearly* against covariates. That view is deliberately conservative, and it is also where a real pattern hides. When perturbations are *clustered on the depth signature itself* — the four-seed E[N] fingerprint with halt confidence and the oracle round r\*, reduced by UMAP and partitioned by Leiden into 9–11 clusters per line (Figure 8) — and each cluster is cross-referenced against a 1,708-gene provenance-tagged drug/toxicity annotation, the clusters resolve into functional strata with sharply different pharmacological profiles.

**The depth axis recovers functional gene classes.** Labelling each Leiden cluster by its most over-represented GO-Biological-Process theme (≥ 1.3× enrichment vs the line background) gives coherent, data-driven identities: the deepest cluster is translation/ribosome-dominated in HepG2 (4.1×), Jurkat (4.1×), and RPE1 (4.5×); DNA-replication/repair and mRNA-splicing occupy mid-depth clusters; transcription-regulation sits shallowest (HepG2 3.7×). K562 is inverted, as everywhere else in this paper: its translation/ribosome cluster is the *shallowest* (E[N] = 2.0, 2.2×), mirroring the depth sign flip of §5.3.

**Approved and clinical drug targets concentrate in that translation/ribosome cluster — in every line.** This is the strongest single pharmacological pattern in the dataset (Table 4). It is invisible to the linear test — the correlation between E[N] and approved/clinical status is weak in every line (|ρ| ≤ 0.21) — because the drug-target density is not spread monotonically along the depth axis; it is concentrated in one functional cluster sitting at whichever depth extreme the ribosomal machinery occupies. Per-cluster, the fraction of approved/clinical drug targets peaks in the translation/ribosome cluster at odds ratios of 2.8–9.8 over the rest of the line, with HepG2 reaching 38% approved/clinical (Fisher p = 6 × 10⁻¹⁴).

| Line | Cluster (depth position) | mean E[N] | % approved/clinical | Odds ratio | p |
|---|---|---|---|---|---|
| HepG2 | c8 (deepest) | 8.0 | 38% | 9.8 | 6 × 10⁻¹⁴ |
| RPE1 | c6 (deepest) | 8.0 | 31% | 5.7 | 1 × 10⁻⁸ |
| K562 | c8 (shallowest) | 2.0 | 29% | 5.6 | 6 × 10⁻⁶ |
| Jurkat | c8 (deep) | 7.7 | 18% | 2.8 | 7 × 10⁻³ |

**Toxicity and essentiality co-localize in the same clusters.** These drug-dense clusters are also the most safety-flagged and most essential: in HepG2 the deep translation cluster is 97% common-essential (OR 13.6) and 90% safety-flagged (OR 5.8); Jurkat's c9 is 97% essential (OR 12.9); K562's shallow translation cluster is 94% safety-flagged (OR 9.4); RPE1's deep c6 is 83% safety-flagged (OR 3.2). The depth signature therefore separates perturbations into strata that differ jointly in druggability *and* toxicity liability, and the drug-and-toxicity-dense stratum sits at whichever depth extreme the translation/ribosome machinery occupies in that line.

### 5.5 The descriptor is non-redundant with network topology — but non-additive for one ranking task

The organization above is real and reproducible across all four lines, and it is mechanistically coherent: the translation/ribosome cluster is simultaneously drug-target-rich (many established oncology targets act on translation and ribosome biogenesis), essential, and safety-flagged, and the halting depth places it at a reproducible extreme of the refinement axis. This is a genuine structuring of the target landscape by the depth signature — not merely a restatement of effect size, since the linear effect-size and depth correlations are weak (§5.3). It is descriptive rather than causal: depth clusters provide a *map of where druggable-but-toxic machinery sits on the refinement axis*, useful for stratifying candidates by both opportunity and liability.

The critical question is whether this map is redundant with what a protein network already encodes.

**No network statistic reproduces E[N].** Across four lines the single strongest correlate is PPI degree at |ρ| = 0.23 (K562) — and even its sign is inconsistent across lines (K562 −0.23 but HepG2 +0.19), the opposite of what a genuine shared determinant would produce. Every line's best network statistic sits *below* a |ρ| = 0.3 novelty ceiling and nowhere near a |ρ| = 0.5 redundancy floor (Figure 10). Partial correlations controlling for effect size and #DE leave these near zero (max |ρ_partial| = 0.14), so E[N] is not a re-encoding of response magnitude either. Halting depth carries per-perturbation ordering that neither network topology nor effect size predicts — it is a genuinely novel descriptor, the non-redundant half of the "useful" claim.

**Non-additive for one target-class ranking task.** Where the result turns negative is narrow and specific. For a single downstream target-class ranking task, added on top of a STRING-network prior built from the same functional modules, E[N] gives no further lift in three of four lines. Critically, non-redundancy as a *descriptor* and non-additivity for that one *classification task* are different claims: the network prior encodes the identical module structure, so once it is present the two are redundant *for that task*, not contradictory. Only the latter claim is negative, and it does not diminish the signature's novelty as a descriptor.

**Caveat: directed coverage.** CollecTRI is TF→target only, so the directed downstream statistics are identically zero for the ∼88% of in-network perturbations that appear solely as targets; the directed test is under-powered and tie-dominated. Restricting to the 78 perturbations that are TFs with ≥ 1 downstream target, E[N] versus out-degree gives ρ = 0.38 (K562, p = 0.02), 0.46 (HepG2, p = 0.001), −0.01 (Jurkat), 0.26 (RPE1) — a modest, *inconsistent* positive in two of four lines, still well below redundancy. The well-powered undirected PPI comparison (n ≈ 900–1030 per line) is the primary result and is unambiguous.

### 5.6 Single-cell refinement depth in primary CD4+ T cells

We applied STATE oracle-ACT to a two-donor, single-cell subsample of the genome-wide Marson/Pritchard GWCD4i CD4+ T-cell CRISPRi screen (donors D1 and D4, three activation states — resting, stimulated 8h, stimulated 48h — 2000 HVG). The Part-2 question is whether inflammatory-program suppressors carry a distinct refinement signature, and whether the depth signature ports where the biology converges (Figure 11).

- **Reproducible within a donor, and reproducibility rises with stimulation:** within-donor split-half ρ = 0.62 → 0.68 → 0.75 across resting, 8h, and 48h.
- **Donor-specific until the biology converges:** cross-donor agreement is ≈ 0 in the resting state (ρ = −0.11, n = 5745) but reaches ρ = 0.49 at the fully stimulated 48h endpoint (n = 5840), exactly where cells from different donors converge onto a shared activation program.
- **Depth separates phenotype categories at 48h** (Kruskal–Wallis p = 5 × 10⁻²⁸): resting-sparing inflammatory-program suppressors converge faster — lower E[N] — than damaging (Cliff's δ = −0.16) and generic (δ = −0.24) perturbations.

This mirrors the cross-line result of §5.2: the signature ports where, and only where, the biology converges — and it independently reproduces concurrent evidence that the informative layer of a single-cell foundation model is set by cellular context [16].

**A cell-context-dependent drug negative.** Clustering the 12-feature CD4 signature (E[N], halt confidence, nonlinear/predicted-error × 3 conditions) yields nine well-separated groups (Leiden modularity 0.66), organized by condition-dependent refinement *dynamics* rather than static pathway family; one cluster is a distinct shallow/fast-converging island (E[N] ≈ 5.9 vs 7.2–7.6 elsewhere). Unlike the Replogle lines, where approved and clinical drug targets concentrated in the translation/ribosome cluster (§5.4, OR 2.8–9.8), no CD4 depth cluster is enriched for drug targets (all OR 0.64–1.25, no significant p), with flat druggability and toxicity across depth clusters (approved-drug-target fraction 5–8%, DepMap-essential 1–2% against a 1.6% background, LoF-intolerant 4–8%). There is no essentiality or druggability contrast across CD4 depth clusters to drive the drug/depth co-localization seen in the Replogle lines; the drug/depth pattern is itself cell-context-dependent, reinforcing rather than weakening the cell-type-specificity conclusion.

### 5.7 Cell-state robustness (exploratory)

As an exploratory side-analysis, we read halting from the four frozen Replogle backbones on each line's own data, conditioning on the basal cell-cycle phase of the input cell (Figure 12).

- **Depth is state-invariant:** E[N] conditioned on G1 versus G2M basal state lies on the diagonal (ρ = 0.996–0.999) in every line.
- **Reproducible within each state:** within-state split-half reproducibility of E[N] / arg min / r\* is ρ ≥ 0.83 in every state and line.
- **Trajectory geometry distinguishes deep from shallow:** the model-implied per-layer trajectory of deep-E[N] perturbations dips mid-trajectory then recovers — the model must re-orient its prediction before snapping onto the target direction — while shallow perturbations approach monotonically.

Halting is read from the frozen backbones with no retraining and no temporal labels. The state-invariance supports the reading of E[N] as a property of the *response*, coupled to the geometry of the model's internal trajectory rather than to the starting state of the cell.

### 5.8 From-scratch pseudobulk (exploratory limitation)

We also asked whether the depth signature survives when the backbone is trained from scratch on pseudobulk rather than read from the frozen pretrained model. It does not, in an informative way: a from-scratch pseudobulk model fits the response but the depth collapses — the oracle round r\* is degenerate for 100% of perturbations (a step-function diagnostic), so there is no per-perturbation depth to recover. We attribute this to the pseudobulk substrate — a single averaged profile per perturbation offers no per-cell refinement structure for the gate to key on — and we cannot yet claim more. We report this as a limitation, not a win: the working regime is the frozen pretrained backbone at single-cell resolution, and the from-scratch pseudobulk case marks one boundary of where the method applies.

---

## 6. Discussion

**The negatives are findings.** This paper is falsification-first by design, and its durable contributions are as much the negatives as the positives.

- **Learned halting is unidentifiable without the oracle scaffold.** A free gate collapses to a constant; a magnitude-anchored gate becomes an effect-size proxy. The signature exists only when depth is read off a frozen, well-fitted backbone with a magnitude-free target and per-perturbation oracle supervision.

- **The signature does not port across contexts.** E[N] is cell-type-specific: reproducible within a line (mean ρ = 0.80) but not shared across lines (ρ = 0.14) or across resting donors (ρ ≈ 0), porting only at the stimulated endpoint where the biology converges. This is the expected behaviour of a context-specific property.

- **The depth signature is a novel descriptor that organizes druggability.** Clustered on its own, E[N] concentrates approved and clinical drug targets (and toxicity and essentiality) in a translation/ribosome cluster at a reproducible depth extreme, and no network statistic reproduces its per-perturbation ordering (best |ρ| = 0.23, below a 0.3 novelty ceiling) — it is not an expensive recomputation of wiring. The one boundary is that it is non-additive with a STRING-network prior built from the same modules for one target-class ranking task.

Taken together, the *incremental target-class ranking* verdict is negative, while the *descriptor-level druggability organization* is positive: the depth signature carries a real, reproducible map of where druggable-but-toxic machinery sits on the refinement axis, and separately fails to sharpen one specific downstream ranking once a network prior encoding the same modules is present. These are two different claims, and only the first is a negative.

**What E[N] measures, and what it does not.** E[N] is a proxy for response complexity — the layer at which a fixed stack's endpoint estimate converges — supported by three robustness results: it is independent of knockdown strength and gene abundance (a property of the response, not the reagent), it is invariant to basal cell state (ρ ≈ 0.99), and it couples to the geometry of the model's internal trajectory. It is not biological time and not causal cascade length; a single endpoint cannot recover either, and we do not claim it does.

**Limitations.** The signature is context-specific and does not transfer across cell types or resting donors, which bounds its use as a portable invariant. The from-scratch pseudobulk case (§5.8) marks one boundary of the method's applicability, and we cannot yet fully attribute that collapse beyond the pseudobulk substrate. The directed-network comparison is under-powered (§5.5); the well-powered undirected result is the primary one. And the drug-target organization is descriptive, not causal.

---

## 7. Conclusion

Methodologically, this paper's contribution is a reframing: adaptive-depth halting, invented to save compute, is read here as a biological measurement — the first such use on a perturbation model, to our knowledge — and it survives the tests a new descriptor must pass while failing, informatively, the ones that bound it. Adaptive-depth halting E[N] is a reproducible, effect-independent, cell-type-specific property of a perturbation's response, recoverable from single-endpoint data when learned halting is read as a measurement off a fixed, gradually refining backbone. It ports across contexts only where the biology converges. Clustered on its own it is a novel descriptor that organizes druggability and toxicity, non-redundant with network topology, though not portable across contexts and non-additive with a network prior for one incremental target-ranking task. When a model knows enough is thus a real, measurable — and local — property of the perturbation.

---

## Figures and Tables

*(Figures and tables are referenced by number; the typeset figures live in the accompanying figure set.)*

- **Figure 1.** Where adaptive-refinement hosts sit: 43 perturbation models (2024–2026) by family track and ACT-graftability; 11 of 43 graftable (HIGH / MEDIUM-HIGH).
- **Figure 2.** The frozen stack's per-layer endpoint estimate refines through layers 1–6 (0.909 → 0.881) then degrades over the final two layers; the convergence layer arg min_k D_k is per-perturbation and spans all eight layers.
- **Figure 3.** Oracle-supervised learned ACT on frozen STATE (K562, n = 968): learned E[N] spreads across perturbations; reproducible across seeds (ρ = 0.76); recovers the oracle round (ρ = 0.61); not an effect-size proxy (partial ρ = −0.04).
- **Figure 4.** Learned-E[N] split-half reproducibility: within each line the learned E[N] reproduces (purple), between the data-derived oracle depth (teal) and the effect-size noise ceiling (grey); reproducible within a cell type (mean 0.80), absent across contexts (cross-line 0.14, cross-donor 0.06).
- **Figure 5.** Cross-cell-line test on frozen pretrained ARC ST-SE-Replogle: within-line reproducible (ρ = 0.76–0.85), cross-line weak (ρ = 0.14, n = 194); reproducibility decays 0.93 → 0.29 → 0.14 → 0.06.
- **Figure 6.** Donor stability — leave-one-donor-out on a pooled three-context CD4+ T-cell model: reproduces within a fitted model (0.93), decays to zero across donors (0.06) through an intermediate backbone-seed tier (0.29).
- **Figure 7.** Halting depth E[N]: reproducible sign, effect-independent (all four lines) — full-panel covariates, target-in-panel controls, and the effect-independence partial test.
- **Figure 8.** Depth-signature UMAP with Leiden clusters annotated by fold-enriched GO-BP theme and mean E[N], per Replogle line; translation/ribosome clusters anchor the depth extreme (deepest in HepG2/Jurkat/RPE1, shallowest in K562).
- **Figure 9.** Halting-depth clusters organize drug-target density and toxicity: approved/clinical targets peak in the translation/ribosome cluster; essentiality and safety-flag density rise toward the depth extreme where the ribosomal machinery sits.
- **Figure 10.** Halting depth E[N] is non-redundant with network topology: E[N] vs directed-GRN cascade depth (no relationship); |ρ| between E[N] and ten network statistics all below the 0.3 novelty ceiling, none near the 0.5 redundancy floor; strongest correlate PPI degree at |ρ| = 0.23 with inconsistent sign.
- **Figure 11.** CD4+ T-cell single-cell refinement depth: within-donor reproducibility rises with stimulation (0.62/0.68/0.75); cross-donor transfer only at the 48h endpoint (ρ 0 → 0.49); resting-sparing suppressors converge faster (Kruskal–Wallis p = 5 × 10⁻²⁸; δ = −0.16 to −0.24).
- **Figure 12.** Cell-state robustness of refinement depth (exploratory, four frozen Replogle lines): trajectory dips then recovers for deep-E[N] perturbations; E[N] state-invariant (ρ = 0.996–0.999 G1 vs G2M); within-state split-half ρ ≥ 0.83.

- **Table 1.** Identifiability-wall outcomes: grafting hosts against the two-corner test (reproducible and effect-independent).
- **Table 2.** Three-tier reproducibility decomposition: what varies between two fits vs E[N] reproducibility ρ.
- **Table 3.** Biological validation, all four lines: Spearman ρ of E[N] against each covariate.
- **Table 4.** Approved/clinical drug targets concentrate in the translation/ribosome depth cluster (Fisher exact vs the rest of the line).

---

## References

*(Numbered citations [1]–[15] in the body correspond to the existing bibliography; align numbering with the source `.bib`. Key sources are listed here with [16] the concurrent single-cell-layer-depth work.)*

- **[1]** A. Graves. *Adaptive Computation Time for Recurrent Neural Networks.* arXiv:1603.08983, 2016.
- **[2]** A. Banino, J. Balaguer, C. Blundell. *PonderNet: Learning to Ponder.* ICML Workshop, 2021.
- **[10]** Diffusion-based single-cell perturbation model (scDiff / diffusion host used in the graftability survey).
- **[11], [13]** Graph message-passing perturbation models (e.g. GEARS; TxPert), used as grafting hosts.
- ARC Institute **STATE** (ST-SE-Replogle variant), frozen backbone.
- J. M. Replogle et al. *Mapping information-rich genotype–phenotype landscapes with genome-scale Perturb-seq.* Cell, 2022.
- Marson / Pritchard **GWCD4i** genome-wide CD4+ T-cell CRISPRi Perturb-seq screen.
- STRING protein–protein interaction database; **CollecTRI** TF→target regulatory network.
- Open Targets, ChEMBL, DepMap, Pharos, gnomAD, FDA — drug/toxicity/essentiality annotation sources (compiled with provenance on every value over the 1,708-gene target union).
- **[16]** V. Y. Civale, R. Semeraro, A. D. Bagdanov, A. Magi. *Intermediate Layers Encode Optimal Biological Representations in Single-Cell Foundation Models.* arXiv:2604.14838, 2026.
