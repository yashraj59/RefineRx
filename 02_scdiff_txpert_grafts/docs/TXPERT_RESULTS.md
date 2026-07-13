# Adaptive-depth TxPert: does a message-passing depth axis escape the effect-size collapse?

**The test.** The scDiff sweep left one hypothesis open: scDiff's refinement axis *re-noises the
same single endpoint* each round, so "rounds needed" is almost definitionally "distance to
endpoint" = effect size. TxPert offers a fundamentally different depth axis — **message-passing
hops over a gene graph**, where depth = regulatory propagation distance, not endpoint magnitude.
If any architecture could give an effect-size-independent complexity signal on single-endpoint
data, this is the one. So I grafted PonderNet halting onto that axis and tested it.

## What was built (grounded in valence-labs/TxPert)
- **Depth axis** = TxPert's `MGAT.forward` message-passing loop (`multi_graph.py` L146-172), which
  already collects one node embedding per GATv2 hop. I made it **adaptive**: one weight-shared,
  dimension-preserving `GATv2Conv` (TxPert's exact conv) applied K=8 times, with a PonderNet
  halting head over the hops - reusing TxPert's `GatedCombiner` sigmoid-gate form (L17).
- **Graph** = co-expression kNN (k=16) over 2,000 gene nodes, built on control cells. Node set
  forced to include all 89 measured perturbed genes (only 9 are in the default HVG set), so each
  perturbation's target gene is a real node whose influence propagates through the graph.
- **Readout** = the halted embedding of the *perturbed* gene node -> decoder -> full expression.
  E[hops] per perturbation = expected message-passing depth at its target node.
- **Data / compute** = same Adamson data (62,477 cells, 89 perturbations with node targets),
  L40 GPU, 30 epochs (~13 s/epoch), TRAIN_EXIT=0. recon 0.2217->0.2148.

## Result: the collapse is architecture-invariant - and it is NOT graph topology

| axis | model | rho(E[N], effect size) | E[N] spread | graph-degree control |
|---|---|---|---|---|
| re-noising (generative) | scDiff | -0.704 | 0.08 of 5 (1.7%) | - |
| **message-passing (graph)** | **TxPert** | **-0.874** | 0.44 of 8 (5.5%) | rho(E[N],deg)=-0.022 |

**1. Message-passing depth collapses to effect size too - more strongly, not less.**
TxPert's E[hops] is anti-correlated with response magnitude at **rho=-0.874**
(Pearson -0.823), even tighter than scDiff's -0.70.
Big-effect knockdowns (CAD, EIF2S1, HARS; shift 5-9) halt in the fewest hops; subtle-effect
knockdowns (MANF, HYOU1, STT3A; shift ~1) take the most. Same near-constant depth
(5.5% of the 8-hop budget).

**2. Graph topology is NOT the mediator - this is the key control.** The obvious worry with a
message-passing axis is that a well-connected (hub) gene reaches the readout in fewer hops
regardless of biology. It doesn't: **rho(E[hops], graph degree) = -0.022**
(essentially zero, Fig panel c), and the effect-size coupling **survives regressing out degree**
(partial rho = -0.873). So the halting head is keying on the
size of the expression change, not on where the gene sits in the graph.

**3. Both architectures agree in direction and magnitude.** Two completely different depth
mechanisms - generative re-noising and graph message-passing - independently converge to
"more effect -> fewer refinement steps, and almost no spread beyond that." That cross-architecture
agreement is exactly the architecture-invariance check the thesis called for, and it comes back
**negative in both**.

## Interpretation for the thesis

This is the **strongest form of the negative result**. It rules out the most plausible escape
hatch: that the effect-size collapse was a quirk of scDiff's re-noising axis. It was not. On
single-endpoint Perturb-seq data, an adaptive-depth halting signal - whether the depth is
diffusion refinement OR graph propagation - reduces to a (mostly constant, effect-size-shaped)
proxy for response magnitude. And the graph-degree control rules out the one confound that was
specific to the message-passing story.

Under the pre-registered rule (*"if the signature collapses to an effect-size proxy, that negative
result is the finding"*), the finding now rests on **two independent architectures plus a topology
control** - a much harder result to dismiss than a single model.

### Honest caveat on the smoke vs. full run
The 2-epoch smoke showed rho~+0.08 (apparent decoupling). That was an *undertrained* halting head
emitting near-random halts - as training converged over 30 epochs, the head learned to key on
effect size and the coupling emerged. Early-training decoupling is noise, not signal; only the
converged model's signature is interpretable. (This is itself a useful methodological note: never
read a per-perturbation signature off an undertrained ponder head.)

## What this leaves for the thesis
The negative result is now robust. The remaining ways to get a *non-redundant* depth signal are
structural, not hyperparameter tweaks:
1. **Intermediate supervision with real depth** - a trajectory/pseudotime target or a mechanistic
   simulator, so "steps needed" is anchored to something other than endpoint distance.
2. **Multi-endpoint or time-resolved data** - the thesis already notes single-endpoint data cannot
   recover causal depth; this experiment is empirical confirmation of that limit.
3. If the goal is the Marson/Pritchard target ranking, the honest path is to rank on the response
   signature directly (effect size, DE structure, program specificity) and treat halting depth as
   **demonstrated-redundant** rather than as an independent axis.
