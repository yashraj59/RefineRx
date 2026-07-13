# Getting a per-perturbation signature OUT of single-endpoint data — the honest result

**Your thesis premise, restated correctly:** the goal is to extract a reproducible,
effect-size-independent per-perturbation *complexity* signature **from** single-endpoint
control-vs-perturbed data — not to give up and demand time-resolved data. My earlier "the limit is
the data" framing was wrong to imply the question was unanswerable; this document answers it
properly, and the answer has a specific, useful shape.

## The flaw in the first two experiments (and why I re-ran)

The scDiff and TxPert grafts both trained the halting head against the **full-magnitude** response
(`||pred − target||²` / endpoint L2). That loss is *dominated by response magnitude*, so a head that
halts when the loss is low mechanically learns "halt once the big-magnitude endpoint is reached."
The −0.87 effect-size collapse was therefore **partly self-inflicted** — I baked magnitude into the
halting objective. That is not a fair test of a *magnitude-independent* complexity signal. So I
rebuilt the target to be magnitude-free.

## The fix: a magnitude-free refinement target

Same adaptive message-passing depth (TxPert's GATv2 hops + PonderNet halting), but the model now
predicts each perturbation's **unit response direction**, and the per-step halting loss is
**cosine distance** `1 − cos(pred, target_dir)` — scale-invariant, bounded [0,2]. Now "how many
refinement steps does this perturbation need" measures the *structure/direction* of the response,
not its size. I then ran the thesis's pre-registered protocol: **reproducibility across seeds first,
non-redundancy after regressing out effect size / #DE genes / cell count / graph degree.**

## Result: removing magnitude works — but trades one failure for the other

| regime | ρ(E[N], effect) | E[N] spread | **reproducibility ρ** | verdict |
|---|---|---|---|---|
| scDiff, magnitude-anchored | −0.70 | 1.7% | high (constant) | reproducible but **redundant** |
| TxPert, magnitude-anchored | −0.87 | 5.5% | high (constant) | reproducible but **redundant** |
| **magnitude-free, full-batch** | **+0.26** | 25% | **0.15** | non-redundant but **not reproducible** |
| **magnitude-free, per-cell** | **-0.42** | 68% | **0.14** | non-redundant but **not reproducible** |

**1. The magnitude-free target does decouple E[N] from effect size.** Coupling drops from −0.87 to
+0.26 (full-batch) / -0.42 (per-cell), and
E[N] finally uses its range (25–68% of
budget vs the old ~2–5%). Covariates (effect size, #DE, cell count, degree) explain only
R²=0.09–0.31 of it. So the
depth signal is **genuinely non-redundant** — it is *not* an effect-size proxy anymore.

**2. But it fails the reproducibility gate — which comes first.** Mean pairwise rank correlation of
the per-perturbation E[hops] ranking across 6 random seeds is only
**0.15** (full-batch) / **0.14**
(per-cell) — right at the n=89 noise floor (ρ≈0.11). The per-seed E[N] means swing from ~1.2
to ~5.2 on the *same data* (Fig panel b). The residual (after removing covariates) is even less
reproducible (0.13).

**3. This is not an optimizer artifact.** I checked the obvious alternative explanation — that the
instability was my full-batch-over-89-means optimizer. Retraining **per-cell with minibatches over
53,296 perturbed cells** (stochastic averaging, like the original runs) gives the *same*
seed-instability. So the irreproducibility is a property of the signal, not the optimization.

## What this means — the real finding

There is a **reproducibility–redundancy trade-off**, and on single-endpoint Adamson data **no regime
occupies the useful corner** (non-redundant AND reproducible; Fig panel a is empty there):

- **Anchor the halting target to magnitude** → the signal is stable across seeds, but it is just a
  re-encoding of effect size (redundant). You learn nothing beyond response magnitude.
- **Remove magnitude** → the signal becomes independent of effect size, but it is no longer a
  property of the *data* — it is a property of the random seed. Different initializations pick
  different "which perturbations are complex" orderings, all fitting the endpoint equally well.

The deep reason is identifiability: a single endpoint per perturbation does not constrain *how many
refinement steps* were needed to reach it. Many depth-assignments reproduce the same endpoint, so
once you stop letting magnitude pin the depth, the depth is underdetermined and the optimizer fills
it arbitrarily. **This is a sharper, more useful statement than "use time-resolved data":** it
identifies exactly *why* single-endpoint halting depth is either redundant or unstable, and it is an
empirical result, not an assumption.

## Honest options for the thesis from here (all within single-endpoint data)

1. **Report the trade-off as the finding.** It is a clean, pre-registered negative with a mechanism
   (identifiability), backed by two architectures × two target types × two optimizers. That is
   publishable as "adaptive-depth halting does not yield a reproducible non-redundant complexity
   signature from single-endpoint Perturb-seq, and here is why."
2. **Constrain the depth so it is identifiable without magnitude.** Anchor refinement to something
   the data *does* pin per perturbation but that isn't raw magnitude — e.g. the geometry of the
   response *direction* relative to a fixed gene-program basis (pathway/GO structure), or an ensemble
   consensus depth (median E[N] across many seeds) tested for reproducibility as its own quantity.
   The consensus route is the natural next run and I can do it directly.
3. **Drop halting depth; keep the direction signature.** The unit response *direction* itself IS a
   reproducible per-perturbation object (it's a data statistic, not a learned depth). Rank
   Marson/Pritchard targets on direction-space structure (program specificity, off-target overlap)
   and treat halting depth as demonstrated-non-viable. This still delivers the application, without
   over-claiming the depth axis.

My recommendation: **(1) as the honest headline, (2) as the one more experiment worth trying** — an
ensemble-consensus depth, testing whether *averaging over seeds* recovers a reproducible signal, or
confirms the underdetermination. I can launch that next.
