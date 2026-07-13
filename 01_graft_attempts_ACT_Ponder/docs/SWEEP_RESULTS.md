# scDiff+PonderNet hyperparameter sweep — can the constant-depth collapse be broken?

**Question (both halves of the request):** (1) sweep `ponder_weight` / `lambda_prior` /
`ponder_beta` to try to break the constant-depth collapse, and (2) deepen the refinement
budget from 5→10 rounds and check whether E[N] actually uses the extra range.

**Method.** 12 configs trained fresh on the real Adamson data (62,477 cells × 2,000 HVG,
92 conditions), one process, data loaded once, on the L40. Each config: 12 epochs, then the
control-relative per-perturbation signature over the 90 named-gene knockdowns and the Stage-0
effect-size confound check (Spearman ρ between E[N] and ‖pert−control‖). Total ~40 min.

## Results (all 12 configs)

| tag | N | w | prior | β | mean E[N] | E[N] range | % of budget | ρ(E[N], shift) | loss_simple |
|---|---|---|---|---|---|---|---|---|---|
| r5_base | 5 | 0.1 | 0.2 | 0.05 | 2.290 | 0.085 | 1.7% | -0.589 | 0.1712 |
| r5_lowprior | 5 | 0.1 | 0.1 | 0.02 | 2.112 | 0.125 | 2.5% | -0.796 | 0.1710 |
| r5_highbeta | 5 | 0.1 | 0.1 | 0.2 | 2.757 | 0.082 | 1.6% | -0.640 | 0.1719 |
| r5_highprior | 5 | 0.1 | 0.3 | 0.02 | 1.807 | 0.120 | 2.4% | -0.841 | 0.1706 |
| r5_w0.5 | 5 | 0.5 | 0.2 | 0.05 | 2.352 | 0.087 | 1.7% | -0.545 | 0.1732 |
| r5_w1.0 | 5 | 1.0 | 0.2 | 0.1 | 2.416 | 0.071 | 1.4% | -0.581 | 0.1741 |
| r10_base | 10 | 0.1 | 0.2 | 0.05 | 3.095 | 0.132 | 1.3% | -0.862 | 0.1720 |
| r10_lowprior | 10 | 0.1 | 0.1 | 0.02 | 3.002 | 0.335 | 3.4% | -0.883 | 0.1720 |
| r10_deepprior | 10 | 0.1 | 0.05 | 0.02 | 3.432 | 0.408 | 4.1% | -0.898 | 0.1722 |
| r10_highbeta | 10 | 0.1 | 0.1 | 0.2 | 4.497 | 0.121 | 1.2% | -0.052 | 0.1727 |
| r10_w0.5 | 10 | 0.5 | 0.1 | 0.05 | 4.294 | 0.329 | 3.3% | -0.703 | 0.1761 |
| r10_w1.0 | 10 | 1.0 | 0.1 | 0.1 | 4.505 | 0.211 | 2.1% | -0.658 | 0.1780 |

## The answer: no — the collapse does not break

**1. Deepening the budget did NOT make E[N] use the extra range.** Going 5→10 rounds moved the
*mean* (E[N] 2.3→3–4.5) but the per-perturbation *spread* stayed near the floor. The widest
spread of any config was **r10_deepprior** at **4.1% of the
10-round budget** (range 0.41). Doubling the budget
bought almost no usable per-perturbation resolution — the model picks a near-global depth and
sticks to it regardless of which gene is knocked down.

**2. The prior/β knobs move the mean, not the spread.** `lambda_prior` and `ponder_beta`
reliably slide mean E[N] anywhere from 1.8 (r5_highprior) to 4.5 (r10_highbeta) — the halting
head halts wherever the KL prior tells it to on average. But no setting converts that into
per-perturbation structure that isn't just effect size (Fig, panel b: spread is flat across the
whole mean-E[N] axis).

**3. Every config with real spread stays an effect-size proxy.** Across the sweep, the more
E[N] varied between perturbations, the *more* strongly it tracked raw effect size
(|ρ| up to 0.90 in the n_refine=10 configs). The one config that decoupled from effect size —
**r10_highbeta** (ρ=-0.052) — did so by going nearly **flat at the
ceiling** (E[N]≈4.5, range only 0.12): it stopped correlating
because it stopped varying at all, not because it found genuine complexity signal.

**No config landed in the useful corner** (Fig, panel a): high per-perturbation spread AND low
effect-size coupling. That corner stayed empty.

## Interpretation for the thesis

Under the pre-registered rule — *"if the signature fails reproducibility or collapses to an
effect-size proxy, that negative result is the finding"* — **this sweep is a strengthened
negative result.** On single-endpoint Adamson data, an x0-refinement PonderNet grafted onto
scDiff does **not** spontaneously discover a per-perturbation "refinement-depth" signal distinct
from response magnitude. The halting behavior is either constant (uninformative) or an effect-size
echo (redundant). Native diffusion loss barely moved across the whole sweep
([0.1706, 0.1780]), so the base model was never
destabilized — the ponder machinery just had no complexity gradient to latch onto.

### Why this is the expected outcome, not a bug
The x0-recurrence re-noises the *same* single endpoint each round. With no intermediate
supervision and no genuine multi-step target, there is no task pressure for one perturbation to
need more rounds than another beyond how far its endpoint sits from control — which is exactly
effect size. This is the structural limit the thesis anticipated for single-endpoint data.

## What would actually be needed to escape it (design implications)
1. **A refinement target with real depth structure** — e.g. supervise intermediate rounds against
   a trajectory/pseudotime or a mechanistic simulator, so "rounds needed" ≠ "distance to endpoint".
2. **An architecture whose depth axis is not a re-noising of the endpoint** — TxPert's message-passing
   hops (adaptive graph depth) or GEARS unrolled propagation give a depth signal tied to regulatory
   distance rather than magnitude; the graftability matrix already flags these as HIGH hosts.
3. **Only then** the reproducibility (seeds/guides/donors) and non-redundancy tests, before any
   biological interpretation on Marson/Pritchard.

Both halves of the "do both" request are answered: the prior/β/weight sweep does not break the
collapse, and the 10-round budget is not used. The negative result is clean and, per the thesis's
own framing, is itself the finding at this stage.
