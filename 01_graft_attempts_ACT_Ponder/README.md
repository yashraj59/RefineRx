# Stage 1 — First ACT/PonderNet halting grafts (what we tried first)

The first attempts to *learn* a per-perturbation refinement depth. We grafted ACT / PonderNet halting
onto small in-house predictors and onto scDiff, then ran a 12-config sweep over the ponder weight,
geometric prior, and β to try to break the "constant-depth collapse," and separately deepened the
refinement budget from 5 to 10 rounds to see whether E[N] would use the extra range. **Neither
worked.** The prior/β knobs slide the *mean* E[N] anywhere from ~1.8 to ~4.5 but never create
per-perturbation *spread*; every configuration with real spread turns out to track raw effect size
(|ρ| up to 0.90), and doubling the budget bought almost no usable resolution (the widest
per-perturbation spread was ≈4% of the 10-round budget). The `code/` holds the halting heads, refiners,
data loaders, sweep and diagnostics; `docs/` records the setup and the sweep verdict; `figures/` shows
the depth distribution and the convergence transient. **This is the project's first negative result,
kept deliberately as evidence:** on single-endpoint data a freely-learned halting depth is either
uninformative (constant) or redundant (an effect-size echo) — the identifiability problem that the
rest of the arc works to escape.
