# Stage 2 — scDiff + TxPert adaptive-depth grafts (architecture-invariance test)

To check that Stage 1's collapse was not an artifact of one architecture, we ported the halting idea to
two more backbones — scDiff (x0-refinement) and TxPert (adaptive message-passing hops) — and, crucially,
added a **magnitude-free** target: the halting head now predicts each perturbation's unit response
*direction*, with a scale-invariant cosine halting loss, so "how many rounds are needed" measures
response structure rather than size. This exposes the finding that defines the wall: a
**reproducibility–redundancy trade-off**. Magnitude-anchored halting is reproducible across seeds but
redundant with effect size (ρ(E[N], effect) = −0.70 for scDiff, −0.87 for TxPert); the magnitude-free
version genuinely decouples from effect size but is **no longer reproducible** — its per-perturbation
ranking swings with the random seed (cross-seed ρ ≈ 0.14–0.15, at the noise floor), and this holds
whether trained full-batch or per-cell with minibatches, so it is a property of the signal, not the
optimizer. The `code/` holds both grafts and their training loops; `docs/` records the TxPert and
signature results. **The mechanism is identifiability:** a single endpoint per perturbation does not
constrain how many refinement steps produced it, so once magnitude stops pinning the depth, the depth
is underdetermined. This negative is sharper than "use time-resolved data" — it says exactly *why*
single-endpoint halting depth is either redundant or unstable, and it is kept as the core evidence for
the thesis.
