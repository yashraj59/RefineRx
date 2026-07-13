# Stage 8 — CD4-native STATE fused halting, trained from scratch (current deliverable)

The current work: rather than reading depth off a *frozen* Replogle backbone applied to CD4 data (Stage
7), this stage trains a **CD4-native STATE model with adaptive-depth halting fused in and trained from
scratch**, so the refinement axis is learned on CD4 biology directly. The halting is magnitude-free and
**jointly calibrated** with the response objective (config: 8-layer Llama backbone, hidden 768,
6 refinement rounds, halt warmup, confidence token), which is the design the earlier grafts showed is
necessary to avoid both the constant-depth collapse and the seed-instability. This is the bridge the
Stage-6 synthesis flagged as "still to build": a CD4 backbone on which the directional-correction
geometry and the suppressor-shallow / stimulator-deep ordering can be tested *on the application
dataset itself*. Either outcome is reportable — if the native E[N] reproduces the coupling the two
threads fuse into a single claim on CD4; if it collapses (the pseudobulk risk), that too is a finding
under the negative-result mandate. `code/` holds the STATE transition-halt model and the halting graft;
`config/` the training YAML and the CD4 run config; `notebooks/` the training and embedding-use
notebooks. **This is in-progress / the current deliverable**, included here so the arc ends where the
project actually stands.
