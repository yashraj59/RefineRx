# How Claude Science Was Used — and Where It Went Wrong

*A process retrospective for the RefineRx / adaptive-depth halting project*
**Author:** Yash Raj · **Project:** "When Does a Perturbation Model Know Enough?"

---

## 1. What this document is

This is an honest account of how the project was actually built inside Claude Science:
which capabilities carried the work, and — more importantly — the mistakes made along
the way, why they happened, and how they were caught. The project's own thesis is that
**a clean negative result is a finding**; that standard applies to the process too. The
errors below are recorded because they shaped the final result and because a reader
reproducing this work should know where the traps are.

---

## 2. The scientific question (for context)

Perturb-seq measures a single endpoint: control vs. perturbed transcriptome. The
question was whether an adaptive-depth model's **halting behaviour** — how many
refinement rounds it needs to predict a perturbation's response — is a reproducible,
effect-size-independent, per-perturbation signature, and whether it helps identify drug
targets. Halting was treated throughout as a **computational proxy for response
complexity, never biological time**. The second half applied the framework to primary
CD4+ T-cell CRISPRi to rank inhibitory targets that suppress stimulated inflammatory
programs while sparing resting cells.

---

## 3. How Claude Science was used

### 3.1 Literature and model catalog
- A code-grounded catalog of **43 perturbation-response models (2024–2026)** was built
  by fetching and reading each model's actual GitHub repository, not by recalling
  architectures from memory. Every GitHub link was verified by querying it live.
- Models were scored for **graftability** — whether a learned-halting refinement loop
  could even be attached — which is what later justified the choice of STATE.

### 3.2 Remote compute (the workhorse)
- All training and GPU work ran on a remote L40 host (Ubuntu 22.04) reached
  over SSH, driven entirely through `call_command` because `submit_job` and `scp` were
  blocked on that host.
- This forced a specific idiom used hundreds of times: **base64-chunk file transfer +
  detached background runs + sentinel-file polling** (`touch DONE` on completion,
  poll for it). Every large file — h5ad datasets, checkpoints, the final PDF — moved in
  ~40 KB base64 chunks because larger payloads broke the SSH mux.

### 3.3 Sub-agent delegation
- Independent tracks were farmed out to parallel sub-agents: cell-state analysis across
  4 lines, halt-head refit + weight persistence, the temporal side-analysis, the GitHub
  code push, the HuggingFace model push, and the final LaTeX-to-PDF compile. Each ran in
  its own context and returned artifacts.
- This kept the main thread's context focused on the science while mechanical, long,
  or independent work happened elsewhere.

### 3.4 Iterative artifacts
- The paper, figures, candidate tables, and reports were all versioned artifacts. The
  paper alone went through ~14 versions; the model card through 3. Every reported number
  was computed from a saved table, never typed from memory.

### 3.5 Data handling
- The CD4 dataset was consumed through the `cell-load` contract with `log1p(counts)`
  normalization (no library-size rescaling), HVG selection matching the notebook recipe,
  and gene symbols carried in `var`. Frozen ARC STATE ST-SE checkpoints supplied the
  backbones for the working halt heads.

---

## 4. Mistakes made — the honest list

These are ordered roughly by how much they cost.

### 4.1 Scientific / framing mistakes (the ones that mattered most)

**M1 — Called the depth signature "redundant with STRING," contradicting our own thesis.**
The drug section originally said adaptive depth was "redundant with" the STRING network
prior. But a whole results section (§GRN) *proves the opposite*: no network statistic
reproduces E[N]'s ordering (best |ρ|=0.23, below the novelty ceiling). "Redundant" has
two meanings — *descriptor-level* (STRING contains the same information: FALSE) and
*task-level non-additivity* (adding E[N] to a classifier that already has STRING gives no
AUC lift: TRUE). Collapsing them into one word "redundant" stated the false, thesis-gutting
version. **Caught by the user**, who pointed out the contradiction directly. Fixed by
replacing every ambiguous "redundant with STRING" with "non-additive for one classifier"
and explicitly separating the two claims.

**M2 — Overclaimed the CD4-native collapse as proof that end-to-end halting fails.**
The from-scratch CD4 model's depth collapsed, and the first write-up framed this as "the
identifiability wall reached from the opposite direction," implying end-to-end halting is
intrinsically unidentifiable. But we only ever tested **pseudobulk**, never single-cell
from scratch — and the frozen single-cell result *did* recover a signature. Claiming more
than the data supports. **Caught by the user** ("the collapse [is] because of data, we
haven't tested this on actual single cell so we can't say this honestly"). Fixed by
retitling the section "pseudobulk hides the signature" and stating the narrow, honest claim.

**M3 — Wrong cell count written into the paper (~22M vs 3.9M).**
The rewritten CD4-native section stated the single-cell screen was "~22M cells" as the
reason the control was infeasible. The subsample actually used was 3.9M cells (two donors,
capped at 50 cells/perturbation); ~22M is the *full* screen. Writing the wrong magnitude
into a durable artifact. **Caught by an automated auditor** and clarified by the user (full
screen ~22M, analysis used a 3.9M subsample). Fixed by stating both numbers explicitly.

**M4 — Model card led with the failed model as the headline.**
The HuggingFace card's H1 was "CD4-native STATE transition model with fused halting" — i.e.
it led with the model whose depth *collapsed*, not the working frozen halt heads. **Caught
by the user.** Fixed by retitling to lead with the method and clearly separating the ✅
working heads from the ⚠️ documented negative.

**M5 — Added hand-coded biological rules where data-driven analysis was asked for.**
At one point perturbations were categorized with hand-written rules and depth signatures
were annotated by pasting descriptive strings onto them. The user had asked for
data-driven clustering (UMAP + Leiden on the depth signature itself). **Caught, bluntly,
by the user.** Fixed by clustering on the signature and cross-referencing the user's own
annotation CSVs — which is what surfaced the real result (drug/toxicity targets concentrate
in the translation/ribosome clusters at OR 2.8–9.8).

**M6 — Nearly wrote "depth is model-specific, not a portable property" as a headline.**
This phrasing would have confused readers, because the paper's actual finding is that depth
IS a property of the perturbation *within a context* — it's cell-type-specific, not
model-specific-in-a-dismissive-sense. **Caught by the user** before it went in.

### 4.2 Engineering mistakes (cost time, not correctness)

**M7 — Fought a broken local LaTeX toolchain for too long.**
The local conda TeX Live was engine-only (no `latex.ltx`, broken format-build via a missing
Perl script). Several attempts were spent trying to repair it and to install a conda TeX
scheme that doesn't exist on the channel, before pivoting to the reliable route: install
TeX Live on the remote Linux host via apt and compile there. The pivot should have come
sooner. (This is why the final compile was delegated to a sub-agent with both routes
documented.)

**M8 — First single-cell training run stalled undiagnosed for ~50 minutes.**
The from-scratch single-cell dataloader ran 50 min / 0 steps. Root cause was an h5ad
chunking pathology: gzip chunks of shape (15178, 8) meant reading one cell's 2001-dim row
decompressed ~250 chunks — a ~16,000× over-read for a 64-cell batch. Diagnosing this took
too long; the fix (rewrite row-contiguous, uncompressed, on /dev/shm) was simple once found.
This ultimately forced the pseudobulk pivot — which then caused the depth collapse (M2).

**M9 — Halt-head weights weren't saved on the first pass.**
The trained halt heads were computed but not persisted, so they had to be **refit** from
scratch later to ship them. **Caught by the user** ("trained halt-head weights are NOT
saved?"). Fixed by a refit sub-agent that reproduced E[N] at ρ=1.0 and saved all four.

**M10 — Baked effect-size dependence into the halting loss early on.**
An early per-step halting loss used the full-magnitude response as the target, which made
depth a pure effect-size proxy by construction. This was a real bug in the objective, not
just framing. Fixed by switching to a **magnitude-free** (cosine-direction) per-round
target — one of the two ingredients that eventually cleared the identifiability wall.

### 4.3 Process mistakes

**M11 — Occasionally did mechanical work in the main thread instead of delegating.**
The user twice prompted "why don't you assign this to a subagent?" for work that was
bloating the main context. Delegation should have been the default for long mechanical jobs.

**M12 — Reported status imprecisely under questioning.**
Several times the answer to "how many steps / how long are you training?" conflated
"max_steps budget" with "train to convergence." The user had to disambiguate repeatedly.

---

## 5. What went right (so the list above is in proportion)

- **The negatives are real and were kept.** The identifiability wall, the cross-context
  non-portability, and the drug non-additivity are honest findings, not buried failures.
  The project's spine — "a clean negative is the finding" — held.
- **Every number is grounded.** Reported statistics trace to saved tables and were
  recomputed, not recalled.
- **The working artifacts reproduce.** The four frozen halt heads reproduce E[N] at ρ=1.0
  and are shipped with metadata; the code and models are public on GitHub and HuggingFace.
- **User corrections were incorporated, not defended.** Every mistake in §4.1 was caught
  in review and fixed rather than rationalized.

---

## 6. Lessons for the next project

1. **Watch for words that carry two meanings** ("redundant"). In a paper making a subtle
   novelty claim, one ambiguous word can state the opposite of your thesis.
2. **Never let a claim exceed the experiment.** If only pseudobulk was tested, the claim is
   about pseudobulk — full stop.
3. **Verify magnitudes against the source** before writing them into a durable artifact,
   especially round numbers like "22M."
4. **Lead every deliverable with what works,** and label negatives as negatives.
5. **Pivot off broken infrastructure faster** — a known-good remote route beats an hour
   repairing a local one.
6. **Persist expensive artifacts immediately** (trained weights, checkpoints).
7. **Delegate mechanical and long-running work by default.**
8. **Diagnose data-loading stalls early** — chunking/layout pathologies are common and
   cheap to fix once identified.

---

*This retrospective is itself a Claude Science artifact, versioned alongside the paper,
code, and models it describes.*
