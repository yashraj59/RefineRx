"""refit_all.py — RE-FIT and PERSIST the halt-head weights for all 4 Replogle lines.

Reproduces EXACTLY the head + training protocol of train_oracle_refine.py (the code that produced the
saved xline_fewshot_{line}/EN_seeds.npy signatures): AdaptiveStateRefine (learnable refinement token +
sequential-hazard halt head + softplus error head) on the FROZEN ST-SE backbone, Stage-C oracle_loss
with tau=0.05, alpha=0.5, beta=1.0, gamma=0.1, delta=0.1, warmup=15, epochs=50, S=64, lr=3e-3, 4 seeds.

The ONLY additions vs the original trainer:
  (1) persist the HEAD weight tensors (refinement_token + halt_head.* + error_head.*, NOT the frozen
      backbone) for every seed -> halthead_{line}.pt.
  (2) verify reproduction: correlate re-fit E[N] vs SAVED E[N] (xline_fewshot_{line}) per line.

D_r SOURCE: recomputed on-the-fly from frozen-backbone forward passes (as in the original). There is NO
disk D_r cache; resp_{line}.npz caches only the response TARGETS (control-relative HVG means), not D_r,
and the head consumes the backbone's learnable-refinement-token hidden states, so backbone forwards are
required. Runs on CPU -> zero GPU contention with the concurrent CD4 run.
"""
import sys, os, json, time, gc
import numpy as np, torch
from scipy.stats import spearmanr
sys.path.insert(0, "/workspace/halt/actrun_state/code")
sys.path.insert(0, "/workspace/halt/state_repo/src")
from load_arc_state import load_arc
from adaptive_state_refine import AdaptiveStateRefine
from arc_replogle_data import build

DEV = os.environ.get("REFIT_DEV", "cpu")
CACHE = "/workspace/halt/hf_cache_stse_all"
SAVED = "/workspace/halt/actrun_state"   # xline_fewshot_{line}/EN_seeds.npy
OUT   = "/workspace/halt/actrun_state/halthead_weights"

# protocol hparams (identical to train_oracle_refine.py defaults, verified against saved provenance JSONs)
EPOCHS = int(os.environ.get("REFIT_EPOCHS", "50"))
WARMUP = int(os.environ.get("REFIT_WARMUP", "15"))
SEEDS  = int(os.environ.get("REFIT_SEEDS", "4"))
S, LR = 64, 3e-3
TAU, ALPHA, BETA, GAMMA, DELTA = 0.05, 0.5, 1.0, 0.1, 0.1
MIN_CELLS = 20
SMOKE = os.environ.get("REFIT_SMOKE", "0") == "1"

torch.set_num_threads(int(os.environ.get("REFIT_THREADS", "24")))

def refit_line(line):
    t0 = time.time()
    data_dir = f"/dev/shm/{line}data"
    print(f"\n===== LINE {line} =====", flush=True)
    arc, vd, hp, mm = load_arc(CACHE, device=DEV, cell_line=line, split="fewshot")
    D = build(cache_glob=CACHE, ckpt_glob=CACHE, min_cells=MIN_CELLS, cell_set_len=S,
              data_dir=data_dir, cell_line=line, split="fewshot", cell_half=None)
    perts = list(D["per_pert"].keys()); P = len(perts)
    rng0 = np.random.default_rng(1234); perm = rng0.permutation(P)
    val_idx = set(perm[:max(1, P // 5)].tolist())
    val_mask = np.array([i in val_idx for i in range(P)])
    print(f"[data] line={line} perts={P} val={val_mask.sum()} train={(~val_mask).sum()} "
          f"input_dim={D['input_dim']} n_genes={D['n_genes']} pert_dim={D['pert_dim']}", flush=True)

    def batch(bp, rng):
        ctrl = D["Xs_ctrl"]; bc = D["batch_ctrl"]; B = len(bp)
        sel = [rng.choice(len(ctrl), S, replace=len(ctrl) < S) for _ in bp]
        basal = np.stack([ctrl[s] for s in sel]); bidx = np.stack([bc[s] for s in sel])
        pert = np.zeros((B, S, D["pert_dim"]), dtype=np.float32)
        for i, g in enumerate(bp): pert[i, :, D["per_pert"][g]["onehot_idx"]] = 1.0
        resp = np.stack([D["per_pert"][g]["resp"] for g in bp])
        td = resp / (np.linalg.norm(resp, axis=1, keepdims=True) + 1e-8)
        td = np.repeat(td[:, None, :], S, axis=1)
        bg = np.broadcast_to(D["ctrl_mean"], (B, S, D["n_genes"]))
        return (torch.tensor(pert, device=DEV), torch.tensor(basal, device=DEV),
                torch.tensor(bidx, dtype=torch.long, device=DEV),
                torch.tensor(td, dtype=torch.float32, device=DEV),
                torch.tensor(np.ascontiguousarray(bg), dtype=torch.float32, device=DEV))

    # per-layer argmin-depth + oracle r* reference (reporting only; not used in training)
    mref = AdaptiveStateRefine(arc).to(DEV); R = mref.num_rounds
    ad = np.zeros((4, P)); orc = np.zeros((4, P))
    for d in range(4):
        rng = np.random.default_rng(700 + d); off = 0
        for i in range(0, P, 32):
            bp = perts[i:i + 32]; pe, ba, bi, td, bg = batch(bp, rng)
            _, _, _, scd = mref.eval_signature(pe, ba, bi, td, bg)
            ad[d, off:off + len(bp)] = scd.argmin(1) + 1
            orc[d, off:off + len(bp)] = mref.oracle_round(torch.tensor(scd), tau=TAU).numpy() + 1
            off += len(bp)
    argmin_depth = ad.mean(0); oracle_star = orc.mean(0)

    ENs = []; HCs = []; head_sds = {}
    for s in range(SEEDS):
        torch.manual_seed(s); np.random.seed(s); rng = np.random.default_rng(s)
        model = AdaptiveStateRefine(arc).to(DEV)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
        tr_perts = [perts[i] for i in range(P) if not val_mask[i]]
        for ep in range(EPOCHS):
            model.train(); order = rng.permutation(len(tr_perts)); use_ponder = (ep >= WARMUP)
            for i in range(0, len(tr_perts), 32):
                bp = [tr_perts[j] for j in order[i:i + 32]]; pe, ba, bi, td, bg = batch(bp, rng)
                out = model.oracle_loss(pe, ba, bi, td, bg, tau=TAU, alpha=ALPHA, beta=BETA,
                                        gamma=GAMMA, delta=DELTA, use_ponder=use_ponder)
                opt.zero_grad(); out["total"].backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
        model.eval(); EN = np.zeros(P); HC = np.zeros(P); off = 0; rng2 = np.random.default_rng(9000 + s)
        for i in range(0, P, 32):
            bp = perts[i:i + 32]; pe, ba, bi, td, bg = batch(bp, rng2)
            en, hc, _, _ = model.eval_signature(pe, ba, bi, td, bg)
            EN[off:off + len(bp)] = en; HC[off:off + len(bp)] = hc; off += len(bp)
        ENs.append(EN); HCs.append(HC)
        # persist ONLY head params (exclude frozen backbone arc.*)
        head_sds[f"seed_{s}"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                                 if not k.startswith("arc.")}
        print(f"[seed {s}] E[N] mean={EN.mean():.3f} std={EN.std():.3f} "
              f"range=[{EN.min():.2f},{EN.max():.2f}] rho(EN,oracle)={spearmanr(EN,oracle_star)[0]:.3f}",
              flush=True)
        del model, opt; gc.collect()

    ENs = np.array(ENs); HCs = np.array(HCs); Em = ENs.mean(0); Hm = HCs.mean(0)
    pair = [spearmanr(ENs[i], ENs[j])[0] for i in range(SEEDS) for j in range(i + 1, SEEDS)
            if ENs[i].std() > 0 and ENs[j].std() > 0]
    repro = float(np.mean(pair)) if pair else float("nan")

    # ---- VERIFY reproduction vs SAVED xline E[N] (align by perturbation name) ----
    saved_seeds = np.load(f"{SAVED}/xline_fewshot_{line}/EN_seeds.npy")   # (4, P_saved)
    saved_Em = saved_seeds.mean(0)
    saved_names = []
    with open(f"{SAVED}/xline_fewshot_{line}/oracle_signature.csv") as fcsv:
        next(fcsv)
        for row in fcsv: saved_names.append(row.split(",")[0])
    saved_names = np.array(saved_names)
    assert len(saved_names) == saved_Em.shape[0], (len(saved_names), saved_Em.shape)
    name2saved = {n: saved_Em[i] for i, n in enumerate(saved_names)}
    common = [g for g in perts if g in name2saved]
    my_al = np.array([Em[perts.index(g)] for g in common])
    sv_al = np.array([name2saved[g] for g in common])
    rho_repro = float(spearmanr(my_al, sv_al)[0])
    pear = float(np.corrcoef(my_al, sv_al)[0, 1])
    # per-seed rho vs saved mean
    perseed_rho = [float(spearmanr(np.array([ENs[s][perts.index(g)] for g in common]), sv_al)[0])
                   for s in range(SEEDS)]
    print(f"[VERIFY {line}] rho(refit_Em, saved_Em)={rho_repro:.4f} pearson={pear:.4f} "
          f"n_common={len(common)}/{P} perseed_rho={[round(x,3) for x in perseed_rho]}", flush=True)

    # ---- SAVE weights + meta ----
    config = dict(head_module="AdaptiveStateRefine", H=int(mref.H), n_layers=int(mref.n_layers),
                  exit_layers=[int(x) for x in mref.exit_layers], num_rounds=int(R), error_head=True,
                  halt_head="Sequential(LayerNorm(H), Linear(H,64), SiLU, Linear(64,1)); last-bias init -2.0",
                  error_head_arch="Sequential(LayerNorm(H), Linear(H,64), SiLU, Linear(64,1)) + softplus",
                  refinement_token="Parameter(H) init randn*0.02, appended at seq index S",
                  pert_dim=int(D["pert_dim"]), input_dim=int(D["input_dim"]), n_genes=int(D["n_genes"]))
    torch.save({**head_sds, "config": config,
                "hparams": dict(epochs=EPOCHS, warmup=WARMUP, seeds=SEEDS, S=S, lr=LR, tau=TAU,
                                alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA, min_cells=MIN_CELLS),
                "perts": np.array(perts)},
               f"{OUT}/halthead_{line}.pt")
    np.save(f"{OUT}/refit_EN_seeds_{line}.npy", ENs)

    meta = dict(
        cell_line=line, split="fewshot", backbone_ckpt=f"{CACHE}/fewshot/{line}/checkpoints/best.ckpt",
        backbone="ST-SE-Replogle llama (frozen, no grad)", n_perturbations=int(P), num_rounds=int(R),
        architecture=config,
        hparams=dict(epochs=EPOCHS, warmup=WARMUP, n_seeds=SEEDS, cell_set_len_S=S, lr=LR, tau=TAU,
                     alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA, min_cells=MIN_CELLS,
                     optimizer="AdamW", grad_clip=1.0, ponder_gated_after_warmup=True, KL=0),
        seeds=list(range(SEEDS)),
        input_feature_spec=dict(
            basal_state="obsm[X_state] SE embedding (input_dim=%d), control cells sampled S=%d/pert" % (D["input_dim"], S),
            pert="one-hot over pert_dim=%d via pert_onehot_map.pt" % D["pert_dim"],
            batch="gem_group via batch_onehot_map.pkl",
            target="magnitude-free: L2-normalized control-relative HVG response direction (n_genes=%d)" % D["n_genes"],
            Dr="per-round cosine distance between round-r decoded prediction direction and target direction (computed from frozen backbone)"),
        dr_source="recomputed on-the-fly from frozen-backbone forward passes (no disk D_r cache; resp_{line}.npz caches response TARGETS only)",
        device=DEV,
        weight_contents="per-seed head state_dicts under keys seed_0..seed_%d (refinement_token + halt_head.* + error_head.*); frozen backbone NOT included. E[N] signature = mean over seeds." % (SEEDS - 1),
        reproduction=dict(rho_spearman_refit_vs_saved=rho_repro, pearson_refit_vs_saved=pear,
                          per_seed_rho_vs_saved=perseed_rho, n_common_perts=len(common),
                          refit_EN_mean=float(Em.mean()), refit_EN_std=float(Em.std()),
                          refit_EN_range=[float(Em.min()), float(Em.max())],
                          saved_EN_mean=float(saved_Em.mean()), saved_EN_std=float(saved_Em.std()),
                          refit_seed_reproducibility_spearman=repro),
        elapsed_sec=round(time.time() - t0, 1))
    json.dump(meta, open(f"{OUT}/halthead_{line}_meta.json", "w"), indent=2)
    print(f"[SAVED {line}] {OUT}/halthead_{line}.pt (+meta) elapsed={meta['elapsed_sec']}s", flush=True)
    open(f"{OUT}/DONE_{line}", "w").write("ok\n")
    del arc, mref, D; gc.collect()
    return dict(line=line, rho=rho_repro, pearson=pear, repro=repro, P=P,
                refit_EN_mean=float(Em.mean()), saved_EN_mean=float(saved_Em.mean()))

def main():
    global OUT
    if SMOKE:
        OUT = OUT + "_smoke"
    os.makedirs(OUT, exist_ok=True)
    lines = sys.argv[1:] if len(sys.argv) > 1 else ["k562", "hepg2", "jurkat", "rpe1"]
    summary = {}
    for L in lines:
        try:
            summary[L] = refit_line(L)
        except Exception as e:
            import traceback; traceback.print_exc()
            summary[L] = dict(line=L, error=str(e))
            open(f"{OUT}/DONE_{L}", "w").write(f"error: {e}\n")
    json.dump(summary, open(f"{OUT}/refit_summary.json", "w"), indent=2)
    print("\n[ALL RESULT]", json.dumps(summary), flush=True)
    open(f"{OUT}/DONE_ALL", "w").write("ok\n")
    print("[done]", flush=True)

if __name__ == "__main__":
    main()
