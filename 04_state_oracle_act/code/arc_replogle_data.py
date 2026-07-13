"""arc_replogle_data.py — ARC's NATIVE k562 eval set for the ST-SE early-exit graft.
adata_real.h5ad: X = 2000 HVG expr, obsm[X_state] = 2058-d SE embedding (model input), obs[gene] = pert.
Everything is already in the model's exact space (no transfer gap, no SE embedding step)."""
import numpy as np, h5py, glob, torch, os

def _read_cat(f, col):
    g = f["obs"][col]
    if isinstance(g, h5py.Group):  # categorical
        cats = np.array([c.decode() if isinstance(c,bytes) else str(c) for c in g["categories"][:]])
        codes = g["codes"][:]
        return cats[codes]
    v = g[:]
    return np.array([x.decode() if isinstance(x,bytes) else x for x in v])

def build(cache_glob="/workspace/halt/hf_cache_se", ckpt_glob="/workspace/halt/hf_cache_se",
          control_label="non-targeting", min_cells=20, cell_set_len=64,
          data_dir="/dev/shm/k562data", cell_line="k562", split="fewshot",
          cell_half=None, split_seed=12345):
    # Prefer locally-staged copies (avoids MooseFS/FUSE recursive-glob + per-read stalls on the
    # 2.2GB h5ad). Fall back to the HF-cache glob only if the staged dir is absent.
    if data_dir and os.path.exists(os.path.join(data_dir, "adata_real.h5ad")):
        real = os.path.join(data_dir, "adata_real.h5ad")
        pmap_path = os.path.join(data_dir, "pert_onehot_map.pt")
        bmap_path = os.path.join(data_dir, "batch_onehot_map.pkl")
    else:
        real = glob.glob(f"{cache_glob}/**/{split}/{cell_line}/eval_best.ckpt/adata_real.h5ad", recursive=True)[0]
        pmap_path = glob.glob(f"{ckpt_glob}/**/{split}/{cell_line}/pert_onehot_map.pt", recursive=True)[0]
        bmap_path = glob.glob(f"{ckpt_glob}/**/{split}/{cell_line}/batch_onehot_map.pkl", recursive=True)[0]
    pmap = torch.load(pmap_path, map_location="cpu", weights_only=False)
    pmap = {str(k): (v.argmax().item() if hasattr(v,"argmax") else int(v)) for k,v in pmap.items()}
    pert_dim = 2024
    f = h5py.File(real, "r")
    X = f["X"][:].astype(np.float32)                 # (N, 2000) HVG expression
    Xs = f["obsm"]["X_state"][:].astype(np.float32)  # (N, 2058) SE embedding (model input)
    gene = _read_cat(f, "gene")                       # (N,) perturbation label
    # batch (gem_group) -> onehot index via ARC batch map
    try:
        import pickle
        bpath = bmap_path if (bmap_path and os.path.exists(bmap_path)) else glob.glob(f"{ckpt_glob}/**/{split}/{cell_line}/batch_onehot_map.pkl", recursive=True)[0]
        bmap = pickle.load(open(bpath,"rb"))
        gg = _read_cat(f, "gem_group")
        bmap = {str(k):(v.argmax() if hasattr(v,"argmax") else int(v)) for k,v in bmap.items()}
        batch_idx_all = np.array([int(bmap.get(str(x),0)) for x in gg])
    except Exception:
        batch_idx_all = np.zeros(len(gene), dtype=int)
    f.close()
    # find control label present in data
    labels = set(np.unique(gene))
    ctrl_label = control_label if control_label in labels else \
        next((c for c in ["non-targeting","control","NTC","neg"] if c in labels), None)
    ctrl = gene == ctrl_label
    if ctrl.sum() == 0: raise ValueError(f"no controls; labels sample {list(labels)[:10]}")
    ctrl_mean = X[ctrl].mean(0)                        # basal gene expr reference
    Xs_ctrl = Xs[ctrl]                                 # control SE embeddings (basal input pool)
    batch_ctrl = batch_idx_all[ctrl]
    per_pert = {}
    _sh_rng = np.random.default_rng(split_seed)
    for g in np.unique(gene):
        if g == ctrl_label or g not in pmap: continue
        m = gene == g
        if m.sum() < min_cells: continue
        # cell_half: deterministically split this perturbation's cells into two disjoint halves
        # (A/B) using a FIXED per-gene seed so half A and half B are identical across two build()
        # calls that differ only in cell_half. The response target is re-estimated on the chosen
        # half -> training the halt head on half A vs half B is the clean learned-E[N] split-half test.
        if cell_half in ("A", "B"):
            idx = np.where(m)[0]; perm = _sh_rng.permutation(idx); h = len(perm)//2
            sel = perm[:h] if cell_half == "A" else perm[h:2*h]
            m = np.zeros_like(m); m[sel] = True
            if m.sum() < max(10, min_cells//2): continue
        resp = X[m].mean(0) - ctrl_mean               # control-relative response (gene space)
        per_pert[g] = dict(onehot_idx=pmap[g], resp=resp.astype(np.float32),
                           n=int(m.sum()), eff=float(np.linalg.norm(resp)),
                           n_de=int((np.abs(resp)>0.1).sum()))
    return dict(ctrl_mean=ctrl_mean.astype(np.float32), Xs_ctrl=Xs_ctrl, batch_ctrl=batch_ctrl,
                per_pert=per_pert, pert_dim=pert_dim, cell_set_len=cell_set_len,
                input_dim=Xs.shape[1], n_genes=X.shape[1])
