"""norm_data_lean.py — per-perturbation UNIT response direction + covariates, NO graph.
target_dir = normalize(mean_expr(pert) - ctrl_mean) on log1p-normalized HVG space.
Covariates for the non-redundancy regression: effect_size (||delta||), n_de, n_cells."""
import numpy as np, scanpy as sc, anndata as ad
from scipy.sparse import issparse

def build(h5ad, n_genes=2000, min_cells=20, seed=0):
    A = sc.read_h5ad(h5ad)
    if "counts" in A.layers: A.X = A.layers["counts"].copy()
    sc.pp.normalize_total(A, target_sum=1e4); sc.pp.log1p(A)
    # HVG subset
    sc.pp.highly_variable_genes(A, n_top_genes=n_genes, flavor="seurat")
    A = A[:, A.var["highly_variable"]].copy()
    X = A.X.toarray() if issparse(A.X) else np.asarray(A.X)
    tg = A.obs["target_gene"].astype(str).to_numpy()
    ctrl = X[tg == "control"]
    ctrl_mean = ctrl.mean(0)
    ctrl_std = ctrl.std(0) + 1e-8
    perts, target_dir, eff, nde, ncells = [], [], [], [], []
    for g in sorted(set(tg)):
        if g in ("control","*",""): continue
        m = tg == g
        if m.sum() < min_cells: continue
        delta = X[m].mean(0) - ctrl_mean
        mag = float(np.linalg.norm(delta))
        if mag < 1e-6: continue
        perts.append(g); target_dir.append(delta/mag)
        eff.append(mag)
        nde.append(int((np.abs(delta) > 2*ctrl_std).sum()))
        ncells.append(int(m.sum()))
    return dict(perts=perts, target_dir=np.array(target_dir, dtype=np.float32),
                effect_size=np.array(eff, dtype=np.float32),
                n_de=np.array(nde), n_cells=np.array(ncells),
                n_genes=target_dir[0].shape[0])
