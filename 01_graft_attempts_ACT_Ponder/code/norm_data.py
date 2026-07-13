"""
norm_data.py — build graph + per-perturbation UNIT response direction + covariates.

Per perturbation p:
  target_dir[p] = normalize( mean_expr(cells_p) - ctrl_mean )     # magnitude-free target
  effect_size[p] = || mean_expr(cells_p) - ctrl_mean ||           # the magnitude we regress out
  n_de[p]       = # genes with |delta| > 2*std(control gene)      # DE-count covariate
  n_cells[p], graph_degree[p]                                     # technical / topology covariates

The model is trained per-CELL but the halting TARGET is the perturbation's shared unit direction,
so E[hops] is a per-perturbation property (how hard the DIRECTION is to reach), not per-cell noise.
"""
import numpy as np, anndata as ad, scipy.sparse as sp


def build(h5ad, n_nodes=2000, knn=16, ctrl_subsample=4000, seed=0):
    rng = np.random.default_rng(seed)
    A = ad.read_h5ad(h5ad)
    X = A.layers["counts"] if "counts" in A.layers else A.X
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    lib = np.asarray(X.sum(1)).ravel(); lib[lib == 0] = 1
    Xn = X.multiply((1e4 / lib)[:, None]).tocsr(); Xn.data = np.log1p(Xn.data)

    var = np.array(A.var_names.astype(str))
    tg = A.obs["target_gene"].astype(str).values
    pert_genes = [g for g in np.unique(tg) if g not in ("control", "*")]
    var_set = {g: i for i, g in enumerate(var)}
    measured_pert = [g for g in pert_genes if g in var_set]

    gene_var = np.asarray(Xn.power(2).mean(0)).ravel() - np.asarray(Xn.mean(0)).ravel() ** 2
    order = np.argsort(-gene_var)
    node_idx, seen = [], set()
    for g in measured_pert:
        i = var_set[g]; node_idx.append(i); seen.add(i)
    for i in order:
        if len(node_idx) >= n_nodes: break
        if i not in seen: node_idx.append(int(i)); seen.add(int(i))
    node_idx = np.array(node_idx); node_genes = var[node_idx]
    gene_to_node = {g: k for k, g in enumerate(node_genes)}
    Xnode = Xn[:, node_idx].toarray().astype(np.float32)

    ctrl_mask = tg == "control"
    ctrl = Xnode[ctrl_mask]
    ctrl_mean = ctrl.mean(0)
    ctrl_std = ctrl.std(0); ctrl_std[ctrl_std == 0] = 1e-6

    # co-expression kNN graph (on control cells)
    Cc = ctrl.copy()
    if Cc.shape[0] > ctrl_subsample:
        Cc = Cc[rng.choice(Cc.shape[0], ctrl_subsample, replace=False)]
    Cc = Cc - Cc.mean(0, keepdims=True)
    sd = Cc.std(0, keepdims=True); sd[sd == 0] = 1; Cc = Cc / sd
    corr = (Cc.T @ Cc) / Cc.shape[0]; np.fill_diagonal(corr, -np.inf)
    nn_idx = np.argpartition(-corr, knn, axis=1)[:, :knn]
    src = np.repeat(np.arange(len(node_genes)), knn); dst = nn_idx.ravel()
    edge_index = np.stack([np.concatenate([src, dst]),
                           np.concatenate([dst, src])], axis=0).astype(np.int64)
    deg = np.bincount(edge_index[0], minlength=len(node_genes))

    # per-perturbation direction + covariates
    perts, target_dir, effect_size, n_de, n_cells, node_of = [], [], [], [], [], []
    for g in measured_pert:
        cells = tg == g
        if cells.sum() < 5:  # need a few cells for a stable mean
            continue
        delta = Xnode[cells].mean(0) - ctrl_mean            # (n_nodes,)
        mag = float(np.linalg.norm(delta))
        if mag < 1e-6:
            continue
        perts.append(g); node_of.append(gene_to_node[g])
        target_dir.append(delta / mag); effect_size.append(mag)
        n_de.append(int((np.abs(delta) > 2 * ctrl_std).sum()))
        n_cells.append(int(cells.sum()))
    return dict(
        Xnode=Xnode, node_genes=node_genes, gene_to_node=gene_to_node,
        edge_index=edge_index, deg=deg, tg=tg, ctrl_mean=ctrl_mean,
        perts=perts, node_of=np.array(node_of),
        target_dir=np.array(target_dir, dtype=np.float32),
        effect_size=np.array(effect_size), n_de=np.array(n_de),
        n_cells=np.array(n_cells))
