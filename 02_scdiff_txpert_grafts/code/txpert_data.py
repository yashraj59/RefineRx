"""
txpert_data.py — build the gene-graph node set, co-expression edges, per-cell targets,
and the two confound covariates (effect size + graph degree) for the adaptive-TxPert experiment.

Node set = (measured perturbed genes) UNION (top-variable genes) up to n_nodes, so nearly every
perturbation's target gene is an actual node in the message-passing graph (only 9/90 perturbed
genes are in the default HVG set, so we must force them in).
"""
import numpy as np, anndata as ad, scipy.sparse as sp


def build(h5ad, n_nodes=2000, knn=16, ctrl_subsample=4000, seed=0):
    rng = np.random.default_rng(seed)
    A = ad.read_h5ad(h5ad)
    # counts -> library-normalized log1p
    X = A.layers["counts"] if "counts" in A.layers else A.X
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    lib = np.asarray(X.sum(1)).ravel(); lib[lib == 0] = 1
    Xn = X.multiply((1e4 / lib)[:, None]).tocsr(); Xn.data = np.log1p(Xn.data)

    var = np.array(A.var_names.astype(str))
    tg = A.obs["target_gene"].astype(str).values
    pert_genes = [g for g in np.unique(tg) if g not in ("control", "*")]
    var_set = {g: i for i, g in enumerate(var)}
    measured_pert = [g for g in pert_genes if g in var_set]     # 89 of 90

    # pick node genes: all measured perturbed genes + top-variance fill
    gene_var = np.asarray(Xn.power(2).mean(0)).ravel() - np.asarray(Xn.mean(0)).ravel() ** 2
    order = np.argsort(-gene_var)
    node_idx, seen = [], set()
    for g in measured_pert:                                     # force perturbed genes in
        i = var_set[g]; node_idx.append(i); seen.add(i)
    for i in order:                                            # fill with most-variable
        if len(node_idx) >= n_nodes: break
        if i not in seen: node_idx.append(int(i)); seen.add(int(i))
    node_idx = np.array(node_idx)
    node_genes = var[node_idx]
    gene_to_node = {g: k for k, g in enumerate(node_genes)}

    Xnode = Xn[:, node_idx].toarray().astype(np.float32)       # (cells, n_nodes)

    # co-expression kNN graph on control cells
    ctrl_mask = tg == "control"
    Cc = Xnode[ctrl_mask]
    if Cc.shape[0] > ctrl_subsample:
        Cc = Cc[rng.choice(Cc.shape[0], ctrl_subsample, replace=False)]
    Cc = Cc - Cc.mean(0, keepdims=True)
    sd = Cc.std(0, keepdims=True); sd[sd == 0] = 1; Cc = Cc / sd
    corr = (Cc.T @ Cc) / Cc.shape[0]                           # (n_nodes, n_nodes)
    np.fill_diagonal(corr, -np.inf)
    nn_idx = np.argpartition(-corr, knn, axis=1)[:, :knn]      # top-k neighbors
    src = np.repeat(np.arange(len(node_genes)), knn)
    dst = nn_idx.ravel()
    edge_index = np.stack([np.concatenate([src, dst]),        # symmetric
                           np.concatenate([dst, src])], axis=0).astype(np.int64)

    # graph degree per node (confound covariate)
    deg = np.bincount(edge_index[0], minlength=len(node_genes))

    # per-cell target + perturbed-node index; drop cells whose target gene isn't a node
    pert_node = np.array([gene_to_node.get(g, -1) for g in tg])
    keep = (pert_node >= 0) & (tg != "control")               # perturbed cells with node targets
    ctrl_mean = Xnode[ctrl_mask].mean(0)                      # effect-size reference

    return dict(
        Xnode=Xnode, node_genes=node_genes, gene_to_node=gene_to_node,
        edge_index=edge_index, deg=deg, tg=tg, pert_node=pert_node,
        keep_idx=np.where(keep)[0], ctrl_mean=ctrl_mean,
        measured_pert=measured_pert)
