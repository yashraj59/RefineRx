"""
train_norm.py — the thesis protocol on a MAGNITUDE-FREE halting target.

Trains the adaptive refiner to predict each perturbation's UNIT response direction, over S seeds.
Readouts:
  (1) REPRODUCIBILITY: mean pairwise Spearman of the per-perturbation E[hops] ranking across seeds.
  (2) NON-REDUNDANCY: regress E[hops] on [effect_size, n_de, n_cells, graph_degree]; report R^2
      (how much of E[hops] is explained by those covariates) and residual std. Partial Spearman of
      E[hops] vs effect_size controlling the rest. If E[hops] is just effect size, R^2 -> 1 and the
      partial-out residual is noise; if it carries independent structure, residual survives AND is
      reproducible across seeds.
  (3) The decisive combination: is the RESIDUAL (E[hops] after removing covariates) itself
      reproducible across seeds? That is the thesis's non-redundant + reproducible bar.
"""
import sys, os, json, argparse, time
import numpy as np, torch
from norm_data import build
from adaptive_refiner_norm import AdaptiveRefinerNorm
from scipy.stats import spearmanr

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_seed(D, edge_index, seed, epochs, n_hops, d, lr, pw, lp, pb):
    torch.manual_seed(seed); np.random.seed(seed)
    n_nodes = len(D['node_genes'])
    model = AdaptiveRefinerNorm(n_nodes, d=d, n_hops=n_hops, ponder_weight=pw,
                                lambda_prior=lp, ponder_beta=pb).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    pert_node = torch.tensor(D['node_of'], device=DEV)
    target_dir = torch.tensor(D['target_dir'], device=DEV)
    P = len(D['perts'])
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(P, device=DEV)
        # full-batch over perturbations (P is small ~89)
        out = model(pert_node[idx], target_dir[idx], edge_index)
        opt.zero_grad(); out['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    sig = model.signature(edge_index)
    E_all = sig['E_N'].numpy()
    return E_all[D['node_of']]   # E[hops] per perturbation (P,)


def partial_spearman(y, x, controls):
    """Spearman(y, x) after linearly regressing 'controls' out of both (on ranks)."""
    from scipy.stats import rankdata
    def resid(v):
        Y = rankdata(v); Cm = np.column_stack([rankdata(c) for c in controls])
        Cm = np.column_stack([np.ones(len(Y)), Cm])
        beta, *_ = np.linalg.lstsq(Cm, Y, rcond=None)
        return Y - Cm @ beta
    ry, rx = resid(y), resid(x)
    if ry.std() < 1e-9 or rx.std() < 1e-9: return 0.0
    return float(spearmanr(ry, rx)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5ad', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--n_nodes', type=int, default=2000); ap.add_argument('--knn', type=int, default=16)
    ap.add_argument('--n_hops', type=int, default=8); ap.add_argument('--d', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=400); ap.add_argument('--lr', type=float, default=5e-3)
    ap.add_argument('--ponder_weight', type=float, default=0.1)
    ap.add_argument('--lambda_prior', type=float, default=0.2)
    ap.add_argument('--ponder_beta', type=float, default=0.05)
    ap.add_argument('--seeds', type=int, default=4); ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)

    print(f'[setup] device={DEV} torch={torch.__version__}', flush=True)
    D = build(args.h5ad, n_nodes=args.n_nodes, knn=args.knn)
    P = len(D['perts'])
    print(f"[data] nodes={len(D['node_genes'])} perts={P} edges={D['edge_index'].shape[1]}", flush=True)
    edge_index = torch.tensor(D['edge_index'], device=DEV)
    if args.smoke:
        args.epochs = 40; args.seeds = 2

    # ---- train S seeds ----
    EN_seeds = []
    for s in range(args.seeds):
        t0 = time.time()
        EN = train_seed(D, edge_index, seed=s, epochs=args.epochs, n_hops=args.n_hops,
                        d=args.d, lr=args.lr, pw=args.ponder_weight,
                        lp=args.lambda_prior, pb=args.ponder_beta)
        EN_seeds.append(EN)
        print(f"[seed {s}] E[N] mean={EN.mean():.3f} std={EN.std():.3f} "
              f"range=[{EN.min():.3f},{EN.max():.3f}] ({time.time()-t0:.1f}s)", flush=True)
    EN_seeds = np.array(EN_seeds)               # (S, P)
    EN_mean = EN_seeds.mean(0)

    # ---- (1) reproducibility across seeds ----
    pair_rhos = []
    for i in range(args.seeds):
        for j in range(i+1, args.seeds):
            if EN_seeds[i].std()>0 and EN_seeds[j].std()>0:
                pair_rhos.append(spearmanr(EN_seeds[i], EN_seeds[j])[0])
    repro = float(np.mean(pair_rhos)) if pair_rhos else float('nan')

    # ---- (2) non-redundancy: regress E[N] on covariates ----
    eff, nde, ncells = D['effect_size'], D['n_de'].astype(float), D['n_cells'].astype(float)
    deg = D['deg'][D['node_of']].astype(float)
    Cov = np.column_stack([np.ones(P), eff, nde, ncells, deg])
    beta, *_ = np.linalg.lstsq(Cov, EN_mean, rcond=None)
    pred = Cov @ beta; ss_res = ((EN_mean-pred)**2).sum(); ss_tot = ((EN_mean-EN_mean.mean())**2).sum()
    R2_covariates = float(1 - ss_res/ss_tot) if ss_tot>0 else float('nan')
    residual = EN_mean - pred                    # E[N] with covariates removed

    # partial correlations of E[N] vs each covariate, controlling the others
    part = dict(
        eff = partial_spearman(EN_mean, eff, [nde, ncells, deg]),
        nde = partial_spearman(EN_mean, nde, [eff, ncells, deg]),
        ncells = partial_spearman(EN_mean, ncells, [eff, nde, deg]),
        deg = partial_spearman(EN_mean, deg, [eff, nde, ncells]),
    )

    # ---- (3) is the RESIDUAL reproducible across seeds? (non-redundant AND reproducible) ----
    resid_seeds = []
    for s in range(args.seeds):
        b, *_ = np.linalg.lstsq(Cov, EN_seeds[s], rcond=None)
        resid_seeds.append(EN_seeds[s] - Cov @ b)
    resid_seeds = np.array(resid_seeds)
    rr = []
    for i in range(args.seeds):
        for j in range(i+1, args.seeds):
            if resid_seeds[i].std()>0 and resid_seeds[j].std()>0:
                rr.append(spearmanr(resid_seeds[i], resid_seeds[j])[0])
    resid_repro = float(np.mean(rr)) if rr else float('nan')

    res = dict(
        n_perturbations=P, n_seeds=args.seeds, n_hops=args.n_hops,
        EN_mean_of_means=float(EN_mean.mean()), EN_std_across_perts=float(EN_mean.std()),
        EN_range=float(EN_mean.max()-EN_mean.min()),
        EN_frac_of_budget=float((EN_mean.max()-EN_mean.min())/args.n_hops),
        reproducibility_spearman=repro,
        R2_explained_by_covariates=R2_covariates,
        residual_std=float(residual.std()),
        partial_spearman=part,
        residual_reproducibility_spearman=resid_repro,
        spearman_EN_effect=float(spearmanr(EN_mean, eff)[0]) if EN_mean.std()>0 else 0.0,
    )
    # save signature
    import csv
    with open(os.path.join(args.out,'signature.csv'),'w',newline='') as f:
        w = csv.writer(f); w.writerow(['perturbation','E_hops_mean','residual','effect_size','n_de','n_cells','graph_degree'])
        for k,g in enumerate(D['perts']):
            w.writerow([g, EN_mean[k], residual[k], eff[k], int(nde[k]), int(ncells[k]), int(deg[k])])
    np.save(os.path.join(args.out,'EN_seeds.npy'), EN_seeds)
    json.dump(res, open(os.path.join(args.out,'protocol_results.json'),'w'), indent=2)
    print('[protocol]', json.dumps(res), flush=True)
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
