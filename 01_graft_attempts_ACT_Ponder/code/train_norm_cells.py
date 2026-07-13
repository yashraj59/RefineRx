"""
train_norm_cells.py — same magnitude-free cosine-distance halting target, but trained PER-CELL
with minibatches (stochastic averaging over ~55k perturbed cells) instead of full-batch over 89
perturbation means. This removes optimization instability as an explanation for irreproducibility:
if E[hops] is STILL seed-unstable here, the irreproducibility is a property of the signal, not
of my optimizer.
"""
import sys, os, json, argparse, time
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from norm_data import build
from adaptive_refiner_norm import AdaptiveRefinerNorm
from scipy.stats import spearmanr, rankdata

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_seed(D, edge_index, cell_pert, cell_node, target_dir_t, seed, epochs, bs,
               n_hops, d, lr, pw, lp, pb):
    torch.manual_seed(seed); np.random.seed(seed)
    model = AdaptiveRefinerNorm(len(D['node_genes']), d=d, n_hops=n_hops,
                                ponder_weight=pw, lambda_prior=lp, ponder_beta=pb).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ds = TensorDataset(cell_node, cell_pert)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)
    for ep in range(epochs):
        model.train()
        for node_b, pert_b in dl:
            tgt = target_dir_t[pert_b.to(DEV)]         # (B, n_genes) unit direction
            out = model(node_b.to(DEV), tgt, edge_index)
            opt.zero_grad(); out['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    sig = model.signature(edge_index)
    return sig['E_N'].numpy()[D['node_of']]


def partial_spearman(y, x, controls):
    def resid(v):
        Y = rankdata(v); Cm = np.column_stack([np.ones(len(Y))] + [rankdata(c) for c in controls])
        beta, *_ = np.linalg.lstsq(Cm, Y, rcond=None); return Y - Cm @ beta
    ry, rx = resid(y), resid(x)
    if ry.std()<1e-9 or rx.std()<1e-9: return 0.0
    return float(spearmanr(ry, rx)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5ad', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--n_nodes', type=int, default=2000); ap.add_argument('--knn', type=int, default=16)
    ap.add_argument('--n_hops', type=int, default=8); ap.add_argument('--d', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=12); ap.add_argument('--bs', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--ponder_weight', type=float, default=0.1)
    ap.add_argument('--lambda_prior', type=float, default=0.2); ap.add_argument('--ponder_beta', type=float, default=0.05)
    ap.add_argument('--seeds', type=int, default=6); ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)

    print(f'[setup] device={DEV} torch={torch.__version__}', flush=True)
    D = build(args.h5ad, n_nodes=args.n_nodes, knn=args.knn)
    P = len(D['perts']); pert_to_i = {g:i for i,g in enumerate(D['perts'])}
    edge_index = torch.tensor(D['edge_index'], device=DEV)
    target_dir_t = torch.tensor(D['target_dir'], device=DEV)   # (P, n_genes)

    # build per-cell arrays: each perturbed cell -> (its gene's node, its perturbation index)
    tg = D['tg']; gene_to_node = D['gene_to_node']
    cn, cp = [], []
    for i,g in enumerate(tg):
        j = pert_to_i.get(g)
        if j is not None and g in gene_to_node:
            cn.append(gene_to_node[g]); cp.append(j)
    cell_node = torch.tensor(cn, dtype=torch.long); cell_pert = torch.tensor(cp, dtype=torch.long)
    print(f"[data] nodes={len(D['node_genes'])} perts={P} perturbed_cells={len(cn)}", flush=True)
    if args.smoke:
        args.epochs=2; args.seeds=2

    EN_seeds=[]
    for s in range(args.seeds):
        t0=time.time()
        EN = train_seed(D, edge_index, cell_pert, cell_node, target_dir_t, s, args.epochs,
                        args.bs, args.n_hops, args.d, args.lr, args.ponder_weight,
                        args.lambda_prior, args.ponder_beta)
        EN_seeds.append(EN)
        print(f"[seed {s}] E[N] mean={EN.mean():.3f} std={EN.std():.3f} "
              f"range=[{EN.min():.3f},{EN.max():.3f}] ({time.time()-t0:.1f}s)", flush=True)
    EN_seeds=np.array(EN_seeds); EN_mean=EN_seeds.mean(0)

    pair=[spearmanr(EN_seeds[i],EN_seeds[j])[0] for i in range(args.seeds) for j in range(i+1,args.seeds)
          if EN_seeds[i].std()>0 and EN_seeds[j].std()>0]
    repro=float(np.mean(pair)) if pair else float('nan')

    eff,nde,ncells=D['effect_size'],D['n_de'].astype(float),D['n_cells'].astype(float)
    deg=D['deg'][D['node_of']].astype(float)
    Cov=np.column_stack([np.ones(P),eff,nde,ncells,deg])
    beta,*_=np.linalg.lstsq(Cov,EN_mean,rcond=None); pred=Cov@beta
    R2=float(1-((EN_mean-pred)**2).sum()/((EN_mean-EN_mean.mean())**2).sum())
    residual=EN_mean-pred
    resid_seeds=np.array([EN_seeds[s]-Cov@np.linalg.lstsq(Cov,EN_seeds[s],rcond=None)[0] for s in range(args.seeds)])
    rr=[spearmanr(resid_seeds[i],resid_seeds[j])[0] for i in range(args.seeds) for j in range(i+1,args.seeds)
        if resid_seeds[i].std()>0 and resid_seeds[j].std()>0]
    resid_repro=float(np.mean(rr)) if rr else float('nan')

    res=dict(n_perturbations=P,n_seeds=args.seeds,n_hops=args.n_hops,training="per_cell_minibatch",
             EN_mean_of_means=float(EN_mean.mean()),EN_std_across_perts=float(EN_mean.std()),
             EN_range=float(EN_mean.max()-EN_mean.min()),EN_frac_of_budget=float((EN_mean.max()-EN_mean.min())/args.n_hops),
             reproducibility_spearman=repro,R2_explained_by_covariates=R2,residual_std=float(residual.std()),
             partial_spearman=dict(eff=partial_spearman(EN_mean,eff,[nde,ncells,deg]),
                                   nde=partial_spearman(EN_mean,nde,[eff,ncells,deg]),
                                   ncells=partial_spearman(EN_mean,ncells,[eff,nde,deg]),
                                   deg=partial_spearman(EN_mean,deg,[eff,nde,ncells])),
             residual_reproducibility_spearman=resid_repro,
             spearman_EN_effect=float(spearmanr(EN_mean,eff)[0]) if EN_mean.std()>0 else 0.0)
    import csv
    with open(os.path.join(args.out,'signature.csv'),'w',newline='') as f:
        w=csv.writer(f); w.writerow(['perturbation','E_hops_mean','residual','effect_size','n_de','n_cells','graph_degree'])
        for k,g in enumerate(D['perts']): w.writerow([g,EN_mean[k],residual[k],eff[k],int(nde[k]),int(ncells[k]),int(deg[k])])
    np.save(os.path.join(args.out,'EN_seeds.npy'),EN_seeds)
    json.dump(res,open(os.path.join(args.out,'protocol_results.json'),'w'),indent=2)
    print('[protocol]',json.dumps(res),flush=True); print('[done]',flush=True)


if __name__=='__main__':
    main()
