"""
train_adaptive_txpert.py — train the adaptive-depth TxPert-style model on Adamson, then
extract the per-perturbation E[hops] signature and run the DOUBLE confound check:
  corr(E[hops], effect_size)  AND  corr(E[hops], graph_degree).
The thesis question: does a message-passing depth axis decouple E[N] from raw effect size
(where scDiff's re-noising axis could not)?
"""
import sys, os, json, argparse, time, types
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset

from txpert_data import build
from adaptive_txpert import AdaptiveTxPert
from scipy.stats import spearmanr, pearsonr

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5ad', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_nodes', type=int, default=2000)
    ap.add_argument('--knn', type=int, default=16)
    ap.add_argument('--n_hops', type=int, default=8)
    ap.add_argument('--d', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch_size', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--ponder_weight', type=float, default=0.1)
    ap.add_argument('--lambda_prior', type=float, default=0.2)
    ap.add_argument('--ponder_beta', type=float, default=0.05)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)

    print(f'[setup] device={DEV} torch={torch.__version__}', flush=True)
    D = build(args.h5ad, n_nodes=args.n_nodes, knn=args.knn)
    n_nodes = len(D['node_genes'])
    print(f"[data] nodes={n_nodes} edges={D['edge_index'].shape[1]} "
          f"perturbed_cells={len(D['keep_idx'])} perts={len(D['measured_pert'])}", flush=True)

    edge_index = torch.tensor(D['edge_index'], device=DEV)
    Xnode = torch.tensor(D['Xnode'], dtype=torch.float32)
    pert_node = torch.tensor(D['pert_node'], dtype=torch.long)
    keep = D['keep_idx']
    if args.smoke:
        keep = keep[:2000]; args.epochs = 2
    ds = TensorDataset(pert_node[keep], torch.tensor(keep))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = AdaptiveTxPert(n_nodes, d=args.d, n_hops=args.n_hops,
                           lambda_prior=args.lambda_prior, ponder_beta=args.ponder_beta,
                           ponder_weight=args.ponder_weight).to(DEV)
    # init basal to control mean
    with torch.no_grad():
        model.basal.copy_(torch.tensor(D['ctrl_mean'], device=DEV))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"[model] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M n_hops={args.n_hops}", flush=True)

    metrics = []
    for ep in range(args.epochs):
        model.train(); agg = {}; nb = 0; t0 = time.time()
        for pn, ci in dl:
            pn = pn.to(DEV); tgt = Xnode[ci].to(DEV)
            out = model(pn, tgt, edge_index)
            opt.zero_grad(); out['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            for k in ('recon', 'ponder', 'total'):
                agg[k] = agg.get(k, 0.0) + float(out[k])
            agg['E_N'] = agg.get('E_N', 0.0) + float(out['E_N_batch'].mean()); nb += 1
        row = {k: v / nb for k, v in agg.items()}; row['epoch'] = ep; row['sec'] = round(time.time()-t0,1)
        metrics.append(row)
        print(f"[ep {ep}] recon={row['recon']:.4f} ponder={row['ponder']:.4f} "
              f"E[N]={row['E_N']:.3f} ({row['sec']}s)", flush=True)
    json.dump(metrics, open(os.path.join(args.out,'metrics.json'),'w'), indent=2)

    # ---- per-perturbation signature (per-NODE E[hops]) ----
    model.eval()
    sig = model.signature(edge_index)
    E_N = sig['E_N'].numpy(); halt_conf = sig['halt_conf'].numpy()
    gene_to_node = D['gene_to_node']; ctrl_mean = D['ctrl_mean']; deg = D['deg']
    tg = D['tg']; Xnode = D['Xnode']

    rows = []
    for g in D['measured_pert']:
        node = gene_to_node[g]
        cells = tg == g
        shift = float(np.linalg.norm(Xnode[cells].mean(0) - ctrl_mean))
        rows.append(dict(perturbation=g, node=int(node), n_cells=int(cells.sum()),
                         E_hops=float(E_N[node]), halt_confidence=float(halt_conf[node]),
                         graph_degree=int(deg[node]), response_shift=shift))
    import csv
    with open(os.path.join(args.out,'signature.csv'),'w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    EN = np.array([r['E_hops'] for r in rows]); RS = np.array([r['response_shift'] for r in rows])
    DEGr = np.array([r['graph_degree'] for r in rows], dtype=float)
    res = dict(n_perturbations=len(rows),
               EN_mean=float(EN.mean()), EN_std=float(EN.std()),
               EN_min=float(EN.min()), EN_max=float(EN.max()),
               EN_range=float(EN.max()-EN.min()), n_hops=args.n_hops,
               EN_frac_of_budget=float((EN.max()-EN.min())/args.n_hops))
    if EN.std() > 0:
        if RS.std() > 0:
            res['spearman_EN_shift'] = float(spearmanr(EN, RS)[0])
            res['pearson_EN_shift'] = float(pearsonr(EN, RS)[0])
        if DEGr.std() > 0:
            res['spearman_EN_degree'] = float(spearmanr(EN, DEGr)[0])
        # partial: does E[N] track shift AFTER regressing out degree? (residual corr)
        if RS.std() > 0 and DEGr.std() > 0:
            from numpy.polynomial import polynomial as P
            def resid(y, x):
                b = np.polyfit(x, y, 1); return y - np.polyval(b, x)
            res['spearman_EN_shift_given_degree'] = float(
                spearmanr(resid(EN, DEGr), resid(RS, DEGr))[0])
    json.dump(res, open(os.path.join(args.out,'confound_check.json'),'w'), indent=2)
    print('[confound]', json.dumps(res), flush=True)
    print('[done] wrote metrics.json, signature.csv, confound_check.json', flush=True)


if __name__ == '__main__':
    main()
