"""
extract_signature.py — load the trained ScDiffPonder checkpoint and compute a PROPER
per-perturbation signature, fixing the response_shift confound to be distance from CONTROL
mean (not the x0 self-recon target, which is identically the input => zero variance => nan).

Signature per perturbation (mean over its cells):
  refinement_rounds     : E[N] from PonderNet halting distribution
  halt_confidence       : -entropy of halting distribution p (sharper = more confident)
  nonlinear_correction  : ||x0_hat[last round] - x0_hat[first round]|| (how much refinement moved it)
  response_shift        : ||mean_expr(pert) - mean_expr(control)||  [effect size, the confound]
Then reports Spearman + Pearson corr(refinement_rounds, response_shift) as the Stage-0
effect-size-confound check the thesis requires.
"""
import sys, os, json, types, argparse
import numpy as np, torch

for m in ['scib','scib.metrics','scib.metrics.ari','scib.metrics.nmi','adjustText','dcor']:
    sys.modules.setdefault(m, types.ModuleType(m))
sys.modules['scib.metrics.ari'].ari = lambda *a, **k: 0
sys.modules['scib.metrics.nmi'].nmi = lambda *a, **k: 0
sys.modules['adjustText'].adjust_text = lambda *a, **k: None
sys.modules['dcor'].distance_correlation = lambda *a, **k: 0.0

from torch.utils.data import DataLoader
from adamson_data import AdamsonPert, collate
from scdiff_ponder_model import ScDiffPonder
from train_scdiff_ponder import build_model_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5ad', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n_hvg', type=int, default=2000)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--embed_dim', type=int, default=256)
    ap.add_argument('--n_refine', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--out', default='.')
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = AdamsonPert(args.h5ad, n_hvg=args.n_hvg)
    print(f'[data] cells={len(ds)} genes={ds.G} conds={ds.n_cond}', flush=True)

    # control mean expression (effect-size reference)
    ctrl_code = ds.cls_to_code.get('control')
    X = ds.input.numpy()
    codes_all = ds.pert.numpy()
    ctrl_mean = X[codes_all == ctrl_code].mean(0) if ctrl_code is not None else X.mean(0)

    mc = build_model_config(ds.G, ds.n_cond, ds.gene_names, depth=args.depth, embed_dim=args.embed_dim)
    model = ScDiffPonder(model_config=mc, timesteps=1000, beta_schedule='linear', loss_type='l2',
        loss_strategy='recon_full', parameterization='x0', cond_key='cond', input_key='input',
        pert_target_key='pert_target', pert_flag=True, classify_flag=False, recon_flag=True,
        recon_sample=False, mask_context=False, mask_noised_context=False, use_ema=False,
        cond_names=['pert'], n_refine=args.n_refine).to(dev)
    sd = torch.load(args.ckpt, map_location=dev)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'[ckpt] loaded (missing={len(missing)} unexpected={len(unexpected)})', flush=True)
    model.eval()

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=2)
    rounds, halt, nonlin, codes = [], [], [], []
    with torch.no_grad():
        for batch in dl:
            x = batch['input'].to(dev)
            cond = {'pert': batch['cond']['pert'].to(dev)}
            # run the refinement loop, capturing per-round x0_hat for nonlinear-correction
            B = x.shape[0]
            full_ctx = torch.zeros_like(x, dtype=torch.bool)
            x_cur = x; x0_first = None; x0_last = None
            lambdas = []
            for k in range(model.n_refine):
                t = torch.full((B,), model.refine_t_schedule[k], device=dev, dtype=torch.long)
                x_ctxt, x_noised, mask = model.prepare_noised_input(x_cur, t, mask=full_ctx)
                pred, mask = model.model(x_ctxt, x_noised, t=t, conditions=cond,
                                         input_gene_list=None, aug_graph=None, mask=mask)
                if k == 0: x0_first = pred
                x0_last = pred
                lambdas.append(model.halt_head(pred))
                x_cur = pred
            from halting import ponder_step_probs
            lam = torch.stack(lambdas, dim=1)                    # (B,N)
            p = ponder_step_probs(lam)                           # (B,N)
            steps = torch.arange(1, model.n_refine + 1, device=dev).float()
            E_N = (p * steps).sum(1)                             # (B,)
            ent = -(p.clamp_min(1e-9).log() * p).sum(1)          # (B,)
            nl = (x0_last - x0_first).norm(dim=1)                # (B,)
            rounds.append(E_N.cpu().numpy()); halt.append((-ent).cpu().numpy())
            nonlin.append(nl.cpu().numpy()); codes.append(batch['cond']['pert'].cpu().numpy())
    rounds = np.concatenate(rounds); halt = np.concatenate(halt)
    nonlin = np.concatenate(nonlin); codes = np.concatenate(codes)

    # per-perturbation aggregation + control-relative response shift
    import csv
    from scipy.stats import spearmanr, pearsonr
    inv = {v: k for k, v in ds.cls_to_code.items()}
    rows = []
    for cc in np.unique(codes):
        m = codes == cc
        pert_mean = X[codes_all == cc].mean(0)
        shift = float(np.linalg.norm(pert_mean - ctrl_mean))
        rows.append(dict(perturbation=inv[int(cc)], n_cells=int(m.sum()),
                         refinement_rounds=float(rounds[m].mean()),
                         halt_confidence=float(halt[m].mean()),
                         nonlinear_correction=float(nonlin[m].mean()),
                         response_shift=shift))
    with open(os.path.join(args.out, 'signature_fixed.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # Stage-0 confound check at the PERTURBATION level (exclude control)
    sub = [r for r in rows if r['perturbation'] != 'control']
    EN = np.array([r['refinement_rounds'] for r in sub])
    RS = np.array([r['response_shift'] for r in sub])
    NL = np.array([r['nonlinear_correction'] for r in sub])
    res = {}
    if len(EN) > 3 and EN.std() > 0 and RS.std() > 0:
        sr, sp = spearmanr(EN, RS); pr, pp = pearsonr(EN, RS)
        res['spearman_EN_shift'] = float(sr); res['spearman_p'] = float(sp)
        res['pearson_EN_shift'] = float(pr); res['pearson_p'] = float(pp)
        # is E[N] more than just effect size? corr of nonlinear-correction with shift too
        if NL.std() > 0:
            res['spearman_nonlin_shift'] = float(spearmanr(NL, RS)[0])
        res['EN_std'] = float(EN.std()); res['EN_range'] = [float(EN.min()), float(EN.max())]
        res['n_perturbations'] = int(len(EN))
    else:
        res['note'] = f'insufficient variance: EN.std={EN.std():.4g} RS.std={RS.std():.4g}'
    json.dump(res, open(os.path.join(args.out, 'confound_check_fixed.json'), 'w'), indent=2)
    print('[confound]', json.dumps(res), flush=True)
    print('[done] wrote signature_fixed.csv, confound_check_fixed.json', flush=True)


if __name__ == '__main__':
    main()
