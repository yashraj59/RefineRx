"""
sweep.py — ponder_weight / lambda_prior / ponder_beta / n_refine sweep on Adamson,
in ONE process. Loads data + control mean ONCE, then trains each config fresh,
computes the control-relative per-perturbation signature + effect-size confound,
and appends to sweep_results.json incrementally (partial-safe).
"""
import sys, os, json, types, time, argparse, traceback
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
from halting import ponder_step_probs
from scipy.stats import spearmanr, pearsonr

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

CONFIGS = [
    # --- n_refine=5: prior / beta / weight sweep ---
    {"tag":"r5_base",       "n_refine":5,  "ponder_weight":0.1, "lambda_prior":0.2,  "ponder_beta":0.05},
    {"tag":"r5_lowprior",   "n_refine":5,  "ponder_weight":0.1, "lambda_prior":0.1,  "ponder_beta":0.02},
    {"tag":"r5_highbeta",   "n_refine":5,  "ponder_weight":0.1, "lambda_prior":0.1,  "ponder_beta":0.20},
    {"tag":"r5_highprior",  "n_refine":5,  "ponder_weight":0.1, "lambda_prior":0.3,  "ponder_beta":0.02},
    {"tag":"r5_w0.5",       "n_refine":5,  "ponder_weight":0.5, "lambda_prior":0.2,  "ponder_beta":0.05},
    {"tag":"r5_w1.0",       "n_refine":5,  "ponder_weight":1.0, "lambda_prior":0.2,  "ponder_beta":0.10},
    # --- n_refine=10: deepen the refinement budget ---
    {"tag":"r10_base",      "n_refine":10, "ponder_weight":0.1, "lambda_prior":0.2,  "ponder_beta":0.05},
    {"tag":"r10_lowprior",  "n_refine":10, "ponder_weight":0.1, "lambda_prior":0.1,  "ponder_beta":0.02},
    {"tag":"r10_deepprior", "n_refine":10, "ponder_weight":0.1, "lambda_prior":0.05, "ponder_beta":0.02},
    {"tag":"r10_highbeta",  "n_refine":10, "ponder_weight":0.1, "lambda_prior":0.1,  "ponder_beta":0.20},
    {"tag":"r10_w0.5",      "n_refine":10, "ponder_weight":0.5, "lambda_prior":0.1,  "ponder_beta":0.05},
    {"tag":"r10_w1.0",      "n_refine":10, "ponder_weight":1.0, "lambda_prior":0.1,  "ponder_beta":0.10},
]


def train_one(ds, X, codes_all, ctrl_mean, cfg, epochs, batch_size):
    torch.manual_seed(0); np.random.seed(0)
    mc = build_model_config(ds.G, ds.n_cond, ds.gene_names, depth=4, embed_dim=256)
    model = ScDiffPonder(
        model_config=mc, timesteps=1000, beta_schedule='linear', loss_type='l2',
        loss_strategy='recon_full', parameterization='x0', cond_key='cond', input_key='input',
        pert_target_key='pert_target', pert_flag=True, classify_flag=False, recon_flag=True,
        recon_sample=False, mask_context=False, mask_noised_context=False, use_ema=False,
        cond_names=['pert'],
        n_refine=cfg['n_refine'], ponder_weight=cfg['ponder_weight'],
        lambda_prior=cfg['lambda_prior'], ponder_beta=cfg['ponder_beta']).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate,
                    num_workers=2, drop_last=True)
    model.train(); final = {}
    for ep in range(epochs):
        agg = {}; nb = 0
        for batch in dl:
            batch['input'] = batch['input'].to(DEV); batch['pert_target'] = batch['pert_target'].to(DEV)
            batch['cond']['pert'] = batch['cond']['pert'].to(DEV)
            loss, ld = model.shared_step(batch)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            for k, v in ld.items(): agg[k] = agg.get(k, 0.0) + float(v)
            nb += 1
        final = {k: v / nb for k, v in agg.items()}

    # --- per-perturbation signature (no grad) ---
    model.eval()
    dl2 = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=2)
    rounds, nonlin, halt, codes = [], [], [], []
    with torch.no_grad():
        for batch in dl2:
            x = batch['input'].to(DEV); cond = {'pert': batch['cond']['pert'].to(DEV)}
            B = x.shape[0]; full = torch.zeros_like(x, dtype=torch.bool); x_cur = x
            lambdas = []; x0f = None; x0l = None
            for k in range(model.n_refine):
                t = torch.full((B,), model.refine_t_schedule[k], device=DEV, dtype=torch.long)
                xc, xn, mask = model.prepare_noised_input(x_cur, t, mask=full)
                pred, mask = model.model(xc, xn, t=t, conditions=cond,
                                         input_gene_list=None, aug_graph=None, mask=mask)
                if k == 0: x0f = pred
                x0l = pred; lambdas.append(model.halt_head(pred)); x_cur = pred
            lam = torch.stack(lambdas, dim=1); p = ponder_step_probs(lam)
            steps = torch.arange(1, model.n_refine + 1, device=DEV).float()
            EN = (p * steps).sum(1); ent = -(p.clamp_min(1e-9).log() * p).sum(1)
            nl = (x0l - x0f).norm(dim=1)
            rounds.append(EN.cpu().numpy()); halt.append((-ent).cpu().numpy())
            nonlin.append(nl.cpu().numpy()); codes.append(batch['cond']['pert'].cpu().numpy())
    rounds = np.concatenate(rounds); halt = np.concatenate(halt)
    nonlin = np.concatenate(nonlin); codes = np.concatenate(codes)

    inv = {v: k for k, v in ds.cls_to_code.items()}
    ENp, RS, NL = [], [], []
    for cc in np.unique(codes):
        name = inv[int(cc)]
        if name in ('control', '*'): continue
        m = codes == cc
        ENp.append(float(rounds[m].mean()))
        RS.append(float(np.linalg.norm(X[codes_all == cc].mean(0) - ctrl_mean)))
        NL.append(float(nonlin[m].mean()))
    ENp = np.array(ENp); RS = np.array(RS); NL = np.array(NL)

    res = dict(cfg)
    res.update(
        final_loss_simple=final.get('train/loss_simple'),
        final_loss=final.get('train/loss'),
        final_loss_ponder=final.get('train/loss_ponder'),
        train_E_N=final.get('train/E_N'),
        EN_mean=float(ENp.mean()), EN_std=float(ENp.std()),
        EN_min=float(ENp.min()), EN_max=float(ENp.max()),
        EN_range=float(ENp.max() - ENp.min()),
        EN_frac_of_budget=float((ENp.max() - ENp.min()) / cfg['n_refine']),
        n_pert=int(len(ENp)))
    if ENp.std() > 0 and RS.std() > 0:
        res['spearman_EN_shift'] = float(spearmanr(ENp, RS)[0])
        res['pearson_EN_shift'] = float(pearsonr(ENp, RS)[0])
        res['spearman_nonlin_shift'] = float(spearmanr(NL, RS)[0])
    else:
        res['note'] = f'no variance: EN.std={ENp.std():.3g}'
    del model, opt
    if DEV == 'cuda': torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5ad', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--n_hvg', type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f'[setup] device={DEV} torch={torch.__version__}', flush=True)
    ds = AdamsonPert(args.h5ad, n_hvg=args.n_hvg)
    X = ds.input.numpy(); codes_all = ds.pert.numpy()
    ctrl_code = ds.cls_to_code.get('control')
    ctrl_mean = X[codes_all == ctrl_code].mean(0) if ctrl_code is not None else X.mean(0)
    print(f'[data] cells={len(ds)} genes={ds.G} conds={ds.n_cond} (loaded once)', flush=True)

    results = []
    for i, cfg in enumerate(CONFIGS):
        t0 = time.time()
        try:
            res = train_one(ds, X, codes_all, ctrl_mean, cfg, args.epochs, args.batch_size)
            res['sec'] = round(time.time() - t0, 1)
            results.append(res)
            print(f"[{i+1}/{len(CONFIGS)}] {cfg['tag']}: "
                  f"E[N]={res['EN_mean']:.3f} range={res['EN_range']:.3f} "
                  f"({res['EN_frac_of_budget']*100:.1f}% of {cfg['n_refine']}) "
                  f"rho_shift={res.get('spearman_EN_shift', float('nan')):+.3f} "
                  f"loss={res['final_loss']:.4f}  ({res['sec']}s)", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(CONFIGS)}] {cfg['tag']}: FAILED {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            results.append({**cfg, 'error': f'{type(e).__name__}: {e}'})
        json.dump(results, open(os.path.join(args.out, 'sweep_results.json'), 'w'), indent=2)
    print('[done] wrote sweep_results.json', flush=True)


if __name__ == '__main__':
    main()
