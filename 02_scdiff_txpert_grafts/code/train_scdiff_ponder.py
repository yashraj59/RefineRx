"""
train_scdiff_ponder.py — end-to-end training of ScDiffPonder on Adamson 2016 UPR Perturb-seq.

Builds the real DiffusionModel config (matching configs/eval_perturbation.yaml, scaled down
for a first run), wraps it in ScDiffPonder (native diffusion loss + PonderNet ponder loss),
trains with a plain torch loop (no Lightning Trainer needed for the smoke/real run), and
writes:
  - metrics.json         : per-epoch base loss, ponder loss, mean E[N], halt entropy
  - signature.csv        : per-perturbation refinement_rounds / halt_confidence / response_shift
  - checkpoint.pt        : model weights
Usage: python train_scdiff_ponder.py --h5ad <path> --epochs N --smoke
"""
import sys, os, json, argparse, types, time
import numpy as np, torch

# --- shim eval-only heavy deps so `import scdiff.model` works for TRAINING (no scib/dcor/adjustText) ---
for m in ['scib','scib.metrics','scib.metrics.ari','scib.metrics.nmi','adjustText','dcor']:
    sys.modules.setdefault(m, types.ModuleType(m))
sys.modules['scib.metrics.ari'].ari = lambda *a, **k: 0
sys.modules['scib.metrics.nmi'].nmi = lambda *a, **k: 0
sys.modules['adjustText'].adjust_text = lambda *a, **k: None
sys.modules['dcor'].distance_correlation = lambda *a, **k: 0.0

from torch.utils.data import DataLoader, random_split
from adamson_data import AdamsonPert, collate
from scdiff_ponder_model import ScDiffPonder


def build_model_config(G, n_cond, gene_names, depth=4, embed_dim=256):
    """DiffusionModel params matching eval_perturbation.yaml (scaled), for x0 pert task."""
    from omegaconf import OmegaConf
    params = dict(
        pretrained_gene_list=gene_names,     # vocabulary == our HVG list
        activation='gelu', norm_layer='layernorm', depth=depth, dropout=0.0,
        cell_mask_ratio=0.25, mask_mode='v2',
        embed_dim=embed_dim, dim_head=64, num_heads=8,
        decoder_embed_dim=embed_dim, decoder_dim_head=64, decoder_num_heads=8,
        mlp_time_embed=False,
        cond_type='crossattn', cond_emb_type='embedding', cond_cat_input=False,
        cond_tokens=1, cond_mask_ratio=0.1,
        cond_num_dict={'pert': n_cond},      # conditioner vocab: 92 target_gene classes
        post_cond_layers=1, post_cond_norm='batchnorm', post_cond_mask_ratio=0.1,
        mask_dec_cond=False, mask_dec_cond_ratio=True,
        decoder_embed_type='embedder', decoder_mask='inv_enc',
        encoder_type='mlp',
    )
    return {'target': 'scdiff.model.DiffusionModel', 'params': params}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5ad', required=True)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--n_hvg', type=int, default=2000)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--embed_dim', type=int, default=256)
    ap.add_argument('--n_refine', type=int, default=5)
    ap.add_argument('--ponder_weight', type=float, default=0.1)
    ap.add_argument('--lambda_prior', type=float, default=0.2)
    ap.add_argument('--ponder_beta', type=float, default=0.05)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--smoke', action='store_true', help='tiny run: 1 epoch, 2 batches, cpu ok')
    ap.add_argument('--out', default='.')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[setup] device={dev} torch={torch.__version__}', flush=True)

    ds = AdamsonPert(args.h5ad, n_hvg=args.n_hvg)
    print(f'[data] cells={len(ds)} genes={ds.G} conditions={ds.n_cond}', flush=True)
    n_val = max(1, int(0.1 * len(ds)))
    tr, va = random_split(ds, [len(ds) - n_val, n_val],
                          generator=torch.Generator().manual_seed(0))
    dl_tr = DataLoader(tr, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                       num_workers=2, drop_last=True)
    dl_va = DataLoader(va, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                       num_workers=2)

    model_config = build_model_config(ds.G, ds.n_cond, ds.gene_names,
                                      depth=args.depth, embed_dim=args.embed_dim)
    model = ScDiffPonder(
        model_config=model_config,
        timesteps=1000, beta_schedule='linear', loss_type='l2',
        loss_strategy='recon_full', parameterization='x0',
        cond_key='cond', input_key='input', pert_target_key='pert_target',
        pert_flag=True, classify_flag=False, recon_flag=True, recon_sample=False,
        mask_context=False, mask_noised_context=False, use_ema=False,
        cond_names=['pert'], monitor='val/loss',
        # ponder knobs
        ponder_weight=args.ponder_weight, n_refine=args.n_refine,
        lambda_prior=args.lambda_prior, ponder_beta=args.ponder_beta,
    ).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[model] params={n_params/1e6:.2f}M  refine_t={model.refine_t_schedule}', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    hist = []
    epochs = 1 if args.smoke else args.epochs
    for ep in range(epochs):
        model.train(); t0 = time.time()
        agg = {}
        for bi, batch in enumerate(dl_tr):
            batch['input'] = batch['input'].to(dev)
            batch['pert_target'] = batch['pert_target'].to(dev)
            batch['cond']['pert'] = batch['cond']['pert'].to(dev)
            loss, ld = model.shared_step(batch)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in ld.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            if args.smoke and bi >= 1:
                break
        n = bi + 1
        row = {k: v / n for k, v in agg.items()}
        row['epoch'] = ep; row['sec'] = round(time.time() - t0, 1)
        hist.append(row)
        print(f"[ep {ep}] " + " ".join(f"{k.split('/')[-1]}={row[k]:.4f}"
              for k in row if 'loss' in k or 'E_N' in k) + f"  ({row['sec']}s)", flush=True)

    os.makedirs(args.out, exist_ok=True)
    json.dump(hist, open(os.path.join(args.out, 'metrics.json'), 'w'), indent=2)

    # ---- per-perturbation signature over the (val) set ----
    model.eval()
    rounds, halt, shift, codes = [], [], [], []
    with torch.no_grad():
        for batch in dl_va:
            x = batch['input'].to(dev); tgt = batch['pert_target'].to(dev)
            cond = {'pert': batch['cond']['pert'].to(dev)}
            sig = model.signature(x, cond, tgt)
            rounds.append(sig['refinement_rounds'].cpu().numpy())
            halt.append(sig['halt_confidence'].cpu().numpy())
            shift.append((tgt - x).norm(dim=1).cpu().numpy())   # response shift proxy
            codes.append(batch['cond']['pert'].cpu().numpy())
            if args.smoke:
                break
    rounds = np.concatenate(rounds); halt = np.concatenate(halt)
    shift = np.concatenate(shift); codes = np.concatenate(codes)
    # aggregate per perturbation code
    import csv
    inv = {v: k for k, v in ds.cls_to_code.items()}
    with open(os.path.join(args.out, 'signature.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['perturbation', 'n_cells', 'refinement_rounds',
                                       'halt_confidence', 'response_shift'])
        for cc in np.unique(codes):
            m = codes == cc
            w.writerow([inv[int(cc)], int(m.sum()), float(rounds[m].mean()),
                        float(halt[m].mean()), float(shift[m].mean())])
    # effect-size confound check: corr(E[N], response_shift)
    if len(rounds) > 2:
        cr = float(np.corrcoef(rounds, shift)[0, 1])
        print(f"[signature] corr(refinement_rounds, response_shift) = {cr:+.3f}", flush=True)
        json.dump({'corr_EN_shift': cr, 'n_val_cells': int(len(rounds))},
                  open(os.path.join(args.out, 'confound_check.json'), 'w'), indent=2)

    torch.save(model.state_dict(), os.path.join(args.out, 'checkpoint.pt'))
    print('[done] wrote metrics.json, signature.csv, confound_check.json, checkpoint.pt', flush=True)


if __name__ == '__main__':
    main()
