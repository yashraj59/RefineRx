"""train_act.py — thesis protocol on MY OWN ACT-4 refiner (act_refiner.ACTRefiner).
Trains S seeds; reports (1) reproducibility = mean pairwise Spearman of per-pert E[N] ranking across
seeds; (2) non-redundancy = R^2 of covariates on E[N] + partial Spearman controlling the rest;
(3) residual reproducibility. Per-perturbation full-batch (89 directions)."""
import sys, os, json, argparse, time
import numpy as np, torch
from scipy.stats import spearmanr, rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from norm_data_lean import build
from act_refiner import ACTRefiner
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def partial_spearman(y, x, controls):
    def resid(v):
        Y = rankdata(v); C = np.column_stack([np.ones(len(Y))]+[rankdata(c) for c in controls])
        b,*_ = np.linalg.lstsq(C, Y, rcond=None); return Y - C@b
    ry, rx = resid(y), resid(x)
    if ry.std()<1e-9 or rx.std()<1e-9: return 0.0
    return float(spearmanr(ry, rx)[0])

def train_seed(D, seed, epochs, d, max_rounds, ponder_tau, lr):
    torch.manual_seed(seed); np.random.seed(seed)
    P = len(D["perts"])
    model = ACTRefiner(D["n_genes"], P, d=d, max_rounds=max_rounds, ponder_tau=ponder_tau).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    tgt = torch.tensor(D["target_dir"], device=DEV)
    idx = torch.arange(P, device=DEV)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        out = model(idx, tgt); out["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    with torch.no_grad():
        EN = model(idx, tgt)["E_N"].cpu().numpy()
    return EN, float(out["recon"].item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n_genes", type=int, default=2000); ap.add_argument("--min_cells", type=int, default=20)
    ap.add_argument("--d", type=int, default=128); ap.add_argument("--max_rounds", type=int, default=4)
    ap.add_argument("--ponder_tau", type=float, default=0.05); ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-3); ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    print(f"[setup] device={DEV} torch={torch.__version__} max_rounds={a.max_rounds}", flush=True)
    D = build(a.h5ad, n_genes=a.n_genes, min_cells=a.min_cells)
    P = len(D["perts"]); print(f"[data] perts={P} genes={D['n_genes']}", flush=True)
    if a.smoke: a.epochs=40; a.seeds=2
    ENs=[]
    for s in range(a.seeds):
        t0=time.time(); EN,recon = train_seed(D, s, a.epochs, a.d, a.max_rounds, a.ponder_tau, a.lr)
        ENs.append(EN)
        print(f"[seed {s}] E[N] mean={EN.mean():.3f} std={EN.std():.3f} range=[{EN.min():.3f},{EN.max():.3f}] recon={recon:.4f} ({time.time()-t0:.1f}s)", flush=True)
    ENs=np.array(ENs); ENmean=ENs.mean(0)
    pair=[spearmanr(ENs[i],ENs[j])[0] for i in range(a.seeds) for j in range(i+1,a.seeds) if ENs[i].std()>0 and ENs[j].std()>0]
    repro=float(np.mean(pair)) if pair else float("nan")
    eff,nde,nc = D["effect_size"],D["n_de"].astype(float),D["n_cells"].astype(float)
    Cov=np.column_stack([np.ones(P),eff,nde,nc]); b,*_=np.linalg.lstsq(Cov,ENmean,rcond=None); pred=Cov@b
    R2=float(1-((ENmean-pred)**2).sum()/((ENmean-ENmean.mean())**2+1e-12).sum()); resid=ENmean-pred
    rs=np.array([ENs[s]-Cov@np.linalg.lstsq(Cov,ENs[s],rcond=None)[0] for s in range(a.seeds)])
    rr=[spearmanr(rs[i],rs[j])[0] for i in range(a.seeds) for j in range(i+1,a.seeds) if rs[i].std()>0 and rs[j].std()>0]
    res=dict(n_perturbations=P,n_seeds=a.seeds,max_rounds=a.max_rounds,ponder_tau=a.ponder_tau,
             EN_mean_of_means=float(ENmean.mean()),EN_std_across_perts=float(ENmean.std()),
             EN_cv=float(ENmean.std()/max(ENmean.mean(),1e-8)),
             EN_range=float(ENmean.max()-ENmean.min()),EN_frac_of_budget=float((ENmean.max()-ENmean.min())/a.max_rounds),
             reproducibility_spearman=repro,R2_explained_by_covariates=R2,residual_std=float(resid.std()),
             partial_spearman=dict(eff=partial_spearman(ENmean,eff,[nde,nc]),
                                   nde=partial_spearman(ENmean,nde,[eff,nc]),
                                   ncells=partial_spearman(ENmean,nc,[eff,nde])),
             residual_reproducibility_spearman=float(np.mean(rr)) if rr else float("nan"),
             spearman_EN_effect=float(spearmanr(ENmean,eff)[0]) if ENmean.std()>0 else 0.0)
    import csv
    with open(os.path.join(a.out,"signature.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["perturbation","E_N_mean","residual","effect_size","n_de","n_cells"])
        for k,g in enumerate(D["perts"]): w.writerow([g,ENmean[k],resid[k],eff[k],int(nde[k]),int(nc[k])])
    np.save(os.path.join(a.out,"EN_seeds.npy"),ENs); json.dump(res,open(os.path.join(a.out,"protocol_results.json"),"w"),indent=2)
    print("[protocol]",json.dumps(res),flush=True); print("[done]",flush=True)

if __name__=="__main__": main()
