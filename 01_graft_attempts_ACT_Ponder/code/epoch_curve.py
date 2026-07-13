"""epoch_curve.py — snapshot per-pert depth spread at increasing epoch counts (single model,
checkpointed) to test whether convergence-depth variability is a training-time transient."""
import sys, os, json, argparse, numpy as np, torch
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from norm_data_lean import build
from act_refiner_conv import ConvRefiner
DEV="cuda" if torch.cuda.is_available() else "cpu"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--h5ad",required=True); ap.add_argument("--out",required=True)
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    D=build(a.h5ad,n_genes=2000,min_cells=20); P=len(D["perts"])
    print(f"[data] perts={P}",flush=True)
    tgt=torch.tensor(D["target_dir"],device=DEV); idx=torch.arange(P,device=DEV)
    eff=D["effect_size"]
    checkpoints=[5,10,20,40,80,160,320,640]
    rows=[]
    torch.manual_seed(0); np.random.seed(0)
    model=ConvRefiner(D["n_genes"],P,d=128,max_rounds=8,eps_converge=0.02,step=0.5).to(DEV)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-3)
    ep=0
    for ck in checkpoints:
        while ep<ck:
            model.train(); opt.zero_grad(); out=model(idx,tgt); out["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); ep+=1
        model.eval()
        with torch.no_grad(): o=model(idx,tgt); dep=o["depth"].cpu().numpy(); recon=float(o["recon"].item())
        cv=float(dep.std()/max(dep.mean(),1e-8))
        rho=float(spearmanr(dep,eff)[0]) if dep.std()>0 else 0.0
        rows.append(dict(epoch=ck,depth_mean=float(dep.mean()),depth_cv=cv,depth_range=float(dep.max()-dep.min()),
                         recon=recon,rho_depth_eff=rho))
        print(f"[ep {ck}] depth_mean={dep.mean():.3f} cv={cv:.4f} range={dep.max()-dep.min():.0f} recon={recon:.4f} rho(dep,eff)={rho:+.3f}",flush=True)
    json.dump(rows,open(os.path.join(a.out,"epoch_curve.json"),"w"),indent=2)
    print("[done]",flush=True)

if __name__=="__main__": main()
