"""diag_roundloss.py — is the per-round task loss flat? If predicting at round 1 is ~as good as
round 8 (field saturates fast), the halt head gets no round-differential signal and collapses to the
KL prior. Dumps per-round cosine distance averaged over perturbations, and per-pert round-of-min."""
import sys, os, json, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from norm_data_lean import build
from bio_act import build_adjacency, BioACT
DEV="cuda" if torch.cuda.is_available() else "cpu"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--h5ad",required=True); ap.add_argument("--graph",required=True)
    ap.add_argument("--out",required=True); ap.add_argument("--mask_seed",action="store_true")
    ap.add_argument("--alpha_fixed",type=float,default=-1); ap.add_argument("--epochs",type=int,default=150)
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    A,node_genes,pert_node_idx=build_adjacency(a.graph,DEV)
    D=build(a.h5ad,n_genes=2000,min_cells=20)
    keep=[i for i,g in enumerate(D["perts"]) if g in pert_node_idx]
    pert_node=[pert_node_idx[D["perts"][i]] for i in keep]
    tgt=torch.tensor(np.asarray(D["target_dir"])[keep],device=DEV); pidx=torch.tensor(pert_node,device=DEV)
    learn_alpha = a.alpha_fixed < 0
    model=BioACT(A,D["n_genes"],max_rounds=8,mask_seed_in_readout=a.mask_seed,learn_alpha=learn_alpha).to(DEV)
    if not learn_alpha: model._alpha_fixed=a.alpha_fixed
    opt=torch.optim.AdamW(model.parameters(),lr=5e-3)
    for ep in range(a.epochs):
        model.train(); opt.zero_grad(); o=model(pidx,tgt); o["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    model.eval()
    # recompute per-round cosine distance WITHOUT halting weighting
    with torch.no_grad():
        B=pidx.size(0); seed=torch.zeros(B,model.N,device=DEV); seed[torch.arange(B),pidx]=1.0
        seed_mask=1.0-seed if a.mask_seed else torch.ones_like(seed)
        h=seed.clone(); al=model.alpha; step_cd=[]
        for t in range(8):
            h=(1-al)*(h@A.t())+al*seed
            pr=torch.nn.functional.normalize(model.dec((h*seed_mask)@model.node_emb),dim=-1,eps=1e-8)
            step_cd.append((1.0-(pr*tgt).sum(-1)).cpu().numpy())
        step_cd=np.array(step_cd).T  # (P, 8)
    per_round_mean=step_cd.mean(0)
    round_of_min=step_cd.argmin(1)+1
    res=dict(alpha=float(model.alpha.item()),mask_seed=a.mask_seed,
             per_round_mean_cd=[float(x) for x in per_round_mean],
             cd_round1_minus_round8=float(per_round_mean[0]-per_round_mean[-1]),
             round_of_min_hist={int(r):int((round_of_min==r).sum()) for r in range(1,9)},
             frac_min_at_round1=float((round_of_min==1).mean()))
    json.dump(res,open(os.path.join(a.out,"diag.json"),"w"),indent=2)
    print("[diag]",json.dumps(res),flush=True)

if __name__=="__main__": main()
