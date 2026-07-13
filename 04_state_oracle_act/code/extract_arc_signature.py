"""extract_arc_signature.py — the DEFINITIVE per-perturbation depth signature on ARC's pretrained
ST-SE-Replogle-k562 foundation model. Depth = argmin over 8 transformer layers of the magnitude-free
per-layer response-direction loss (which layer best predicts each perturbation's response direction).
Full thesis protocol: reproducibility across independent basal draws + non-redundancy after regressing out
effect_size / n_de / n_cells. All ARC weights frozen; NO learned halt head needed — depth is read directly
from the pretrained model's own layerwise convergence."""
import sys, os, glob, json
import numpy as np, torch
from scipy.stats import spearmanr, rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/halt/state_repo/src")
from load_arc_state import load_arc
from bioact_state import BioActState
from arc_replogle_data import build
DEV="cuda"

def partial_spearman(y,x,ctrls):
    def resid(v):
        Y=rankdata(v); C=np.column_stack([np.ones(len(Y))]+[rankdata(c) for c in ctrls])
        b,*_=np.linalg.lstsq(C,Y,rcond=None); return Y-C@b
    ry,rx=resid(y),resid(x)
    return 0.0 if ry.std()<1e-9 or rx.std()<1e-9 else float(spearmanr(ry,rx)[0])

def main():
    out="/workspace/halt/actrun_state/ar_signature"; os.makedirs(out,exist_ok=True)
    arc,vd,hp,mm=load_arc("/workspace/halt/hf_cache_se",device=DEV)
    D=build(cache_glob="/workspace/halt/hf_cache_se",ckpt_glob="/workspace/halt/hf_cache_se",min_cells=20,cell_set_len=64)
    perts=list(D["per_pert"].keys()); m=BioActState(arc,None).to(DEV); S=64
    def batch(bp,rng):
        ctrl=D["Xs_ctrl"]; bc=D["batch_ctrl"]; B=len(bp)
        sel=[rng.choice(len(ctrl),S,replace=len(ctrl)<S) for _ in bp]
        basal=np.stack([ctrl[s] for s in sel]); bidx=np.stack([bc[s] for s in sel])
        pert=np.zeros((B,S,D["pert_dim"]),dtype=np.float32)
        for i,g in enumerate(bp): pert[i,:,D["per_pert"][g]["onehot_idx"]]=1.0
        resp=np.stack([D["per_pert"][g]["resp"] for g in bp]); td=resp/(np.linalg.norm(resp,axis=1,keepdims=True)+1e-8)
        td=np.repeat(td[:,None,:],S,axis=1); bg=np.broadcast_to(D["ctrl_mean"],(B,S,D["n_genes"]))
        return (torch.tensor(pert,device=DEV),torch.tensor(basal,device=DEV),torch.tensor(bidx,dtype=torch.long,device=DEV),
                torch.tensor(td,dtype=torch.float32,device=DEV),torch.tensor(np.ascontiguousarray(bg),dtype=torch.float32,device=DEV))
    # per-layer loss curves for many independent draws -> depth = argmin; also mean curve per pert
    n_draws=8; depths=np.zeros((n_draws,len(perts))); curves=np.zeros((len(perts),8))
    for d in range(n_draws):
        rng=np.random.default_rng(100+d); off=0
        for i in range(0,len(perts),32):
            bp=perts[i:i+32]; pe,ba,bi,td,bg=batch(bp,rng)
            cd=m.per_layer_direction_loss(pe,ba,bi,td,bg).numpy()
            depths[d,off:off+len(bp)]=cd.argmin(1)+1
            if d==0: curves[off:off+len(bp)]=cd
            off+=len(bp)
    depth_mean=depths.mean(0)
    # reproducibility across the 8 draws (mean pairwise spearman)
    pair=[spearmanr(depths[i],depths[j])[0] for i in range(n_draws) for j in range(i+1,n_draws)]
    repro=float(np.mean(pair))
    eff=np.array([D["per_pert"][g]["eff"] for g in perts]); nde=np.array([D["per_pert"][g]["n_de"] for g in perts]).astype(float)
    ncells=np.array([D["per_pert"][g]["n"] for g in perts]).astype(float)
    P=len(perts); Cov=np.column_stack([np.ones(P),eff,nde,ncells])
    b,*_=np.linalg.lstsq(Cov,depth_mean,rcond=None); resid=depth_mean-Cov@b
    R2=float(1-((depth_mean-Cov@b)**2).sum()/(((depth_mean-depth_mean.mean())**2).sum()+1e-12))
    # residual reproducibility: does the effect-independent part still replicate? split draws in half
    dh1=depths[:4].mean(0); dh2=depths[4:].mean(0)
    def resid_of(v):
        bb,*_=np.linalg.lstsq(Cov,v,rcond=None); return v-Cov@bb
    resid_repro=float(spearmanr(resid_of(dh1),resid_of(dh2))[0])
    res=dict(model="arc_st_se_replogle_k562_layer_argmin_depth", n_perturbations=P, n_layers=8, n_draws=n_draws,
             depth_mean=float(depth_mean.mean()), depth_std=float(depth_mean.std()),
             depth_cv=float(depth_mean.std()/depth_mean.mean()), depth_range=[float(depth_mean.min()),float(depth_mean.max())],
             reproducibility_spearman=repro, R2_explained_by_covariates=R2, residual_std=float(resid.std()),
             residual_reproducibility_spearman=resid_repro,
             partial_spearman=dict(eff=partial_spearman(depth_mean,eff,[nde,ncells]),
                                   nde=partial_spearman(depth_mean,nde,[eff,ncells]),
                                   ncells=partial_spearman(depth_mean,ncells,[eff,nde])),
             spearman_depth_effect=float(spearmanr(depth_mean,eff)[0]),
             per_layer_mean_curve=[float(x) for x in curves.mean(0)],
             depth_hist={int(k):int(v) for k,v in zip(*np.unique(np.round(depth_mean).astype(int),return_counts=True))})
    import csv
    with open(os.path.join(out,"arc_signature.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["perturbation","depth_mean","effect_size","n_de","n_cells"])
        for k,g in enumerate(perts): w.writerow([g,round(depth_mean[k],3),round(eff[k],4),int(nde[k]),int(ncells[k])])
    np.save(os.path.join(out,"depths_draws.npy"),depths); np.save(os.path.join(out,"curves.npy"),curves)
    json.dump(res,open(os.path.join(out,"arc_protocol_results.json"),"w"),indent=2)
    print("[RESULT]",json.dumps(res)); print("[done]")

if __name__=="__main__": main()
