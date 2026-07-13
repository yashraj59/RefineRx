"""train_arc_act.py — train the GENUINE learned ACT halt head (convergence features, small-lambda init,
KL=0) on ARC's frozen ST-SE-Replogle-k562. Tests whether LEARNED E[N] (a) uses the depth range,
(b) recovers the reproducible argmin depth, (c) is reproducible + effect-independent."""
import sys, os, json, argparse, glob
import numpy as np, torch
from scipy.stats import spearmanr, rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/halt/state_repo/src")
from load_arc_state import load_arc
from bioact_act import BioActACT
from arc_replogle_data import build
DEV="cuda"

def partial_spearman(y,x,ctrls):
    def resid(v):
        Y=rankdata(v); C=np.column_stack([np.ones(len(Y))]+[rankdata(c) for c in ctrls])
        b,*_=np.linalg.lstsq(C,Y,rcond=None); return Y-C@b
    ry,rx=resid(y),resid(x)
    return 0.0 if ry.std()<1e-9 or rx.std()<1e-9 else float(spearmanr(ry,rx)[0])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True); ap.add_argument("--epochs",type=int,default=40)
    ap.add_argument("--seeds",type=int,default=4); ap.add_argument("--S",type=int,default=64)
    ap.add_argument("--lr",type=float,default=3e-3); ap.add_argument("--entropy_beta",type=float,default=0.0)
    ap.add_argument("--smoke",action="store_true")
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    if a.smoke: a.epochs=8; a.seeds=2
    print(f"[setup] device={DEV} epochs={a.epochs} seeds={a.seeds} entropy_beta={a.entropy_beta}",flush=True)
    arc,vd,hp,mm=load_arc("/workspace/halt/hf_cache_se",device=DEV)
    D=build(cache_glob="/workspace/halt/hf_cache_se",ckpt_glob="/workspace/halt/hf_cache_se",min_cells=20,cell_set_len=a.S)
    perts=list(D["per_pert"].keys()); S=a.S
    print(f"[data] perts={len(perts)}",flush=True)
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
    # argmin-depth reference (the rho=0.836 signal) from many draws
    m_ref=BioActACT(arc).to(DEV); argmin_d=np.zeros((6,len(perts)))
    for d in range(6):
        rng=np.random.default_rng(500+d); off=0
        for i in range(0,len(perts),32):
            bp=perts[i:i+32]; pe,ba,bi,td,bg=batch(bp,rng)
            _,cd=m_ref.eval_EN(pe,ba,bi,td,bg); argmin_d[d,off:off+len(bp)]=cd.argmin(1)+1; off+=len(bp)
    argmin_depth=argmin_d.mean(0)
    ENs=[]; first_lams=None
    for s in range(a.seeds):
        torch.manual_seed(s); np.random.seed(s); rng=np.random.default_rng(s)
        model=BioActACT(arc, entropy_beta=a.entropy_beta).to(DEV)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
        for ep in range(a.epochs):
            model.train(); order=rng.permutation(len(perts))
            for i in range(0,len(perts),32):
                bp=[perts[j] for j in order[i:i+32]]
                pe,ba,bi,td,bg=batch(bp,rng); out=model(pe,ba,bi,td,bg)
                opt.zero_grad(); out["total"].backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0); opt.step()
        model.eval(); EN=np.zeros(len(perts)); off=0; rng2=np.random.default_rng(9000+s)
        for i in range(0,len(perts),32):
            bp=perts[i:i+32]; pe,ba,bi,td,bg=batch(bp,rng2)
            en,_=model.eval_EN(pe,ba,bi,td,bg); EN[off:off+len(bp)]=en; off+=len(bp)
        ENs.append(EN)
        print(f"[seed {s}] E[N] mean={EN.mean():.3f} std={EN.std():.3f} range=[{EN.min():.2f},{EN.max():.2f}] "
              f"rho(EN,argmin)={spearmanr(EN,argmin_depth)[0]:.3f}",flush=True)
    ENs=np.array(ENs); Em=ENs.mean(0)
    eff=np.array([D["per_pert"][g]["eff"] for g in perts]); nde=np.array([D["per_pert"][g]["n_de"] for g in perts]).astype(float)
    ncells=np.array([D["per_pert"][g]["n"] for g in perts]).astype(float)
    pair=[spearmanr(ENs[i],ENs[j])[0] for i in range(a.seeds) for j in range(i+1,a.seeds) if ENs[i].std()>0 and ENs[j].std()>0]
    repro=float(np.mean(pair)) if pair else float("nan")
    P=len(perts); Cov=np.column_stack([np.ones(P),eff,nde,ncells])
    b,*_=np.linalg.lstsq(Cov,Em,rcond=None); R2=float(1-((Em-Cov@b)**2).sum()/(((Em-Em.mean())**2).sum()+1e-12))
    res=dict(model="arc_st_se_replogle_k562_LEARNED_ACT", n_perturbations=P, n_layers=8, n_seeds=a.seeds,
             EN_mean=float(Em.mean()), EN_std=float(Em.std()), EN_cv=float(Em.std()/max(Em.mean(),1e-8)),
             EN_range=[float(Em.min()),float(Em.max())], EN_frac_of_budget=float((Em.max()-Em.min())/8),
             reproducibility_spearman=repro, R2_explained_by_covariates=R2,
             rho_EN_argmin_depth=float(spearmanr(Em,argmin_depth)[0]) if Em.std()>0 else 0.0,
             spearman_EN_effect=float(spearmanr(Em,eff)[0]) if Em.std()>0 else 0.0,
             partial_eff=partial_spearman(Em,eff,[nde,ncells]) if Em.std()>0 else 0.0)
    import csv
    with open(os.path.join(a.out,"act_signature.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["perturbation","EN_learned","argmin_depth","effect_size"])
        for k,g in enumerate(perts): w.writerow([g,round(Em[k],3),round(argmin_depth[k],3),round(eff[k],4)])
    np.save(os.path.join(a.out,"EN_seeds.npy"),ENs); np.save(os.path.join(a.out,"argmin_depth.npy"),argmin_depth)
    json.dump(res,open(os.path.join(a.out,"act_protocol_results.json"),"w"),indent=2)
    print("[RESULT]",json.dumps(res),flush=True); print("[done]",flush=True)

if __name__=="__main__": main()
