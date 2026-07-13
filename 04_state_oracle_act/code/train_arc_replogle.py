"""train_arc_replogle.py — freeze ARC ST-SE-Replogle-k562 weights, train ONLY the early-exit halt head
on ARC's NATIVE k562 data (where the model predicts well), extract per-perturbation exit-depth E[N],
run reproducibility + non-redundancy + per-layer diagnostic. Magnitude-free ponder loss."""
import sys, os, json, argparse, glob
import numpy as np, torch
from scipy.stats import spearmanr, rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/halt/state_repo/src")
from load_arc_state import load_arc
from bioact_state import BioActState
from arc_replogle_data import build
DEV="cuda" if torch.cuda.is_available() else "cpu"

def partial_spearman(y,x,ctrls):
    def resid(v):
        Y=rankdata(v); C=np.column_stack([np.ones(len(Y))]+[rankdata(c) for c in ctrls])
        b,*_=np.linalg.lstsq(C,Y,rcond=None); return Y-C@b
    ry,rx=resid(y),resid(x)
    return 0.0 if ry.std()<1e-9 or rx.std()<1e-9 else float(spearmanr(ry,rx)[0])

def make_batch(D, perts, S, rng):
    ctrl=D["Xs_ctrl"]; bc=D["batch_ctrl"]; B=len(perts)
    sel=[rng.choice(len(ctrl), S, replace=len(ctrl)<S) for _ in perts]
    basal=np.stack([ctrl[s] for s in sel])                       # (B,S,2058) X_state
    bidx =np.stack([bc[s] for s in sel])                          # (B,S)
    pert=np.zeros((B,S,D["pert_dim"]),dtype=np.float32)
    for i,g in enumerate(perts): pert[i,:,D["per_pert"][g]["onehot_idx"]]=1.0
    resp=np.stack([D["per_pert"][g]["resp"] for g in perts])      # (B,2000)
    tdir=resp/(np.linalg.norm(resp,axis=1,keepdims=True)+1e-8)
    tdir=np.repeat(tdir[:,None,:], S, axis=1)                     # (B,S,2000)
    bgenes=np.broadcast_to(D["ctrl_mean"],(B,S,D["n_genes"]))     # basal gene expr for residual
    return (torch.tensor(pert,device=DEV), torch.tensor(basal,device=DEV),
            torch.tensor(bidx,dtype=torch.long,device=DEV),
            torch.tensor(tdir,dtype=torch.float32,device=DEV),
            torch.tensor(np.ascontiguousarray(bgenes),dtype=torch.float32,device=DEV))

def train_seed(D, arc, seed, epochs, S, lam_prior, beta, lr):
    torch.manual_seed(seed); np.random.seed(seed); rng=np.random.default_rng(seed)
    model=BioActState(arc,None,lambda_prior=lam_prior,ponder_beta=beta).to(DEV)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr)
    perts=list(D["per_pert"].keys())
    for ep in range(epochs):
        model.train(); order=rng.permutation(len(perts))
        for i in range(0,len(perts),32):
            bp=[perts[j] for j in order[i:i+32]]
            pert,basal,bidx,tdir,bg=make_batch(D,bp,S,rng)
            out=model(pert,basal,bidx,tdir,bg)
            opt.zero_grad(); out["total"].backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0); opt.step()
    model.eval(); EN={}
    with torch.no_grad():
        for i in range(0,len(perts),32):
            bp=perts[i:i+32]; pert,basal,bidx,tdir,bg=make_batch(D,bp,S,rng)
            out=model(pert,basal,bidx,tdir,bg)
            for g,e in zip(bp,out["E_N"].cpu().numpy()): EN[g]=float(e)
    return model, EN, perts

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True); ap.add_argument("--cache_glob",default="/workspace/halt/hf_cache_se")
    ap.add_argument("--epochs",type=int,default=40); ap.add_argument("--seeds",type=int,default=4)
    ap.add_argument("--S",type=int,default=64); ap.add_argument("--lambda_prior",type=float,default=0.25)
    ap.add_argument("--ponder_beta",type=float,default=0.0); ap.add_argument("--lr",type=float,default=2e-3)
    ap.add_argument("--min_cells",type=int,default=20); ap.add_argument("--smoke",action="store_true")
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    print(f"[setup] device={DEV}",flush=True)
    arc,vd,hp,mm=load_arc(a.cache_glob,device=DEV)
    print(f"[arc] loaded missing={len(mm['missing'])} unexpected={len(mm['unexpected'])} layers={arc.transformer_backbone.config.num_hidden_layers} input_dim={vd['input_dim']}",flush=True)
    D=build(cache_glob=a.cache_glob, ckpt_glob=a.cache_glob, min_cells=a.min_cells, cell_set_len=a.S)
    perts=list(D["per_pert"].keys())
    print(f"[data] perts={len(perts)} input_dim={D['input_dim']} n_genes={D['n_genes']}",flush=True)
    # sanity: ARC final-layer prediction quality on native data
    rng=np.random.default_rng(7); sp=sorted(perts,key=lambda g:-D['per_pert'][g]['eff'])[:8]
    pe,ba,bi,td,bg=make_batch(D,sp,a.S,rng)
    with torch.no_grad():
        m0=BioActState(arc,None).to(DEV); cds=m0.per_layer_direction_loss(pe,ba,bi,td,bg).numpy()
    print(f"[sanity] final-layer cos-sim to true dir (top-eff): {[round(1-x,3) for x in cds[:,-1]]}",flush=True)
    if a.smoke: a.epochs=6; a.seeds=2
    ENs=[]; diag=None
    for s in range(a.seeds):
        model,EN,perts=train_seed(D,arc,s,a.epochs,a.S,a.lambda_prior,a.ponder_beta,a.lr)
        ENs.append(np.array([EN[g] for g in perts]))
        if s==0:
            rng=np.random.default_rng(999); pl=[]
            for i in range(0,len(perts),32):
                bp=perts[i:i+32]; pe,ba,bi,td,bg=make_batch(D,bp,a.S,rng)
                pl.append(model.per_layer_direction_loss(pe,ba,bi,td,bg).numpy())
            pl=np.concatenate(pl,0)
            diag=dict(per_layer_mean=[float(x) for x in pl.mean(0)],
                      layer1_minus_layerK=float(pl.mean(0)[0]-pl.mean(0)[-1]),
                      round_of_min_hist={int(r+1):int((pl.argmin(1)==r).sum()) for r in range(8)})
        print(f"[seed {s}] E[N] mean={ENs[-1].mean():.3f} std={ENs[-1].std():.3f} range={ENs[-1].max()-ENs[-1].min():.3f}",flush=True)
    ENs=np.array(ENs); Em=ENs.mean(0)
    eff=np.array([D["per_pert"][g]["eff"] for g in perts]); nde=np.array([D["per_pert"][g]["n_de"] for g in perts]).astype(float)
    ncells=np.array([D["per_pert"][g]["n"] for g in perts]).astype(float)
    pair=[spearmanr(ENs[i],ENs[j])[0] for i in range(a.seeds) for j in range(i+1,a.seeds) if ENs[i].std()>0 and ENs[j].std()>0]
    repro=float(np.mean(pair)) if pair else float("nan")
    P=len(perts); Cov=np.column_stack([np.ones(P),eff,nde,ncells])
    b,*_=np.linalg.lstsq(Cov,Em,rcond=None); resid=Em-Cov@b
    R2=float(1-((Em-Cov@b)**2).sum()/(((Em-Em.mean())**2).sum()+1e-12))
    res=dict(model="arc_st_se_replogle_k562_earlyexit",n_perturbations=P,n_seeds=a.seeds,n_layers=8,
             EN_mean=float(Em.mean()),EN_std=float(Em.std()),EN_cv=float(Em.std()/max(Em.mean(),1e-8)),
             EN_range=float(Em.max()-Em.min()),EN_frac_of_budget=float((Em.max()-Em.min())/8),
             reproducibility_spearman=repro,R2_explained_by_covariates=R2,residual_std=float(resid.std()),
             partial_spearman=dict(eff=partial_spearman(Em,eff,[nde,ncells]),
                                   nde=partial_spearman(Em,nde,[eff,ncells]),
                                   ncells=partial_spearman(Em,ncells,[eff,nde])),
             spearman_EN_effect=float(spearmanr(Em,eff)[0]) if Em.std()>0 else 0.0,
             per_layer_diagnostic=diag)
    import csv
    with open(os.path.join(a.out,"signature.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["perturbation","E_N_mean","effect_size","n_de","n_cells"])
        for k,g in enumerate(perts): w.writerow([g,Em[k],eff[k],int(nde[k]),int(ncells[k])])
    np.save(os.path.join(a.out,"EN_seeds.npy"),ENs)
    json.dump(res,open(os.path.join(a.out,"protocol_results.json"),"w"),indent=2)
    print("[protocol]",json.dumps(res),flush=True); print("[done]",flush=True)

if __name__=="__main__": main()
