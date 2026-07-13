"""train_oracle_refine.py — Stage B/C oracle-supervised adaptive-depth on ARC frozen ST-SE-Replogle-k562.

Stage B: per-perturbation oracle stopping round r* (tau) from the per-round distributional loss D_r.
Stage C: joint calibration L = D(mix) + alpha*mean_r D_r + beta*(-log q_{r*}) + gamma*E[R]/R (post warm-up)
         + delta*Huber(e_hat_r, sg D_r). alpha,beta,gamma,delta on VALIDATION perts; test never touched.
Guardrail: NO same-gene/cross-guide depth-coupling term (kept non-circular for the guide-repro test).
Signature = learned expected_rounds + halt_confidence; compared to oracle r* and per-layer argmin depth."""
import sys, os, json, argparse, math
import numpy as np, torch
from scipy.stats import spearmanr, rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/halt/state_repo/src")
from load_arc_state import load_arc
from adaptive_state_refine import AdaptiveStateRefine
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
    ap.add_argument("--out",required=True); ap.add_argument("--epochs",type=int,default=50)
    ap.add_argument("--warmup",type=int,default=15); ap.add_argument("--seeds",type=int,default=4)
    ap.add_argument("--S",type=int,default=64); ap.add_argument("--lr",type=float,default=3e-3)
    ap.add_argument("--tau",type=float,default=0.05); ap.add_argument("--alpha",type=float,default=0.5)
    ap.add_argument("--beta",type=float,default=1.0); ap.add_argument("--gamma",type=float,default=0.1)
    ap.add_argument("--delta",type=float,default=0.1); ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--cache",default="/workspace/halt/hf_cache_se"); ap.add_argument("--cell_line",default="k562")
    ap.add_argument("--split",default="fewshot"); ap.add_argument("--data_dir",default="/dev/shm/k562data")
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    if a.smoke: a.epochs=16; a.warmup=6; a.seeds=2
    print(f"[setup] epochs={a.epochs} warmup={a.warmup} seeds={a.seeds} tau={a.tau} "
          f"alpha={a.alpha} beta={a.beta} gamma={a.gamma} delta={a.delta}",flush=True)
    arc,vd,hp,mm=load_arc(a.cache,device=DEV,cell_line=a.cell_line,split=a.split)
    D=build(cache_glob=a.cache,ckpt_glob=a.cache,min_cells=20,cell_set_len=a.S,
            data_dir=a.data_dir,cell_line=a.cell_line,split=a.split)
    perts=list(D["per_pert"].keys()); S=a.S; P=len(perts)
    # train/val split (val used ONLY for reporting/selection; here fixed hparams, val for honest metrics)
    rng0=np.random.default_rng(1234); perm=rng0.permutation(P); val_idx=set(perm[:max(1,P//5)].tolist())
    val_mask=np.array([i in val_idx for i in range(P)])
    print(f"[data] perts={P} val={val_mask.sum()} train={ (~val_mask).sum() }",flush=True)
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
    # per-layer argmin-depth reference (the rho=0.836 computed signal), and oracle r* reference
    mref=AdaptiveStateRefine(arc).to(DEV); R=mref.num_rounds
    ad=np.zeros((4,P)); orc=np.zeros((4,P))
    for d in range(4):
        rng=np.random.default_rng(700+d); off=0
        for i in range(0,P,32):
            bp=perts[i:i+32]; pe,ba,bi,td,bg=batch(bp,rng)
            _,_,_,scd=mref.eval_signature(pe,ba,bi,td,bg)
            ad[off:off+len(bp)] if False else None
            ad[d,off:off+len(bp)]=scd.argmin(1)+1
            orc[d,off:off+len(bp)]=mref.oracle_round(torch.tensor(scd),tau=a.tau).numpy()+1
            off+=len(bp)
    argmin_depth=ad.mean(0); oracle_star=orc.mean(0)
    print(f"[ref] argmin depth range [{argmin_depth.min():.1f},{argmin_depth.max():.1f}] "
          f"oracle r* range [{oracle_star.min():.1f},{oracle_star.max():.1f}] mean {oracle_star.mean():.2f}",flush=True)
    ENs=[]; HCs=[]
    for s in range(a.seeds):
        torch.manual_seed(s); np.random.seed(s); rng=np.random.default_rng(s)
        model=AdaptiveStateRefine(arc).to(DEV)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
        tr_perts=[perts[i] for i in range(P) if not val_mask[i]]
        for ep in range(a.epochs):
            model.train(); order=rng.permutation(len(tr_perts)); use_ponder=(ep>=a.warmup)
            for i in range(0,len(tr_perts),32):
                bp=[tr_perts[j] for j in order[i:i+32]]; pe,ba,bi,td,bg=batch(bp,rng)
                out=model.oracle_loss(pe,ba,bi,td,bg,tau=a.tau,alpha=a.alpha,beta=a.beta,
                                      gamma=a.gamma,delta=a.delta,use_ponder=use_ponder)
                opt.zero_grad(); out["total"].backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0); opt.step()
        model.eval(); EN=np.zeros(P); HC=np.zeros(P); off=0; rng2=np.random.default_rng(9000+s)
        for i in range(0,P,32):
            bp=perts[i:i+32]; pe,ba,bi,td,bg=batch(bp,rng2)
            en,hc,_,_=model.eval_signature(pe,ba,bi,td,bg); EN[off:off+len(bp)]=en; HC[off:off+len(bp)]=hc; off+=len(bp)
        ENs.append(EN); HCs.append(HC)
        vspread=EN[val_mask]
        print(f"[seed {s}] E[N] mean={EN.mean():.3f} std={EN.std():.3f} range=[{EN.min():.2f},{EN.max():.2f}] "
              f"rho(EN,oracle)={spearmanr(EN,oracle_star)[0]:.3f} rho(EN,argmin)={spearmanr(EN,argmin_depth)[0]:.3f} "
              f"val_std={vspread.std():.3f} hc={HC.mean():.3f}",flush=True)
    ENs=np.array(ENs); HCs=np.array(HCs); Em=ENs.mean(0); Hm=HCs.mean(0)
    eff=np.array([D["per_pert"][g]["eff"] for g in perts]); nde=np.array([D["per_pert"][g]["n_de"] for g in perts]).astype(float)
    ncells=np.array([D["per_pert"][g]["n"] for g in perts]).astype(float)
    pair=[spearmanr(ENs[i],ENs[j])[0] for i in range(a.seeds) for j in range(i+1,a.seeds) if ENs[i].std()>0 and ENs[j].std()>0]
    repro=float(np.mean(pair)) if pair else float("nan")
    Cov=np.column_stack([np.ones(P),eff,nde,ncells]); b,*_=np.linalg.lstsq(Cov,Em,rcond=None)
    R2=float(1-((Em-Cov@b)**2).sum()/(((Em-Em.mean())**2).sum()+1e-12)) if Em.std()>0 else 1.0
    # honest metrics on VALIDATION perts only for the headline
    vm=val_mask
    res=dict(model=f"arc_st_se_replogle_{a.split}_{a.cell_line}_ORACLE_ACT", cell_line=a.cell_line, split=a.split,
             n_perturbations=P, num_rounds=int(R), n_seeds=a.seeds,
             tau=a.tau, alpha=a.alpha, beta=a.beta, gamma=a.gamma, delta=a.delta, warmup=a.warmup, epochs=a.epochs,
             EN_mean=float(Em.mean()), EN_std=float(Em.std()), EN_cv=float(Em.std()/max(Em.mean(),1e-8)),
             EN_range=[float(Em.min()),float(Em.max())], EN_frac_of_budget=float((Em.max()-Em.min())/R),
             reproducibility_spearman=repro, R2_explained_by_covariates=R2,
             rho_EN_oracle=float(spearmanr(Em,oracle_star)[0]) if Em.std()>0 else 0.0,
             rho_EN_argmin=float(spearmanr(Em,argmin_depth)[0]) if Em.std()>0 else 0.0,
             spearman_EN_effect=float(spearmanr(Em,eff)[0]) if Em.std()>0 else 0.0,
             partial_eff=partial_spearman(Em,eff,[nde,ncells]) if Em.std()>0 else 0.0,
             val_reproducibility=float(np.mean([spearmanr(ENs[i][vm],ENs[j][vm])[0] for i in range(a.seeds) for j in range(i+1,a.seeds) if ENs[i][vm].std()>0 and ENs[j][vm].std()>0])) if a.seeds>1 and Em[vm].std()>0 else float("nan"),
             val_rho_EN_oracle=float(spearmanr(Em[vm],oracle_star[vm])[0]) if Em[vm].std()>0 else 0.0,
             halt_confidence_mean=float(Hm.mean()), rho_haltconf_effect=float(spearmanr(Hm,eff)[0]) if Hm.std()>0 else 0.0,
             oracle_star_mean=float(oracle_star.mean()), oracle_star_std=float(oracle_star.std()))
    import csv
    with open(os.path.join(a.out,"oracle_signature.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["perturbation","expected_rounds","halt_confidence","oracle_rstar","argmin_depth","effect_size","n_de","n_cells","is_val"])
        for k,g in enumerate(perts): w.writerow([g,round(Em[k],3),round(Hm[k],4),round(oracle_star[k],3),round(argmin_depth[k],3),round(eff[k],4),int(nde[k]),int(ncells[k]),int(val_mask[k])])
    np.save(os.path.join(a.out,"EN_seeds.npy"),ENs); np.save(os.path.join(a.out,"oracle_star.npy"),oracle_star)
    json.dump(res,open(os.path.join(a.out,"oracle_protocol_results.json"),"w"),indent=2)
    print("[RESULT]",json.dumps(res),flush=True); print("[done]",flush=True)

if __name__=="__main__": main()
