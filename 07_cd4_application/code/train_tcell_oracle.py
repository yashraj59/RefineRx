"""train_tcell_oracle.py — Stage A (fit+freeze backbone) then Stage C (oracle ACT) on T-cell HVG response.
Signature = per-(gene,context) learned E[N] + halt_confidence. Protocol: reproducibility across seeds,
non-redundancy vs covariates, recovery of oracle r*. Contexts kept separate (no same-gene coupling term)."""
import sys, os, json, argparse, math
import numpy as np, torch
from scipy.stats import spearmanr, rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tcell_backbone import TcellBackbone, OracleACTHead
DEV="cuda"

def partial_spearman(y,x,ctrls):
    def resid(v):
        Y=rankdata(v); C=np.column_stack([np.ones(len(Y))]+[rankdata(c) for c in ctrls])
        b,*_=np.linalg.lstsq(C,Y,rcond=None); return Y-C@b
    ry,rx=resid(y),resid(x)
    return 0.0 if ry.std()<1e-9 or rx.std()<1e-9 else float(spearmanr(ry,rx)[0])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True); ap.add_argument("--data",default="/workspace/halt/data/tcell/tcell_hvg.npz")
    ap.add_argument("--backbone_epochs",type=int,default=60); ap.add_argument("--act_epochs",type=int,default=50)
    ap.add_argument("--warmup",type=int,default=15); ap.add_argument("--seeds",type=int,default=4)
    ap.add_argument("--backbone_seed",type=int,default=0)
    # early stopping on val cos-dist: train to convergence, not a fixed epoch count. backbone_epochs = safety cap.
    ap.add_argument("--es_patience",type=int,default=15); ap.add_argument("--es_min_delta",type=float,default=1e-4)
    ap.add_argument("--H",type=int,default=256); ap.add_argument("--layers",type=int,default=8)
    ap.add_argument("--lr",type=float,default=2e-3); ap.add_argument("--tau",type=float,default=0.05)
    ap.add_argument("--alpha",type=float,default=0.5); ap.add_argument("--beta",type=float,default=1.0)
    ap.add_argument("--gamma",type=float,default=0.1); ap.add_argument("--delta",type=float,default=0.1)
    ap.add_argument("--smoke",action="store_true"); a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    if a.smoke: a.backbone_epochs=15; a.act_epochs=12; a.warmup=4; a.seeds=2
    print(f"[setup] bb_ep={a.backbone_epochs} act_ep={a.act_epochs} warmup={a.warmup} seeds={a.seeds} "
          f"H={a.H} L={a.layers} tau={a.tau} a={a.alpha} b={a.beta} g={a.gamma} d={a.delta}",flush=True)
    Z=np.load(a.data, allow_pickle=True)
    resp=Z["resp"].astype(np.float32); basal=Z["basal"].astype(np.float32)
    genes=Z["genes"]; ctxs=Z["ctxs"]; contexts=list(Z["contexts"])
    eff=Z["eff"].astype(np.float32); nde=Z["n_de"].astype(float); ncells=Z["n_cells"].astype(float)
    n_hvg=resp.shape[1]; U=resp.shape[0]
    gene_list=sorted(set(genes.tolist())); gidx_map={g:i for i,g in enumerate(gene_list)}
    gene_idx=np.array([gidx_map[g] for g in genes]); ctx_idx=np.array([contexts.index(c) for c in ctxs])
    n_pert=len(gene_list); n_ctx=len(contexts)
    # target: L2-normalized response direction (magnitude-free)
    tdir=resp/ (np.linalg.norm(resp,axis=1,keepdims=True)+1e-8)
    print(f"[data] units={U} genes={n_pert} contexts={n_ctx} hvg={n_hvg}",flush=True)
    # train/val split on UNITS (stratified by context)
    rng0=np.random.default_rng(1234); perm=rng0.permutation(U); val_mask=np.zeros(U,bool); val_mask[perm[:U//5]]=True
    G=torch.tensor(gene_idx,device=DEV); C=torch.tensor(ctx_idx,device=DEV)
    B0=torch.tensor(basal,device=DEV); TD=torch.tensor(tdir,device=DEV); BG=torch.zeros((U,n_hvg),device=DEV) # response measured from 0 (already control-relative)
    tr=np.where(~val_mask)[0]
    # Stage A: fit ONE backbone to convergence, then FREEZE and share across all ACT seeds (mirrors K562 —
    # the depth signature is a property of a single fitted model; reproducibility is across ACT-head seeds).
    torch.manual_seed(a.backbone_seed); np.random.seed(a.backbone_seed); rngb=np.random.default_rng(a.backbone_seed)
    bb=TcellBackbone(n_pert,n_ctx,n_hvg,H=a.H,n_layers=a.layers).to(DEV)
    opt=torch.optim.AdamW(bb.parameters(),lr=a.lr)
    va=np.where(val_mask)[0]  # held-out units for early-stopping monitor
    import copy
    def val_cosdist():
        bb.eval()
        with torch.no_grad():
            pr,_=bb.rounds(G[va],C[va],B0[va])
            pd_=torch.nn.functional.normalize(pr,dim=-1,eps=1e-8)
            # round-mean val cos-dist (monitor the depth-averaged predictive quality)
            return float((1-(pd_*TD[va][:,None]).sum(-1)).mean().item())
    # EARLY STOPPING: train until val cos-dist stops improving (min_delta over `patience` epochs).
    # No fixed epoch count — backbone_epochs is only a safety cap. Best weights are restored.
    best_val=float("inf"); best_state=None; best_ep=-1; wait=0
    stop_ep=a.backbone_epochs-1
    for ep in range(a.backbone_epochs):
        bb.train(); order=rngb.permutation(len(tr)); ep_loss=0.0; nb=0
        for i in range(0,len(tr),512):
            bi=tr[order[i:i+512]]; preds,_=bb.rounds(G[bi],C[bi],B0[bi])
            pdir=torch.nn.functional.normalize(preds,dim=-1,eps=1e-8)
            loss=(1-(pdir*TD[bi][:,None]).sum(-1)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); ep_loss+=loss.item(); nb+=1
        vcd=val_cosdist()
        if vcd < best_val - a.es_min_delta:
            best_val=vcd; best_state=copy.deepcopy(bb.state_dict()); best_ep=ep; wait=0
        else:
            wait+=1
        if ep%10==0 or wait>=a.es_patience:
            print(f"[bb ep{ep}] train cos-dist={ep_loss/nb:.4f} val={vcd:.4f} best={best_val:.4f}@{best_ep} wait={wait}",flush=True)
        if wait>=a.es_patience:
            stop_ep=ep; print(f"[bb EARLY STOP] converged at ep{ep}; restoring best epoch {best_ep} (val={best_val:.4f})",flush=True)
            break
    if best_state is not None: bb.load_state_dict(best_state)
    bb_stop_epoch=best_ep; bb_stop_val=best_val
    print(f"[bb DONE] trained {stop_ep+1} epochs, best epoch {best_ep}, val cos-dist {best_val:.4f}",flush=True)
    for p in bb.parameters(): p.requires_grad_(False)
    bb.eval()
    with torch.no_grad():
        preds,toks=bb.rounds(G,C,B0)
        pdir=torch.nn.functional.normalize(preds,dim=-1,eps=1e-8)
        scd=(1-(pdir*TD[:,None]).sum(-1)).cpu().numpy()
    argmin_depth=scd.argmin(1)+1
    oracle_star=OracleACTHead.oracle_round(torch.tensor(scd),tau=a.tau).numpy()+1
    oracle_ref=(argmin_depth,oracle_star)
    # per-round loss curve to confirm the backbone actually improves with depth
    print(f"[ref] argmin range[{argmin_depth.min()},{argmin_depth.max()}] r* range[{oracle_star.min()},{oracle_star.max()}] "
          f"mean{oracle_star.mean():.2f} | per-round cos-dist={np.round(scd.mean(0),4).tolist()}",flush=True)
    ENs=[]; HCs=[]
    for s in range(a.seeds):
        torch.manual_seed(100+s); np.random.seed(100+s); rng=np.random.default_rng(100+s)
        # Stage C: oracle ACT head on the SHARED frozen backbone
        head=OracleACTHead(a.H,a.layers).to(DEV); hopt=torch.optim.AdamW(head.parameters(),lr=3e-3)
        for ep in range(a.act_epochs):
            head.train(); order=rng.permutation(len(tr)); up=(ep>=a.warmup)
            for i in range(0,len(tr),512):
                bi=tr[order[i:i+512]]
                with torch.no_grad(): preds,toks=bb.rounds(G[bi],C[bi],B0[bi])
                out=head.loss(preds,toks,TD[bi],BG[bi],tau=a.tau,alpha=a.alpha,beta=a.beta,gamma=a.gamma,delta=a.delta,use_ponder=up)
                hopt.zero_grad(); out["total"].backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(),1.0); hopt.step()
        head.eval(); EN=np.zeros(U); HC=np.zeros(U)
        for i in range(0,U,1024):
            with torch.no_grad(): preds,toks=bb.rounds(G[i:i+1024],C[i:i+1024],B0[i:i+1024])
            en,hc,_=head.signature(preds,toks,TD[i:i+1024],BG[i:i+1024]); EN[i:i+1024]=en; HC[i:i+1024]=hc
        ENs.append(EN); HCs.append(HC)
        print(f"[seed {s}] E[N] mean={EN.mean():.3f} std={EN.std():.3f} range=[{EN.min():.2f},{EN.max():.2f}] "
              f"rho(EN,oracle)={spearmanr(EN,oracle_ref[1])[0]:.3f} val_std={EN[val_mask].std():.3f} hc={HC.mean():.3f}",flush=True)
    ENs=np.array(ENs); HCs=np.array(HCs); Em=ENs.mean(0); Hm=HCs.mean(0)
    argmin_depth,oracle_star=oracle_ref
    pair=[spearmanr(ENs[i],ENs[j])[0] for i in range(a.seeds) for j in range(i+1,a.seeds) if ENs[i].std()>0 and ENs[j].std()>0]
    repro=float(np.mean(pair)) if pair else float("nan")
    Cov=np.column_stack([np.ones(U),eff,nde,ncells]); b,*_=np.linalg.lstsq(Cov,Em,rcond=None)
    R2=float(1-((Em-Cov@b)**2).sum()/(((Em-Em.mean())**2).sum()+1e-12)) if Em.std()>0 else 1.0
    vm=val_mask
    res=dict(model="tcell_GWCD4i_HVG_ORACLE_ACT", n_units=int(U), n_genes=int(n_pert), n_contexts=int(n_ctx),
             backbone_seed=int(a.backbone_seed),
             backbone_stop_epoch=int(bb_stop_epoch), backbone_val_cosdist=float(bb_stop_val),
             es_patience=int(a.es_patience), es_min_delta=float(a.es_min_delta),
             num_rounds=int(a.layers), n_seeds=a.seeds, tau=a.tau, alpha=a.alpha, beta=a.beta, gamma=a.gamma, delta=a.delta,
             EN_mean=float(Em.mean()), EN_std=float(Em.std()), EN_cv=float(Em.std()/max(Em.mean(),1e-8)),
             EN_range=[float(Em.min()),float(Em.max())], EN_frac_of_budget=float((Em.max()-Em.min())/a.layers),
             reproducibility_spearman=repro, val_reproducibility=float(np.mean([spearmanr(ENs[i][vm],ENs[j][vm])[0] for i in range(a.seeds) for j in range(i+1,a.seeds) if ENs[i][vm].std()>0 and ENs[j][vm].std()>0])) if a.seeds>1 and Em[vm].std()>0 else float("nan"),
             R2_explained_by_covariates=R2, partial_eff=partial_spearman(Em,eff,[nde,ncells]) if Em.std()>0 else 0.0,
             rho_EN_oracle=float(spearmanr(Em,oracle_star)[0]) if Em.std()>0 else 0.0,
             rho_EN_argmin=float(spearmanr(Em,argmin_depth)[0]) if Em.std()>0 else 0.0,
             spearman_EN_effect=float(spearmanr(Em,eff)[0]) if Em.std()>0 else 0.0,
             halt_confidence_mean=float(Hm.mean()))
    import csv
    with open(os.path.join(a.out,"tcell_signature.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["gene","context","expected_rounds","halt_confidence","oracle_rstar","argmin_depth","effect_size","n_de","n_cells","is_val"])
        for k in range(U): w.writerow([genes[k],ctxs[k],round(Em[k],3),round(Hm[k],4),round(oracle_star[k],2),round(argmin_depth[k],2),round(float(eff[k]),4),int(nde[k]),int(ncells[k]),int(val_mask[k])])
    np.save(os.path.join(a.out,"EN_seeds.npy"),ENs)
    json.dump(res,open(os.path.join(a.out,"tcell_protocol_results.json"),"w"),indent=2)
    print("[RESULT]",json.dumps(res),flush=True); print("[done]",flush=True)

if __name__=="__main__": main()
