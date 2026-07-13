"""train_bioact.py — thesis protocol on the TRUE biology-fused ACT (learned PonderNet halt head
reading biological cascade features). Reports: reproducibility across seeds, non-redundancy (R2 of
effect/nDE/cells/graph-degree on E[N] + partial Spearman), and an epoch-curve to test whether the
LEARNED halting survives full training (the toy convergence model collapsed; the question is whether
biology-fused learned halting does not)."""
import sys, os, json, argparse, time
import numpy as np, torch
from scipy.stats import spearmanr, rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from norm_data_lean import build
from bio_act import build_adjacency, BioACT
DEV="cuda" if torch.cuda.is_available() else "cpu"

def partial_spearman(y,x,ctrls):
    def resid(v):
        Y=rankdata(v); C=np.column_stack([np.ones(len(Y))]+[rankdata(c) for c in ctrls])
        b,*_=np.linalg.lstsq(C,Y,rcond=None); return Y-C@b
    ry,rx=resid(y),resid(x)
    return 0.0 if ry.std()<1e-9 or rx.std()<1e-9 else float(spearmanr(ry,rx)[0])

def train_one(A, pert_node, D, seed, epochs, d, max_rounds, lam_prior, beta, lr, snapshots=None, mask_seed=False):
    torch.manual_seed(seed); np.random.seed(seed)
    model=BioACT(A, D["n_genes"], d=d, max_rounds=max_rounds, lambda_prior=lam_prior, ponder_beta=beta,
                 mask_seed_in_readout=mask_seed).to(DEV)
    opt=torch.optim.AdamW(model.parameters(),lr=lr)
    tgt=torch.tensor(D["target_dir"],device=DEV); pidx=torch.tensor(pert_node,device=DEV)
    snaps={}
    for ep in range(epochs):
        model.train(); opt.zero_grad(); out=model(pidx,tgt); out["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if snapshots and (ep+1) in snapshots:
            model.eval()
            with torch.no_grad(): o=model(pidx,tgt); en=o["E_N"].cpu().numpy()
            snaps[ep+1]=dict(EN_cv=float(en.std()/max(en.mean(),1e-8)), EN_range=float(en.max()-en.min()),
                             recon=float(o["recon"].item()), EN_mean=float(en.mean()))
            model.train()
    model.eval()
    with torch.no_grad(): o=model(pidx,tgt); EN=o["E_N"].cpu().numpy(); recon=float(o["recon"].item())
    return EN, recon, snaps

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--h5ad",required=True); ap.add_argument("--graph",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--n_genes",type=int,default=2000); ap.add_argument("--min_cells",type=int,default=20)
    ap.add_argument("--d",type=int,default=128); ap.add_argument("--max_rounds",type=int,default=8)
    ap.add_argument("--lambda_prior",type=float,default=0.3); ap.add_argument("--ponder_beta",type=float,default=0.02)
    ap.add_argument("--epochs",type=int,default=400); ap.add_argument("--lr",type=float,default=5e-3)
    ap.add_argument("--seeds",type=int,default=6); ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--mask_seed",action="store_true")
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    A, node_genes, pert_node_idx = build_adjacency(a.graph, DEV)
    print(f"[setup] device={DEV} graph_nodes={A.size(0)} max_rounds={a.max_rounds}",flush=True)
    D=build(a.h5ad,n_genes=a.n_genes,min_cells=a.min_cells)
    # align: keep only perts that are graph nodes, in D's order; map to node index
    keep=[i for i,g in enumerate(D["perts"]) if g in pert_node_idx]
    pert_node=[pert_node_idx[D["perts"][i]] for i in keep]
    for k in ["target_dir","effect_size","n_de","n_cells"]: D[k]=np.asarray(D[k])[keep]
    D["perts"]=[D["perts"][i] for i in keep]
    P=len(D["perts"]); print(f"[data] perts_on_graph={P} genes={D['n_genes']}",flush=True)
    deg=(A>0).float().sum(1).cpu().numpy()-1; pdeg=deg[pert_node]
    if a.smoke: a.epochs=80; a.seeds=3
    snapshots=[10,20,40,80,160,320,400] if not a.smoke else [20,80]
    ENs=[]; recons=[]; snap0=None
    for s in range(a.seeds):
        t0=time.time(); EN,recon,snaps=train_one(A,pert_node,D,s,a.epochs,a.d,a.max_rounds,a.lambda_prior,a.ponder_beta,a.lr,
                                                  snapshots=snapshots if s==0 else None, mask_seed=a.mask_seed)
        ENs.append(EN); recons.append(recon)
        if s==0: snap0=snaps
        print(f"[seed {s}] E[N] mean={EN.mean():.3f} std={EN.std():.3f} range=[{EN.min():.3f},{EN.max():.3f}] recon={recon:.4f} ({time.time()-t0:.1f}s)",flush=True)
    ENs=np.array(ENs); Em=ENs.mean(0)
    pair=[spearmanr(ENs[i],ENs[j])[0] for i in range(a.seeds) for j in range(i+1,a.seeds) if ENs[i].std()>0 and ENs[j].std()>0]
    repro=float(np.mean(pair)) if pair else float("nan")
    eff,nde,nc=D["effect_size"],D["n_de"].astype(float),D["n_cells"].astype(float)
    Cov=np.column_stack([np.ones(P),eff,nde,nc,pdeg]); b,*_=np.linalg.lstsq(Cov,Em,rcond=None); pred=Cov@b
    R2=float(1-((Em-pred)**2).sum()/(((Em-Em.mean())**2).sum()+1e-12)); resid=Em-pred
    res=dict(model="bio_act",n_perturbations=P,n_seeds=a.seeds,max_rounds=a.max_rounds,
             lambda_prior=a.lambda_prior,ponder_beta=a.ponder_beta,mask_seed=bool(a.mask_seed),recon_mean=float(np.mean(recons)),
             EN_mean=float(Em.mean()),EN_std_across_perts=float(Em.std()),EN_cv=float(Em.std()/max(Em.mean(),1e-8)),
             EN_range=float(Em.max()-Em.min()),EN_frac_of_budget=float((Em.max()-Em.min())/a.max_rounds),
             reproducibility_spearman=repro,R2_explained_by_covariates=R2,residual_std=float(resid.std()),
             partial_spearman=dict(eff=partial_spearman(Em,eff,[nde,nc,pdeg]),nde=partial_spearman(Em,nde,[eff,nc,pdeg]),
                                   ncells=partial_spearman(Em,nc,[eff,nde,pdeg]),degree=partial_spearman(Em,pdeg,[eff,nde,nc])),
             spearman_EN_effect=float(spearmanr(Em,eff)[0]) if Em.std()>0 else 0.0,
             spearman_EN_degree=float(spearmanr(Em,pdeg)[0]) if Em.std()>0 else 0.0,
             epoch_curve=snap0)
    import csv
    with open(os.path.join(a.out,"signature.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["perturbation","E_N_mean","residual","effect_size","n_de","n_cells","graph_degree"])
        for k,g in enumerate(D["perts"]): w.writerow([g,Em[k],resid[k],eff[k],int(nde[k]),int(nc[k]),int(pdeg[k])])
    np.save(os.path.join(a.out,"EN_seeds.npy"),ENs); json.dump(res,open(os.path.join(a.out,"protocol_results.json"),"w"),indent=2)
    print("[protocol]",json.dumps(res),flush=True); print("[done]",flush=True)

if __name__=="__main__": main()
