"""train_bio.py — thesis protocol on the biology-FUSED BioHaltRefiner. Depth = rounds until the
perturbation's cascade FRONT saturates on the FIXED STRING graph. Because depth is pinned to fixed
biology (not a free gate, not reconstruction convergence), the test emphasis shifts to NON-REDUNDANCY:
is network-cascade-depth more than graph degree / effect size? Reports reproducibility across seeds
(should hold by construction; confirmed), R^2 of covariates INCLUDING graph degree, partial Spearman."""
import sys, os, json, argparse, time
import numpy as np, torch
from scipy.stats import spearmanr, rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from norm_data_lean import build
from bio_halt import build_adjacency, BioHaltRefiner
DEV="cuda" if torch.cuda.is_available() else "cpu"

def partial_spearman(y,x,ctrls):
    def resid(v):
        Y=rankdata(v); C=np.column_stack([np.ones(len(Y))]+[rankdata(c) for c in ctrls])
        b,*_=np.linalg.lstsq(C,Y,rcond=None); return Y-C@b
    ry,rx=resid(y),resid(x)
    return 0.0 if ry.std()<1e-9 or rx.std()<1e-9 else float(spearmanr(ry,rx)[0])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--h5ad",required=True); ap.add_argument("--graph",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--n_genes",type=int,default=2000); ap.add_argument("--min_cells",type=int,default=20)
    ap.add_argument("--d",type=int,default=128); ap.add_argument("--max_rounds",type=int,default=8)
    ap.add_argument("--eps_sat",type=float,default=0.01); ap.add_argument("--epochs",type=int,default=400)
    ap.add_argument("--lr",type=float,default=5e-3); ap.add_argument("--seeds",type=int,default=6)
    ap.add_argument("--smoke",action="store_true")
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    A, graph_genes = build_adjacency(a.graph, DEV)
    print(f"[setup] device={DEV} graph_nodes={A.size(0)} max_rounds={a.max_rounds}",flush=True)
    D=build(a.h5ad,n_genes=a.n_genes,min_cells=a.min_cells)
    # align perturbation order to graph node order (graph built from the same sorted 88 genes)
    assert graph_genes == D["perts"], f"gene order mismatch: {len(graph_genes)} vs {len(D['perts'])}"
    P=len(D["perts"]); print(f"[data] perts={P} genes={D['n_genes']}",flush=True)
    # graph degree per node (the covariate we MUST regress out)
    deg = (A > 0).float().sum(1).cpu().numpy() - 1  # minus self-loop
    if a.smoke: a.epochs=60; a.seeds=3
    tgt=torch.tensor(D["target_dir"],device=DEV); idx=torch.arange(P,device=DEV)
    Ds=[]; alphas=[]; recons=[]
    for s in range(a.seeds):
        torch.manual_seed(s); np.random.seed(s)
        model=BioHaltRefiner(A, D["n_genes"], d=a.d, max_rounds=a.max_rounds, eps_sat=a.eps_sat).to(DEV)
        opt=torch.optim.AdamW(model.parameters(),lr=a.lr)
        t0=time.time()
        for ep in range(a.epochs):
            model.train(); opt.zero_grad(); out=model(idx,tgt); out["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval()
        with torch.no_grad(): o=model(idx,tgt); dep=o["depth"].cpu().numpy(); recon=float(o["recon"].item())
        Ds.append(dep); alphas.append(float(model.alpha.item())); recons.append(recon)
        print(f"[seed {s}] depth mean={dep.mean():.3f} std={dep.std():.3f} range=[{dep.min():.0f},{dep.max():.0f}] alpha={model.alpha.item():.3f} recon={recon:.4f} ({time.time()-t0:.1f}s)",flush=True)
    Ds=np.array(Ds); Dm=Ds.mean(0)
    pair=[spearmanr(Ds[i],Ds[j])[0] for i in range(a.seeds) for j in range(i+1,a.seeds) if Ds[i].std()>0 and Ds[j].std()>0]
    repro=float(np.mean(pair)) if pair else float("nan")
    eff,nde,nc=D["effect_size"],D["n_de"].astype(float),D["n_cells"].astype(float)
    Cov=np.column_stack([np.ones(P),eff,nde,nc,deg]); b,*_=np.linalg.lstsq(Cov,Dm,rcond=None); pred=Cov@b
    R2=float(1-((Dm-pred)**2).sum()/(((Dm-Dm.mean())**2).sum()+1e-12)); resid=Dm-pred
    res=dict(model="bio_halt",n_perturbations=P,n_seeds=a.seeds,max_rounds=a.max_rounds,eps_sat=a.eps_sat,
             alpha_mean=float(np.mean(alphas)),recon_mean=float(np.mean(recons)),
             depth_mean=float(Dm.mean()),depth_std_across_perts=float(Dm.std()),
             depth_cv=float(Dm.std()/max(Dm.mean(),1e-8)),depth_range=float(Dm.max()-Dm.min()),
             depth_frac_of_budget=float((Dm.max()-Dm.min())/a.max_rounds),
             reproducibility_spearman=repro,R2_explained_by_covariates=R2,residual_std=float(resid.std()),
             partial_spearman=dict(eff=partial_spearman(Dm,eff,[nde,nc,deg]),
                                   nde=partial_spearman(Dm,nde,[eff,nc,deg]),
                                   ncells=partial_spearman(Dm,nc,[eff,nde,deg]),
                                   degree=partial_spearman(Dm,deg,[eff,nde,nc])),
             spearman_depth_effect=float(spearmanr(Dm,eff)[0]) if Dm.std()>0 else 0.0,
             spearman_depth_degree=float(spearmanr(Dm,deg)[0]) if Dm.std()>0 else 0.0)
    import csv
    with open(os.path.join(a.out,"signature.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["perturbation","depth_mean","residual","effect_size","n_de","n_cells","graph_degree"])
        for k,g in enumerate(D["perts"]): w.writerow([g,Dm[k],resid[k],eff[k],int(nde[k]),int(nc[k]),int(deg[k])])
    np.save(os.path.join(a.out,"depth_seeds.npy"),Ds); json.dump(res,open(os.path.join(a.out,"protocol_results.json"),"w"),indent=2)
    print("[protocol]",json.dumps(res),flush=True); print("[done]",flush=True)

if __name__=="__main__": main()
