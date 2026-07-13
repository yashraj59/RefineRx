"""train_bioact_txpert.py — thesis protocol on the biology-fused ACT grafted onto TxPert.
Reproducibility across seeds; magnitude-free E[N] signature; non-redundancy vs effect_size + graph
degree; and the DECISIVE per-hop diagnostic (does TxPert's learned message passing give a per-hop
cosine-distance curve that varies across perturbations, or saturate at hop 1 like the fixed toy?)."""
import sys, os, json, argparse, time
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr, rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from txpert_data import build
from bio_act_txpert import BioActTxPert
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

def partial_spearman(y,x,ctrls):
    def resid(v):
        Y=rankdata(v); C=np.column_stack([np.ones(len(Y))]+[rankdata(c) for c in ctrls])
        b,*_=np.linalg.lstsq(C,Y,rcond=None); return Y-C@b
    ry,rx=resid(y),resid(x)
    return 0.0 if ry.std()<1e-9 or rx.std()<1e-9 else float(spearmanr(ry,rx)[0])

def train_seed(D, seed, args, edge_index, Xnode, pert_node, keep):
    torch.manual_seed(seed); np.random.seed(seed)
    n_nodes=len(D['node_genes'])
    ds=TensorDataset(pert_node[keep], torch.tensor(keep))
    dl=DataLoader(ds,batch_size=args.batch_size,shuffle=True,drop_last=True)
    model=BioActTxPert(n_nodes,d=args.d,n_hops=args.n_hops,lambda_prior=args.lambda_prior,
                       ponder_beta=args.ponder_beta,ponder_weight=args.ponder_weight).to(DEV)
    with torch.no_grad(): model.basal.copy_(torch.tensor(D['ctrl_mean'],device=DEV))
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    for ep in range(args.epochs):
        model.train()
        for pn,ci in dl:
            pn=pn.to(DEV); tgt=Xnode[ci].to(DEV)
            out=model(pn,tgt,edge_index)
            opt.zero_grad(); out['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    model.eval()
    sig=model.signature(edge_index)
    return model, sig['E_N'].numpy()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--h5ad',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--n_nodes',type=int,default=2000); ap.add_argument('--knn',type=int,default=16)
    ap.add_argument('--n_hops',type=int,default=8); ap.add_argument('--d',type=int,default=128)
    ap.add_argument('--epochs',type=int,default=25); ap.add_argument('--batch_size',type=int,default=1024)
    ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--ponder_weight',type=float,default=0.1)
    ap.add_argument('--lambda_prior',type=float,default=0.2); ap.add_argument('--ponder_beta',type=float,default=0.05)
    ap.add_argument('--seeds',type=int,default=4); ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args(); os.makedirs(args.out,exist_ok=True)
    print(f'[setup] device={DEV} torch={torch.__version__}',flush=True)
    D=build(args.h5ad,n_nodes=args.n_nodes,knn=args.knn)
    n_nodes=len(D['node_genes'])
    print(f"[data] nodes={n_nodes} edges={D['edge_index'].shape[1]} perts={len(D['measured_pert'])}",flush=True)
    edge_index=torch.tensor(D['edge_index'],device=DEV)
    Xnode=torch.tensor(D['Xnode'],dtype=torch.float32)
    pert_node=torch.tensor(D['pert_node'],dtype=torch.long)
    keep=D['keep_idx']
    if args.smoke: keep=keep[:3000]; args.epochs=3; args.seeds=3
    # per-pert covariates
    gene_to_node=D['gene_to_node']; ctrl_mean=D['ctrl_mean']; deg=D['deg']; tg=D['tg']; Xn=D['Xnode']
    perts=D['measured_pert']; nodes=np.array([gene_to_node[g] for g in perts])
    eff=np.array([np.linalg.norm(Xn[tg==g].mean(0)-ctrl_mean) for g in perts])
    ncells=np.array([int((tg==g).sum()) for g in perts]); pdeg=deg[nodes].astype(float)
    # target DIRECTIONS for the diagnostic
    tdir=np.array([ (Xn[tg==g].mean(0)-ctrl_mean) for g in perts]); tdir=tdir/ (np.linalg.norm(tdir,axis=1,keepdims=True)+1e-8)
    ENs=[]; diag=None
    for s in range(args.seeds):
        t0=time.time(); model,EN=train_seed(D,s,args,edge_index,Xnode,pert_node,keep)
        ENs.append(EN[nodes])  # E[N] at the perturbed nodes only
        if s==0:
            pd=model.per_hop_direction_loss(torch.tensor(nodes,device=DEV),
                                            torch.tensor(tdir,dtype=torch.float32,device=DEV),edge_index).numpy()
            diag=dict(per_hop_mean=[float(x) for x in pd.mean(0)],
                      hop1_minus_hopK=float(pd.mean(0)[0]-pd.mean(0)[-1]),
                      round_of_min_hist={int(r+1):int((pd.argmin(1)==r).sum()) for r in range(args.n_hops)},
                      per_hop_std_across_perts=[float(x) for x in pd.std(0)])
        print(f"[seed {s}] E[N]@perts mean={EN[nodes].mean():.3f} std={EN[nodes].std():.3f} range={EN[nodes].max()-EN[nodes].min():.3f} ({time.time()-t0:.1f}s)",flush=True)
    ENs=np.array(ENs); Em=ENs.mean(0)
    pair=[spearmanr(ENs[i],ENs[j])[0] for i in range(args.seeds) for j in range(i+1,args.seeds) if ENs[i].std()>0 and ENs[j].std()>0]
    repro=float(np.mean(pair)) if pair else float('nan')
    P=len(perts); Cov=np.column_stack([np.ones(P),eff,ncells.astype(float),pdeg])
    b,*_=np.linalg.lstsq(Cov,Em,rcond=None); pred=Cov@b
    R2=float(1-((Em-pred)**2).sum()/(((Em-Em.mean())**2).sum()+1e-12)); resid=Em-pred
    res=dict(model='bio_act_txpert',n_perturbations=P,n_seeds=args.seeds,n_hops=args.n_hops,
             EN_mean=float(Em.mean()),EN_std_across_perts=float(Em.std()),EN_cv=float(Em.std()/max(Em.mean(),1e-8)),
             EN_range=float(Em.max()-Em.min()),EN_frac_of_budget=float((Em.max()-Em.min())/args.n_hops),
             reproducibility_spearman=repro,R2_explained_by_covariates=R2,residual_std=float(resid.std()),
             partial_spearman=dict(eff=partial_spearman(Em,eff,[ncells.astype(float),pdeg]),
                                   ncells=partial_spearman(Em,ncells.astype(float),[eff,pdeg]),
                                   degree=partial_spearman(Em,pdeg,[eff,ncells.astype(float)])),
             spearman_EN_effect=float(spearmanr(Em,eff)[0]) if Em.std()>0 else 0.0,
             spearman_EN_degree=float(spearmanr(Em,pdeg)[0]) if Em.std()>0 else 0.0,
             per_hop_diagnostic=diag)
    import csv
    with open(os.path.join(args.out,'signature.csv'),'w',newline='') as f:
        w=csv.writer(f); w.writerow(['perturbation','E_N_mean','residual','effect_size','n_cells','graph_degree'])
        for k,g in enumerate(perts): w.writerow([g,Em[k],resid[k],eff[k],int(ncells[k]),int(pdeg[k])])
    np.save(os.path.join(args.out,'EN_seeds.npy'),ENs); json.dump(res,open(os.path.join(args.out,'protocol_results.json'),'w'),indent=2)
    print('[protocol]',json.dumps(res),flush=True); print('[done]',flush=True)

if __name__=='__main__': main()
