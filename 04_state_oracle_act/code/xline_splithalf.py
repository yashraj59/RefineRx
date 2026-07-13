"""xline_splithalf.py — within-line reproducibility of the DATA-derived depth signal.
Split each perturbation's cells into two disjoint halves, re-estimate the control-relative
response on each, run the FROZEN ARC backbone's per-round refinement, and read the oracle
stopping round r* + argmin depth on each half. Cross-half Spearman(r*_A, r*_B) tests whether
depth is a stable within-cell-type property (robust to which cells you sample) — no ACT head,
no seed lottery, no cross-cell-type confound. Effect-size cross-half rho is the sanity ceiling."""
import sys, os, json, argparse, glob, pickle
import numpy as np, torch, h5py
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/halt/state_repo/src")
from load_arc_state import load_arc
from adaptive_state_refine import AdaptiveStateRefine
DEV="cuda"
def tonp(x): return x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)
def read_cat(f, col):
    g=f["obs"][col]
    if isinstance(g,h5py.Group):
        cats=np.array([c.decode() if isinstance(c,bytes) else str(c) for c in g["categories"][:]])
        return cats[g["codes"][:]]
    v=g[:]; return np.array([x.decode() if isinstance(x,bytes) else x for x in v])
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache",required=True); ap.add_argument("--cell_line",required=True)
    ap.add_argument("--split",default="fewshot"); ap.add_argument("--out",required=True)
    ap.add_argument("--S",type=int,default=64); ap.add_argument("--tau",type=float,default=0.05)
    ap.add_argument("--min_cells",type=int,default=40); ap.add_argument("--seed",type=int,default=0)
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    arc,vd,hp,mm=load_arc(a.cache,device=DEV,cell_line=a.cell_line,split=a.split)
    real=glob.glob(f"{a.cache}/**/{a.split}/{a.cell_line}/eval_best.ckpt/adata_real.h5ad",recursive=True)[0]
    pmap_path=glob.glob(f"{a.cache}/**/{a.split}/{a.cell_line}/pert_onehot_map.pt",recursive=True)[0]
    bmap_path=glob.glob(f"{a.cache}/**/{a.split}/{a.cell_line}/batch_onehot_map.pkl",recursive=True)[0]
    pmap=torch.load(pmap_path,map_location="cpu",weights_only=False)
    pmap={str(k):(v.argmax().item() if hasattr(v,"argmax") else int(v)) for k,v in pmap.items()}
    pert_dim=2024
    f=h5py.File(real,"r")
    X=f["X"][:].astype(np.float32); Xs=f["obsm"]["X_state"][:].astype(np.float32); gene=read_cat(f,"gene")
    try:
        gg=read_cat(f,"gem_group"); bmap=pickle.load(open(bmap_path,"rb"))
        bmap={str(k):(v.argmax() if hasattr(v,"argmax") else int(v)) for k,v in bmap.items()}
        batch_all=np.array([int(bmap.get(str(x),0)) for x in gg])
    except Exception: batch_all=np.zeros(len(gene),dtype=int)
    f.close()
    labels=set(np.unique(gene)); ctrl_label=next((c for c in ["non-targeting","control","NTC","neg"] if c in labels),None)
    ctrl=gene==ctrl_label; ctrl_mean=X[ctrl].mean(0); Xs_ctrl=Xs[ctrl]; batch_ctrl=batch_all[ctrl]; n_genes=X.shape[1]
    rng=np.random.default_rng(a.seed); perts=[]; respA={}; respB={}; onh={}
    for g in np.unique(gene):
        if g==ctrl_label or g not in pmap: continue
        idx=np.where(gene==g)[0]
        if len(idx)<a.min_cells: continue
        perm=rng.permutation(idx); h=len(perm)//2; A=perm[:h]; B=perm[h:2*h]
        respA[g]=X[A].mean(0)-ctrl_mean; respB[g]=X[B].mean(0)-ctrl_mean; onh[g]=pmap[g]; perts.append(g)
    P=len(perts); print(f"[data] line={a.cell_line} perts={P} (>= {a.min_cells} cells)",flush=True)
    mref=AdaptiveStateRefine(arc).to(DEV); R=mref.num_rounds; S=a.S
    def make_batch(bp, rng2):
        sel=[rng2.choice(len(Xs_ctrl),S,replace=len(Xs_ctrl)<S) for _ in bp]
        basal=np.stack([Xs_ctrl[s] for s in sel]); bidx=np.stack([batch_ctrl[s] for s in sel])
        pe=np.zeros((len(bp),S,pert_dim),dtype=np.float32)
        for i,g in enumerate(bp): pe[i,:,onh[g]]=1.0
        bg=np.broadcast_to(ctrl_mean,(len(bp),S,n_genes))
        return (torch.tensor(pe,device=DEV),torch.tensor(basal,device=DEV),
                torch.tensor(bidx,dtype=torch.long,device=DEV),
                torch.tensor(np.ascontiguousarray(bg),dtype=torch.float32,device=DEV))
    def td_of(bp, resp):
        r=np.stack([resp[g] for g in bp]); d=r/(np.linalg.norm(r,axis=1,keepdims=True)+1e-8)
        return torch.tensor(np.repeat(d[:,None,:],S,axis=1),dtype=torch.float32,device=DEV)
    rstarA=np.zeros(P); rstarB=np.zeros(P); argA=np.zeros(P); argB=np.zeros(P); rng2=np.random.default_rng(9000+a.seed); off=0
    for i in range(0,P,32):
        bp=perts[i:i+32]; pe,ba,bi,bg=make_batch(bp,rng2)
        with torch.no_grad():
            _,_,_,scdA=mref.eval_signature(pe,ba,bi,td_of(bp,respA),bg)
            _,_,_,scdB=mref.eval_signature(pe,ba,bi,td_of(bp,respB),bg)
        scdA=tonp(scdA); scdB=tonp(scdB)
        rstarA[off:off+len(bp)]=mref.oracle_round(torch.tensor(scdA),tau=a.tau).numpy()+1
        rstarB[off:off+len(bp)]=mref.oracle_round(torch.tensor(scdB),tau=a.tau).numpy()+1
        argA[off:off+len(bp)]=scdA.argmin(1)+1; argB[off:off+len(bp)]=scdB.argmin(1)+1; off+=len(bp)
    effA=np.array([np.linalg.norm(respA[g]) for g in perts]); effB=np.array([np.linalg.norm(respB[g]) for g in perts])
    res=dict(cell_line=a.cell_line, split=a.split, n_perts=P, num_rounds=int(R), tau=a.tau, min_cells=a.min_cells,
             oracle_rstar_splithalf_spearman=float(spearmanr(rstarA,rstarB)[0]),
             argmin_splithalf_spearman=float(spearmanr(argA,argB)[0]),
             effect_splithalf_spearman=float(spearmanr(effA,effB)[0]),
             rstarA_mean=float(rstarA.mean()), rstarB_mean=float(rstarB.mean()))
    import csv
    with open(os.path.join(a.out,f"splithalf_{a.cell_line}.csv"),"w",newline="") as fo:
        w=csv.writer(fo); w.writerow(["pert","rstar_A","rstar_B","argmin_A","argmin_B","eff_A","eff_B"])
        for k,g in enumerate(perts): w.writerow([g,rstarA[k],rstarB[k],argA[k],argB[k],round(float(effA[k]),4),round(float(effB[k]),4)])
    json.dump(res,open(os.path.join(a.out,f"splithalf_{a.cell_line}.json"),"w"),indent=2)
    print("[RESULT]",json.dumps(res),flush=True); print("[done]",flush=True)
if __name__=="__main__": main()
