"""tcell_hvg_prep_lodo.py — leave-one-donor-out, single-condition HVG response prep (cell-load log1p norm).
For a given culture_condition and held-out donor, aggregate per-gene control-relative response using ONLY the
3 training donors (log1p raw counts, NO CP10K), on the FIXED 2000 HVGs from the pooled log1p run so every fold
and condition share identical features. Emits one npz per (condition, heldout_donor):
  tcell_lodo_{COND}_hold-{DONOR}.npz  with resp/basal/genes/ctxs(single)/eff/n_de/n_cells/hvg_names/contexts.
Usage: python tcell_hvg_prep_lodo.py --cond Rest --holdout CE0006864
       (or --all to loop every condition x donor)
Normalization in-place on CSR .data to keep peak RAM low."""
import h5py, numpy as np, scipy.sparse as sp, os, argparse

F="/workspace/halt/data/tcell/GWCD4i.pseudobulk_merged.h5ad"; OUT="/workspace/halt/data/tcell/lodo"
HVGF="/workspace/halt/data/tcell/fixed_hvg_names.npy"

def build(cond, holdout, Xh_all, cc,ccc, gt,gtc, dn,dnc, pg,pgc, keep, ncells, hvg_names):
    ci=ccc.index(cond); targ=gtc.index("targeting"); ctrl=gtc.index("non-targeting"); hi=dnc.index(holdout)
    # rows in this condition, training donors only (exclude held-out), keep_for_DE
    base=(cc==ci)&keep&(dn!=hi)
    rows=np.where(base)[0]
    Xh=Xh_all[rows]  # already log1p'd + HVG-subset in memory
    gt_s=gt[rows]; pg_s=pg[rows]; nc_s=ncells[rows]
    ctrl_mask=(gt_s==ctrl); b=Xh[ctrl_mask].mean(0)  # basal from train-donor controls
    resp=[]; bas=[]; genes=[]; ctxs=[]; effs=[]; ndes=[]; ncell_tot=[]
    for gi,gn in enumerate(pgc):
        m=(gt_s==targ)&(pg_s==gi); n=int(m.sum())
        if n<1: continue
        pm=Xh[m].mean(0); r=pm-b
        resp.append(r); bas.append(b); genes.append(gn); ctxs.append(cond)
        effs.append(float(np.linalg.norm(r))); ndes.append(int((np.abs(r)>0.25).sum())); ncell_tot.append(float(nc_s[m].sum()))
    resp=np.array(resp,dtype=np.float32); bas=np.array(bas,dtype=np.float32)
    os.makedirs(OUT,exist_ok=True)
    fn=os.path.join(OUT,f"tcell_lodo_{cond}_hold-{holdout}.npz")
    np.savez(fn, resp=resp, basal=bas, genes=np.array(genes), ctxs=np.array(ctxs),
             eff=np.array(effs,dtype=np.float32), n_de=np.array(ndes), n_cells=np.array(ncell_tot,dtype=np.float32),
             hvg_names=np.array(hvg_names), contexts=np.array([cond]))
    print(f"[save] {fn}  units={resp.shape[0]} genes={len(set(genes))} (train donors={[d for d in dnc if d!=holdout]})",flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cond",default=None); ap.add_argument("--holdout",default=None)
    ap.add_argument("--all",action="store_true"); a=ap.parse_args()
    f=h5py.File(F,"r"); o=f["obs"]
    def cat(col):
        g=o[col]; return g["codes"][:], [x.decode() if isinstance(x,bytes) else x for x in g["categories"][:]]
    gt,gtc=cat("guide_type"); cc,ccc=cat("culture_condition"); dn,dnc=cat("donor_id"); pg,pgc=cat("perturbed_gene_name")
    keep=o["keep_for_DE"][:].astype(bool); ncells=o["n_cells"][:]
    var_names=[v.decode() if isinstance(v,bytes) else v for v in f["var"]["gene_name"][:]]
    hvg_names=np.load(HVGF,allow_pickle=True); vmap={g:i for i,g in enumerate(var_names)}
    hvg_idx=np.array([vmap[g] for g in hvg_names]); print(f"[hvg] fixed set {len(hvg_idx)} mapped",flush=True)
    # READ X ONCE: full matrix -> log1p -> subset to fixed HVGs (all downstream folds slice from this in memory)
    sh=f["X"].attrs["shape"]; print(f"[load] {int(sh[0])} x {int(sh[1])}",flush=True)
    X=sp.csr_matrix((f["X"]["data"][:],f["X"]["indices"][:],f["X"]["indptr"][:]),shape=(int(sh[0]),int(sh[1])))
    X=X.astype(np.float32); np.log1p(X.data,out=X.data)
    Xh_all=X[:,hvg_idx].toarray().astype(np.float32); del X
    print(f"[prep] log1p + HVG subset in memory: {Xh_all.shape}",flush=True)
    jobs=[(c,d) for c in ccc for d in dnc] if a.all else [(a.cond,a.holdout)]
    for c_,d_ in jobs:
        print(f"[fold] cond={c_} holdout={d_}",flush=True)
        build(c_,d_, Xh_all, cc,ccc, gt,gtc, dn,dnc, pg,pgc, keep, ncells, hvg_names)
    f.close(); print("[done]",flush=True)

if __name__=="__main__": main()
