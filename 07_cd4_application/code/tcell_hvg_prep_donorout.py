"""tcell_hvg_prep_donorout.py — leave-one-donor-out on POOLED 3-context data (cell-load log1p norm).
Same as the pooled log1p prep (all 3 conditions, per-(gene,context) control-relative response, fixed 2000 HVGs)
but EXCLUDING one donor from both targeting and control aggregation. Keeps the backbone in the multi-context
learning regime (unlike the single-condition folds which underfit). One npz per held-out donor:
  tcell_donorout_hold-{DONOR}.npz  (34k-scale, 3 contexts). --all loops all 4 donors from ONE matrix read."""
import h5py, numpy as np, scipy.sparse as sp, os, argparse
F="/workspace/halt/data/tcell/GWCD4i.pseudobulk_merged.h5ad"; OUT="/workspace/halt/data/tcell/donorout"
HVGF="/workspace/halt/data/tcell/fixed_hvg_names.npy"

def build(holdout, Xh_all, cc,ccc, gt,gtc, dn,dnc, pg,pgc, keep, ncells, hvg_names):
    targ=gtc.index("targeting"); ctrl=gtc.index("non-targeting"); hi=dnc.index(holdout)
    keepd = keep & (dn!=hi)   # training donors only
    resp=[]; bas=[]; genes=[]; ctxs=[]; effs=[]; ndes=[]; ncell_tot=[]
    for ci,cn in enumerate(ccc):
        cm = keepd & (cc==ci)
        b = Xh_all[cm & (gt==ctrl)].mean(0)   # basal from this condition's train-donor controls
        for gi,gn in enumerate(pgc):
            m = cm & (gt==targ) & (pg==gi); n=int(m.sum())
            if n<1: continue
            pm=Xh_all[m].mean(0); r=pm-b
            resp.append(r); bas.append(b); genes.append(gn); ctxs.append(cn)
            effs.append(float(np.linalg.norm(r))); ndes.append(int((np.abs(r)>0.25).sum())); ncell_tot.append(float(ncells[m].sum()))
    resp=np.array(resp,dtype=np.float32); bas=np.array(bas,dtype=np.float32)
    os.makedirs(OUT,exist_ok=True)
    fn=os.path.join(OUT,f"tcell_donorout_hold-{holdout}.npz")
    np.savez(fn, resp=resp, basal=bas, genes=np.array(genes), ctxs=np.array(ctxs),
             eff=np.array(effs,dtype=np.float32), n_de=np.array(ndes), n_cells=np.array(ncell_tot,dtype=np.float32),
             hvg_names=np.array(hvg_names), contexts=np.array(ccc))
    print(f"[save] {fn}  units={resp.shape[0]} (train donors={[d for d in dnc if d!=holdout]})",flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--holdout",default=None); ap.add_argument("--all",action="store_true"); a=ap.parse_args()
    f=h5py.File(F,"r"); o=f["obs"]
    def cat(col):
        g=o[col]; return g["codes"][:], [x.decode() if isinstance(x,bytes) else x for x in g["categories"][:]]
    gt,gtc=cat("guide_type"); cc,ccc=cat("culture_condition"); dn,dnc=cat("donor_id"); pg,pgc=cat("perturbed_gene_name")
    keep=o["keep_for_DE"][:].astype(bool); ncells=o["n_cells"][:]
    var_names=[v.decode() if isinstance(v,bytes) else v for v in f["var"]["gene_name"][:]]
    hvg_names=np.load(HVGF,allow_pickle=True); vmap={g:i for i,g in enumerate(var_names)}
    hvg_idx=np.array([vmap[g] for g in hvg_names]); print(f"[hvg] fixed set {len(hvg_idx)} mapped",flush=True)
    sh=f["X"].attrs["shape"]; print(f"[load] {int(sh[0])} x {int(sh[1])}",flush=True)
    X=sp.csr_matrix((f["X"]["data"][:],f["X"]["indices"][:],f["X"]["indptr"][:]),shape=(int(sh[0]),int(sh[1])))
    X=X.astype(np.float32); np.log1p(X.data,out=X.data)
    Xh_all=X[:,hvg_idx].toarray().astype(np.float32); del X
    print(f"[prep] log1p+HVG in memory {Xh_all.shape}",flush=True)
    donors=dnc if a.all else [a.holdout]
    for d in donors:
        print(f"[fold] holdout={d}",flush=True)
        build(d, Xh_all, cc,ccc, gt,gtc, dn,dnc, pg,pgc, keep, ncells, hvg_names)
    f.close(); print("[done]",flush=True)

if __name__=="__main__": main()
