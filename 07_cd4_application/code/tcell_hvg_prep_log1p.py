"""tcell_hvg_prep_log1p.py — CELL-LOAD normalization variant: raw counts -> log1p ONLY (no CP10K/1e4 target-sum).
Per cell-load's model-input contract (is_log1p=True; no normalize_total in the training path), the STATE/HVG
model expects log1p(raw counts) with NO library-size rescaling. This differs from tcell_hvg_prep.py which applied
CP10K(target_sum=1e4)+log1p. Everything else (2000 HVG selection, per-(gene,context) control-relative response,
covariates) is identical, so a side-by-side with tcell_hvg.npz isolates the normalization effect on depth signal.
Output: tcell_hvg_log1p.npz."""
import h5py, numpy as np, scipy.sparse as sp, json, os
F="/workspace/halt/data/tcell/GWCD4i.pseudobulk_merged.h5ad"; OUT="/workspace/halt/data/tcell"
f=h5py.File(F,"r")
sh=f["X"].attrs["shape"]; n_obs,n_var=int(sh[0]),int(sh[1])
print(f"[load] {n_obs} x {n_var}",flush=True)
X=sp.csr_matrix((f["X"]["data"][:],f["X"]["indices"][:],f["X"]["indptr"][:]),shape=(n_obs,n_var))
o=f["obs"]
def cat(col):
    g=o[col]; return g["codes"][:], [c.decode() if isinstance(c,bytes) else c for c in g["categories"][:]]
gt,gtc=cat("guide_type"); cc,ccc=cat("culture_condition"); dn,dnc=cat("donor_id")
pg,pgc=cat("perturbed_gene_name")
keepDE=o["keep_for_DE"][:]; ncells=o["n_cells"][:]
var_names=[v.decode() if isinstance(v,bytes) else v for v in f["var"]["gene_name"][:]]
f.close()
targ_code=gtc.index("targeting"); ctrl_code=gtc.index("non-targeting")
keep = keepDE.astype(bool)
print(f"[qc] keep_for_DE rows: {keep.sum()}",flush=True)
# CELL-LOAD normalization: log1p on RAW counts, NO CP10K/1e4 rescaling.
Xk=X[keep].astype(np.float32); del X; cc_k=cc[keep]; gt_k=gt[keep]; pg_k=pg[keep]; nc_k=ncells[keep]
np.log1p(Xk.data, out=Xk.data)
print("[norm] log1p ONLY (cell-load convention; no CP10K/1e4) done",flush=True)
# HVG: top 2000 by variance across kept rows (dense per-gene var via E[X^2]-E[X]^2)
mean=np.asarray(Xk.mean(0)).ravel(); meansq=np.asarray(Xk.multiply(Xk).mean(0)).ravel()
varg=meansq-mean**2; hvg=np.argsort(varg)[::-1][:2000]; hvg.sort()
Xh=Xk[:,hvg].toarray().astype(np.float32); hvg_names=[var_names[i] for i in hvg]
print(f"[hvg] selected 2000, matrix {Xh.shape}",flush=True)
# per-context control mean (basal)
basal={}
for ci,cn in enumerate(ccc):
    m=(cc_k==ci)&(gt_k==ctrl_code); basal[cn]=Xh[m].mean(0)
# per-(gene,context) targeting mean -> response = mean - basal[context]
rows=[]; resp=[]; bas=[]; genes=[]; ctxs=[]; effs=[]; ndes=[]; ncell_tot=[]
for ci,cn in enumerate(ccc):
    b=basal[cn]
    for gi,gn in enumerate(pgc):
        m=(cc_k==ci)&(gt_k==targ_code)&(pg_k==gi)
        n=int(m.sum())
        if n<1: continue
        pm=Xh[m].mean(0); r=pm-b
        resp.append(r); bas.append(b); genes.append(gn); ctxs.append(cn)
        effs.append(float(np.linalg.norm(r))); ndes.append(int((np.abs(r)>0.25).sum())); ncell_tot.append(float(nc_k[m].sum()))
resp=np.array(resp,dtype=np.float32); bas=np.array(bas,dtype=np.float32)
print(f"[agg] {resp.shape[0]} (gene,context) units across {len(set(genes))} genes x {len(ccc)} contexts",flush=True)
np.savez(os.path.join(OUT,"tcell_hvg_log1p.npz"), resp=resp, basal=bas, genes=np.array(genes), ctxs=np.array(ctxs),
         eff=np.array(effs,dtype=np.float32), n_de=np.array(ndes), n_cells=np.array(ncell_tot,dtype=np.float32),
         hvg_names=np.array(hvg_names), contexts=np.array(ccc))
print("[save] tcell_hvg_log1p.npz",flush=True); print("[done]",flush=True)
