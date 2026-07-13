
import h5py, numpy as np, scipy.sparse as sp, json, os
from itertools import combinations
F="/workspace/halt/data/tcell/GWCD4i.pseudobulk_merged.h5ad"
HVGF="/workspace/halt/data/tcell/fixed_hvg_names.npy"
f=h5py.File(F,"r"); o=f["obs"]
def cat(col):
    g=o[col]; return g["codes"][:], [x.decode() if isinstance(x,bytes) else x for x in g["categories"][:]]
gt,gtc=cat("guide_type"); cc,ccc=cat("culture_condition"); dn,dnc=cat("donor_id"); pg,pgc=cat("perturbed_gene_name")
keep=o["keep_for_DE"][:].astype(bool)
var_names=[v.decode() if isinstance(v,bytes) else v for v in f["var"]["gene_name"][:]]
hvg_names=np.load(HVGF,allow_pickle=True); vmap={g:i for i,g in enumerate(var_names)}
hvg_idx=np.array([vmap[g] for g in hvg_names])
sh=f["X"].attrs["shape"]
X=sp.csr_matrix((f["X"]["data"][:],f["X"]["indices"][:],f["X"]["indptr"][:]),shape=(int(sh[0]),int(sh[1])))
X=X.astype(np.float32); np.log1p(X.data,out=X.data)
Xh=X[:,hvg_idx].toarray().astype(np.float32); del X
targ=gtc.index("targeting"); ctrl=gtc.index("non-targeting")
# per (donor,condition): basal = mean control; per gene response = mean(targ) - basal
# Build dict: (cond, donor) -> {gene: response_vec}
resp={}   # resp[(ci,di)][gene] = vec
for ci,cn in enumerate(ccc):
    for di,d in enumerate(dnc):
        cm = keep & (cc==ci) & (dn==di)
        b = Xh[cm & (gt==ctrl)].mean(0)
        gg={}
        for gi in range(len(pgc)):
            m = cm & (gt==targ) & (pg==gi)
            if m.sum()<1: continue
            gg[gi]=Xh[m].mean(0)-b
        resp[(ci,di)]=gg
# For each condition, align genes present in ALL 4 donors; build magnitude table and direction cosines
out={"conditions":{}}
for ci,cn in enumerate(ccc):
    donors=list(range(len(dnc)))
    shared=set(resp[(ci,0)].keys())
    for di in donors[1:]: shared&=set(resp[(ci,di)].keys())
    shared=sorted(shared)
    # magnitude table: (n_shared, 4)
    mag=np.array([[np.linalg.norm(resp[(ci,di)][g]) for di in donors] for g in shared])  # (G,4)
    # cross-donor magnitude Spearman
    from scipy.stats import spearmanr
    magrho=[spearmanr(mag[:,i],mag[:,j])[0] for i,j in combinations(donors,2)]
    # cross-donor response-direction cosine, per unit, averaged
    cos_pairs={}
    for i,j in combinations(donors,2):
        cs=[]
        for k,g in enumerate(shared):
            a=resp[(ci,i)][g]; b=resp[(ci,j)][g]
            na=np.linalg.norm(a); nb=np.linalg.norm(b)
            if na>1e-8 and nb>1e-8: cs.append(float(a@b/(na*nb)))
        cos_pairs[f"{dnc[i]}|{dnc[j]}"]=float(np.mean(cs))
    out["conditions"][cn]={"n_shared_genes":len(shared),
        "xdonor_magnitude_spearman_mean":float(np.mean(magrho)),
        "xdonor_magnitude_spearman_pairs":[round(r,3) for r in magrho],
        "xdonor_direction_cosine_mean":float(np.mean(list(cos_pairs.values()))),
        "xdonor_direction_cosine_pairs":{k:round(v,3) for k,v in cos_pairs.items()}}
    print(f"[{cn}] n={len(shared)} xdonor_mag_rho={np.mean(magrho):.3f} xdonor_dir_cos={np.mean(list(cos_pairs.values())):.3f}",flush=True)
f.close()
json.dump(out, open("/workspace/halt/actrun_state/xdonor_rawresp.json","w"), indent=1)
print("[done]",flush=True)
