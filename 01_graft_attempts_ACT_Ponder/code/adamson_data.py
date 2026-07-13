"""
adamson_data.py — adapt the Adamson 2016 UPR Perturb-seq h5ad to scDiff's batch dict.

Emits exactly the keys ScDiff.shared_step reads:
  batch["input"] : (B, G) normalized log1p expression (perturbed cell)
  batch["cond"]  : {"pert": (B,) long codes}   # conditioner vocabulary = 92 target_gene classes
  batch["pert_target"] : per-cell target (== input for x0 self-recon training)

Grouping: obs["target_gene"] (92 classes incl. 'control'). HVG-subset to keep the model
small for a first end-to-end run (var["highly_variable"] flags 2000 genes).
"""
import numpy as np, scanpy as sc, torch
from torch.utils.data import Dataset


class AdamsonPert(Dataset):
    def __init__(self, h5ad_path, n_hvg=2000, group_key="target_gene", normalize=True):
        ad = sc.read_h5ad(h5ad_path)
        # use raw counts layer if present, else X
        if "counts" in ad.layers:
            ad.X = ad.layers["counts"].copy()
        # HVG subset (use precomputed flag if available, else compute)
        if "highly_variable" in ad.var and ad.var["highly_variable"].sum() >= n_hvg:
            hv = ad.var["highly_variable"].values.astype(bool)
            ad = ad[:, hv].copy()
        else:
            sc.pp.highly_variable_genes(ad, n_top_genes=n_hvg, flavor="seurat_v3")
            ad = ad[:, ad.var["highly_variable"].values].copy()
        if normalize:
            sc.pp.normalize_total(ad, target_sum=1e4)
            sc.pp.log1p(ad)
        X = ad.X
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        self.input = torch.tensor(X, dtype=torch.float32)
        self.gene_names = list(ad.var_names)
        # condition codes over target_gene vocabulary
        grp = ad.obs[group_key].astype(str).values
        self.classes = sorted(np.unique(grp).tolist())
        self.cls_to_code = {c: i for i, c in enumerate(self.classes)}
        self.pert = torch.tensor([self.cls_to_code[g] for g in grp], dtype=torch.long)
        self.n_cond = len(self.classes)
        self.G = self.input.shape[1]

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, i):
        return {
            "input": self.input[i],
            "cond": {"pert": self.pert[i]},
            "pert_target": self.input[i],   # x0 self-recon target for training
        }


def collate(batch):
    out = {
        "input": torch.stack([b["input"] for b in batch]),
        "cond": {"pert": torch.stack([b["cond"]["pert"] for b in batch])},
        "pert_target": torch.stack([b["pert_target"] for b in batch]),
    }
    return out
