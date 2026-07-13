"""
adaptive_refiner_norm.py — adaptive message-passing depth, but the halting target is the
UNIT-NORMALIZED response DIRECTION, not full-magnitude expression.

WHY (thesis-critical fix): the previous graft's per-step loss was ||pred_k - target||^2 on the
full-magnitude response. That loss scales with response magnitude, so a halting head that halts
when loss is low MECHANICALLY ties E[N] to effect size. That is a self-inflicted confound, not a
property of the data. To ask the thesis's real question — is there a magnitude-INDEPENDENT
per-perturbation complexity signal — the refinement target must be magnitude-free.

Here: predict the L2-normalized response direction; per-step loss = 1 - cos(pred_k, target_dir)
(cosine distance, bounded [0,2], scale-invariant). E[hops] then reflects how many message-passing
rounds are needed to get the DIRECTION right, decoupled from how big the response is.

Message passing is TxPert's GATv2Conv applied K times (weight-shared, dim-preserving); halting
head over node state reuses TxPert's GatedCombiner sigmoid-gate form.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from halting import HaltingHead, ponder_step_probs, ponder_loss


class AdaptiveRefinerNorm(nn.Module):
    def __init__(self, n_genes, d=128, heads=4, n_hops=8, dropout=0.1,
                 lambda_prior=0.2, ponder_beta=0.05, ponder_weight=0.1):
        super().__init__()
        self.n_genes = n_genes; self.n_hops = n_hops
        self.lambda_prior = lambda_prior; self.ponder_beta = ponder_beta
        self.ponder_weight = ponder_weight
        self.gene_emb = nn.Embedding(n_genes, d)
        self.conv = GATv2Conv(d, d, heads=heads, concat=False, dropout=dropout,
                              add_self_loops=True, bias=True)
        self.norm = nn.LayerNorm(d)
        self.halt_head = HaltingHead(dim=d, hidden=d)
        # decoder: node embedding -> gene-space DIRECTION (unnormalized; we normalize after)
        self.decoder = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, n_genes))

    def propagate(self, edge_index):
        h = self.gene_emb.weight
        states, lambdas = [], []
        for k in range(self.n_hops):
            h = self.conv(h, edge_index)
            h = self.norm(F.silu(h))
            states.append(h); lambdas.append(self.halt_head(h))
        lam = torch.stack(lambdas, dim=1)
        p = ponder_step_probs(lam)
        steps = torch.arange(1, self.n_hops + 1, device=lam.device).float()
        E_N = (p * steps).sum(dim=1)
        return dict(states=states, lam=lam, p=p, E_N=E_N)

    @staticmethod
    def _cos_dist(pred, target_unit):
        # target_unit is already L2-normalized; normalize pred, return 1 - cos  (per row)
        pred_u = F.normalize(pred, dim=-1, eps=1e-8)
        return 1.0 - (pred_u * target_unit).sum(dim=-1)   # (B,)

    def forward(self, pert_idx, target_dir, edge_index):
        """pert_idx: (B,) perturbed-gene node index; target_dir: (B, n_genes) UNIT response dir."""
        prop = self.propagate(edge_index)
        step_losses = []
        for k in range(self.n_hops):
            pred_k = self.decoder(prop["states"][k][pert_idx])      # (B, n_genes) direction
            step_losses.append(self._cos_dist(pred_k, target_dir))  # (B,) magnitude-free
        step_losses = torch.stack(step_losses, dim=1)               # (B, K)
        lam_g = prop["lam"][pert_idx]
        ploss, aux = ponder_loss(step_losses, lam_g,
                                 lambda_prior=self.lambda_prior, beta=self.ponder_beta)
        recon = step_losses[:, -1].mean()          # final-step cosine distance (task quality)
        total = recon + self.ponder_weight * ploss
        return dict(recon=recon, ponder=ploss, total=total,
                    E_N_batch=aux["expected_n"], prop=prop)

    @torch.no_grad()
    def signature(self, edge_index):
        prop = self.propagate(edge_index)
        p = prop["p"]
        ent = -(p.clamp_min(1e-9).log() * p).sum(dim=1)
        return dict(E_N=prop["E_N"].cpu(), halt_conf=(-ent).cpu())
