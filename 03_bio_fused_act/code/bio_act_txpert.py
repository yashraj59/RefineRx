"""bio_act_txpert.py — biology-fused ACT grafted onto the EXISTING TxPert message-passing model.

Built on top of valence-labs/TxPert (faithful GATv2Conv message passing, weight-shared for adaptive
depth — see adaptive_txpert.py header). Two changes make it the biology-fused, magnitude-free ACT the
thesis needs:

  (1) HALT HEAD READS BIOLOGICAL CASCADE FEATURES, not the raw embedding. Each hop the head sees
      features derived from TxPert's OWN learned propagation on the real gene graph:
        - move_k   = ||h_k - h_{k-1}||  per node  (is the cascade still changing this node?)
        - norm_k   = ||h_k||            per node  (how strong is the propagated signal?)
        - hop_frac = k / K
      Biology enters through the halt head's inputs: these are properties of how the perturbation
      propagates on TxPert's message-passing graph.

  (2) MAGNITUDE-FREE PONDER LOSS. The per-hop task loss driving halting is cosine distance on the
      L2-normalized response DIRECTION (target - basal), not full-magnitude MSE. This removes the
      effect-size coupling that made earlier halting a magnitude proxy.

Diagnostic baked in (`per_hop_direction_loss`): does TxPert's LEARNED message passing produce a
per-hop cosine-distance curve that VARIES across perturbations? (The fixed-APPNP toy saturated in one
hop -> no signal. TxPert's expressive propagation is the real test.)
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from halting import ponder_step_probs, ponder_loss


class CascadeHaltHead(nn.Module):
    """Reads BIOLOGICAL cascade features (move, norm, hop_frac) -> per-node halt prob."""
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, feats):                       # feats: (n_nodes, 3)
        return torch.sigmoid(self.net(feats)).squeeze(-1)   # (n_nodes,)


class BioActTxPert(nn.Module):
    def __init__(self, n_genes, d=128, heads=4, n_hops=8, dropout=0.1,
                 lambda_prior=0.2, ponder_beta=0.05, ponder_weight=0.1):
        super().__init__()
        self.n_genes = n_genes; self.n_hops = n_hops
        self.lambda_prior = lambda_prior; self.ponder_beta = ponder_beta; self.ponder_weight = ponder_weight
        self.gene_emb = nn.Embedding(n_genes, d)
        self.conv = GATv2Conv(d, d, heads=heads, concat=False, dropout=dropout, add_self_loops=True, bias=True)
        self.norm = nn.LayerNorm(d)
        self.halt_head = CascadeHaltHead()          # reads cascade features, NOT raw embedding
        self.decoder = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, n_genes))
        self.basal = nn.Parameter(torch.zeros(n_genes))

    def propagate(self, edge_index):
        h = self.gene_emb.weight
        states, feats = [], []
        prev = h
        for k in range(self.n_hops):
            h = self.conv(h, edge_index); h = self.norm(F.silu(h))
            states.append(h)
            move = (h - prev).norm(dim=-1, keepdim=True)          # (n,1) cascade movement this hop
            nrm  = h.norm(dim=-1, keepdim=True)                   # (n,1) signal strength
            hopf = torch.full_like(move, (k + 1) / self.n_hops)   # (n,1) hop fraction
            feats.append(torch.cat([move, nrm, hopf], dim=1))     # (n,3) BIOLOGICAL cascade features
            prev = h
        lam = torch.stack([self.halt_head(f) for f in feats], dim=1)   # (n, K) LEARNED halt probs
        p = ponder_step_probs(lam)                                     # (n, K) halting distribution
        H = torch.stack(states, dim=1)                                 # (n, K, d)
        z_halted = (p.unsqueeze(-1) * H).sum(dim=1)
        steps = torch.arange(1, self.n_hops + 1, device=lam.device).float()
        E_N = (p * steps).sum(dim=1)
        return dict(states=states, lam=lam, p=p, z_halted=z_halted, E_N=E_N)

    def _dir(self, z):                              # response DIRECTION from an embedding
        return F.normalize(self.decoder(z), dim=-1, eps=1e-8)

    def forward(self, pert_idx, target, edge_index):
        prop = self.propagate(edge_index)
        z_g = prop["z_halted"][pert_idx]
        pred = self.basal.unsqueeze(0) + self.decoder(z_g)            # full endpoint (for recon)
        recon = F.mse_loss(pred, target)
        # MAGNITUDE-FREE per-hop task loss: cosine distance on response DIRECTION
        tgt_dir = F.normalize(target - self.basal.unsqueeze(0), dim=-1, eps=1e-8)   # (B, G)
        step_cd = []
        for k in range(self.n_hops):
            pred_dir_k = self._dir(prop["states"][k][pert_idx])       # (B, G)
            step_cd.append(1.0 - (pred_dir_k * tgt_dir).sum(-1))      # (B,) cosine distance
        step_cd = torch.stack(step_cd, dim=1)                        # (B, K) magnitude-free
        lam_g = prop["lam"][pert_idx]
        ploss, aux = ponder_loss(step_cd, lam_g, lambda_prior=self.lambda_prior, beta=self.ponder_beta)
        total = recon + self.ponder_weight * ploss
        return dict(pred=pred, recon=recon, ponder=ploss, total=total,
                    E_N_batch=aux["expected_n"], step_cd=step_cd.detach(), prop=prop)

    @torch.no_grad()
    def per_hop_direction_loss(self, pert_nodes, target_dirs, edge_index):
        """DIAGNOSTIC: per-hop cosine distance (no halting weighting) for each perturbation.
        Returns (P, K) — does the per-hop signal VARY across perts, or saturate at hop 1?"""
        prop = self.propagate(edge_index)
        out = []
        for k in range(self.n_hops):
            pd = self._dir(prop["states"][k][pert_nodes])            # (P, G)
            out.append(1.0 - (pd * target_dirs).sum(-1))            # (P,)
        return torch.stack(out, dim=1).cpu()                        # (P, K)

    @torch.no_grad()
    def signature(self, edge_index):
        prop = self.propagate(edge_index); p = prop["p"]
        ent = -(p.clamp_min(1e-9).log() * p).sum(dim=1)
        return dict(E_N=prop["E_N"].cpu(), halt_conf=(-ent).cpu())
