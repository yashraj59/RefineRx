"""bio_halt.py — biology-FUSED adaptive-depth model. The key departure from the earlier toys:
halting is NOT learned from reconstruction loss (a computational proxy a single endpoint can't pin).
Instead, refinement propagates the perturbation signal outward along a FIXED real regulatory/PPI graph
(STRING), and depth = how many propagation rounds until the biological perturbation FRONT saturates.

Because the graph is fixed biology, depth is pinned by where the perturbed gene sits in the network:
  - a hub with a dense, tightly-connected neighborhood saturates its reachable set fast  -> low depth
  - a gene whose downstream effect must traverse long / branching / feedback paths saturates slowly
    -> high depth
This cannot collapse to a training-time transient (the earlier failure) because the halting signal
comes from the FIXED graph geometry, not from a free predictor's convergence.

Mechanism per round t (message passing on the STRING adjacency A, row-normalized):
  h_{t+1} = (1-alpha)*A h_t + alpha*seed        # personalized-propagation (APPNP-style), seed = pert one-hot
  front_t = fraction of nodes with h_t > tau     # size of the activated biological front
  HALT when the front stops growing: front_t - front_{t-1} < eps_sat   (saturation of the cascade)
The model then predicts the response direction from the SETTLED field h_halt; the halting depth is a
pure function of graph geometry + the (learned) propagation coefficient, shared across perturbations.
"""
from __future__ import annotations
import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F


def build_adjacency(graph_json, device):
    G = json.load(open(graph_json))
    n = G["n_nodes"]; edges = G["edges"]; w = G["weights"]
    A = torch.zeros(n, n, device=device)
    for (i, j), s in zip(edges, w):
        A[i, j] = max(A[i, j].item(), s)          # keep strongest edge, undirected already doubled
    # add self loops, then row-normalize -> stochastic propagation operator
    A = A + torch.eye(n, device=device)
    A = A / A.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return A, G["genes"]


class BioHaltRefiner(nn.Module):
    """Biology-fused: propagation depth on a FIXED STRING graph sets the halting; a small decoder
    reads the settled field into a response direction. Halting depth is NOT free — it is determined
    by graph geometry via a learned-but-shared propagation coefficient alpha and threshold tau."""
    def __init__(self, A, n_genes_out, d=128, max_rounds=8, min_rounds=1,
                 eps_sat=0.01, learn_alpha=True):
        super().__init__()
        self.register_buffer("A", A)                 # (N, N) fixed biological operator
        self.N = A.size(0); self.max_rounds = max_rounds; self.min_rounds = min_rounds
        self.eps_sat = eps_sat
        # learned scalars (shared across ALL perturbations -> depth differences come from graph, not per-pert params)
        self.alpha_raw = nn.Parameter(torch.tensor(0.0)) if learn_alpha else None
        self._alpha_fixed = 0.15
        self.tau_raw = nn.Parameter(torch.tensor(-2.0))  # sigmoid -> ~0.12 activation threshold
        # readout: settled per-node field (N,) -> response direction over output genes
        self.node_emb = nn.Parameter(torch.randn(self.N, d) * 0.02)
        self.dec = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, n_genes_out))

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_raw) if self.alpha_raw is not None else torch.tensor(self._alpha_fixed, device=self.A.device)

    def propagate(self, pert_idx):
        """Personalized propagation from each perturbed node; returns settled field (B,N), depth (B,)."""
        B = pert_idx.size(0); dev = self.A.device
        seed = torch.zeros(B, self.N, device=dev)
        seed[torch.arange(B), pert_idx] = 1.0
        h = seed.clone()
        a = self.alpha; tau = torch.sigmoid(self.tau_raw)
        prev_front = (h > tau).float().mean(dim=1)          # (B,)
        depth = torch.full((B,), float(self.max_rounds), device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        fields = [h]
        for t in range(self.max_rounds):
            h = (1 - a) * (h @ self.A.t()) + a * seed        # APPNP-style propagation on FIXED graph
            fields.append(h)
            front = (h > tau).float().mean(dim=1)            # activated fraction (biological front size)
            grow = front - prev_front
            newly = (~done) & (grow < self.eps_sat) & ((t + 1) >= self.min_rounds)
            depth = torch.where(newly, torch.full_like(depth, float(t + 1)), depth)
            done = done | newly
            prev_front = front
        return h, depth

    def forward(self, pert_idx, target_dir=None):
        h, depth = self.propagate(pert_idx)                  # (B,N) settled field, (B,) depth
        # read settled field into direction: weight node embeddings by settled activation
        z = h @ self.node_emb                                # (B, d)
        pred = F.normalize(self.dec(z), dim=-1, eps=1e-8)
        out = dict(pred=pred, depth=depth, field=h)
        if target_dir is not None:
            out["recon"] = (1.0 - (pred * target_dir).sum(dim=1)).mean()
            out["total"] = out["recon"]
        return out

    @torch.no_grad()
    def signature(self, pert_idx):
        return self.propagate(pert_idx)[1].cpu()
