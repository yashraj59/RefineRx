"""bio_act.py — TRUE biology-fused ACT / PonderNet. Unlike bio_halt.py (which used a FIXED
saturation stopping rule and no learned halting), here the halting is a genuine LEARNED PonderNet
policy — a halt head emits per-round halt probabilities, forming a halting distribution, trained
through a ponder loss (expected task loss under the distribution + KL to a geometric prior). This IS
ACT.

The fusion of biology: the halt head does NOT see reconstruction loss (a computational proxy a single
endpoint cannot pin, which collapsed every earlier learned-halting attempt). It sees BIOLOGICAL
CASCADE FEATURES computed from propagation on a FIXED STRING graph:
  - front size            (fraction of network nodes currently activated)
  - front growth          (new activation this hop vs last)
  - new-territory fraction (nodes crossing threshold for the FIRST time this hop)
  - mean field level       (how strong the activated field is)
These are properties of where the perturbed gene sits in the fixed biological network. Because the
graph is fixed, the halt head is keying on biology, not on a free predictor's convergence — the
hypothesis under test is that THIS makes learned halting reproducible & non-redundant.

  h_{t+1} = (1-alpha)*A h_t + alpha*seed          # propagation on FIXED STRING graph
  bio_feat_t = [front, growth, new_territory, mean_field, t/T]
  lambda_t   = halt_head(bio_feat_t)              # LEARNED halt probability (PonderNet)
  p_t        = lambda_t * prod_{j<t}(1-lambda_j)  # halting distribution (last step absorbs remainder)
  E[N]       = sum_t p_t * t                      # LEARNED expected depth
  pred       = sum_t p_t * readout(h_t)           # halting-distribution-weighted response direction
  loss       = sum_t p_t * cosine_dist(pred_t, target) + beta * KL(p || Geometric(lambda_prior))
"""
from __future__ import annotations
import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F


def build_adjacency(graph_json, device):
    G = json.load(open(graph_json))
    n = G["n_nodes"]
    A = torch.zeros(n, n, device=device)
    for (i, j), s in zip(G["edges"], G["weights"]):
        A[i, j] = max(A[i, j].item(), s)
    A = A + torch.eye(n, device=device)
    A = A / A.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return A, G["node_genes"], G["pert_node_idx"]


def ponder_probs(lambdas):
    """lambdas: (B, T) per-round halt probs -> p: (B, T) halting distribution, last step absorbs rest."""
    B, T = lambdas.shape
    cont = torch.cumprod(torch.cat([torch.ones(B, 1, device=lambdas.device), 1 - lambdas[:, :-1]], 1), 1)
    p = lambdas * cont
    # renormalize so the distribution sums to 1 (final step takes the remainder)
    p = p + (cont[:, -1:] * (1 - lambdas[:, -1:])) * F.one_hot(torch.full((B,), T - 1, device=lambdas.device), T)
    return p


class BioACT(nn.Module):
    def __init__(self, A, n_genes_out, d=128, max_rounds=8, tau=0.05,
                 lambda_prior=0.3, ponder_beta=0.02, learn_alpha=True,
                 mask_seed_in_readout=False):
        super().__init__()
        self.register_buffer("A", A)
        self.N = A.size(0); self.max_rounds = max_rounds
        self.lambda_prior = lambda_prior; self.ponder_beta = ponder_beta
        # if True, zero the seed node in the field BEFORE the readout, so the response
        # prediction must come from the PROPAGATED cascade (not memorized seed->answer).
        # This is what gives the magnitude-free ponder loss a round-specific signal:
        # a perturbation whose response genes are distal is under-predicted at round 1,
        # so the halt head learns to wait -> depth tracks cascade reach, not the KL prior.
        self.mask_seed_in_readout = mask_seed_in_readout
        self.alpha_raw = nn.Parameter(torch.tensor(0.0)) if learn_alpha else None
        self._alpha_fixed = 0.15
        self.tau = tau
        # LEARNED halt head: 5 biological cascade features -> halt probability
        self.halt_head = nn.Sequential(nn.Linear(5, 64), nn.SiLU(), nn.Linear(64, 1))
        # readout: settled field -> response direction
        self.node_emb = nn.Parameter(torch.randn(self.N, d) * 0.02)
        self.dec = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, n_genes_out))

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_raw) if self.alpha_raw is not None else torch.tensor(self._alpha_fixed, device=self.A.device)

    def forward(self, pert_idx, target_dir=None):
        B = pert_idx.size(0); dev = self.A.device
        seed = torch.zeros(B, self.N, device=dev); seed[torch.arange(B), pert_idx] = 1.0
        seed_mask = 1.0 - seed if self.mask_seed_in_readout else torch.ones_like(seed)
        h = seed.clone(); a = self.alpha
        prev_on = (h > self.tau).float()
        lambdas, preds = [], []
        for t in range(self.max_rounds):
            h = (1 - a) * (h @ self.A.t()) + a * seed          # propagate on FIXED graph
            on = (h > self.tau).float()
            front = on.mean(1, keepdim=True)                    # (B,1) activated fraction
            growth = (on - prev_on).clamp(min=0).mean(1, keepdim=True)   # net new activation
            new_terr = ((on > 0) & (prev_on == 0)).float().mean(1, keepdim=True)  # first-time nodes
            mean_field = h.mean(1, keepdim=True)
            rfrac = torch.full((B, 1), (t + 1) / self.max_rounds, device=dev)
            bio_feat = torch.cat([front, growth, new_terr, mean_field, rfrac], dim=1)   # (B,5) BIOLOGY
            lambdas.append(torch.sigmoid(self.halt_head(bio_feat)))     # LEARNED halt prob
            h_read = h * seed_mask                              # optionally hide the seed node
            preds.append(F.normalize(self.dec(h_read @ self.node_emb), dim=-1, eps=1e-8))
            prev_on = on
        lam = torch.cat(lambdas, dim=1)                         # (B,T)
        p = ponder_probs(lam)                                  # (B,T) halting distribution
        steps = torch.arange(1, self.max_rounds + 1, device=dev).float()
        E_N = (p * steps).sum(1)                                # (B,) LEARNED expected depth
        P = torch.stack(preds, dim=1)                           # (B,T,G)
        pred_mix = F.normalize((p.unsqueeze(-1) * P).sum(1), dim=-1, eps=1e-8)
        out = dict(pred=pred_mix, E_N=E_N, p=p, lambdas=lam)
        if target_dir is not None:
            step_cd = 1.0 - (P * target_dir.unsqueeze(1)).sum(-1)      # (B,T) per-round cosine dist
            recon = (p * step_cd).sum(1).mean()                        # expected task loss under p
            # KL(p || Geometric(lambda_prior)) — PonderNet regularizer
            gp = self.lambda_prior * (1 - self.lambda_prior) ** torch.arange(self.max_rounds, device=dev).float()
            gp = gp / gp.sum()
            kl = (p * (torch.log(p + 1e-9) - torch.log(gp.unsqueeze(0) + 1e-9))).sum(1).mean()
            out["recon"] = recon; out["kl"] = kl
            out["total"] = recon + self.ponder_beta * kl
        return out

    @torch.no_grad()
    def signature(self, pert_idx):
        return self.forward(pert_idx)["E_N"].cpu()
