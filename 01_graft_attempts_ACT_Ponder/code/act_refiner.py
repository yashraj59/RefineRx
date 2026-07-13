"""
act_refiner.py — MY OWN adaptive-computation refiner, INSPIRED BY (not copied from) the rampert
HALTING_SETUP.md findings. This does not import or reuse rampert / HybridMemoryRAMLite; it is a
clean minimal recurrent refiner that encodes the three design lessons from that system's working
ACT-4 config:

  (1) ANTI-COLLAPSE CEILING: max_rounds=4 (their finding: 8 collapses to a uniform ~2; 4 stays
      adaptive because the halt head can specialize instead of spreading).
  (2) SCALE-INVARIANT HALT FEATURES: the halt head sees ONLY shape statistics of the unit response
      direction so far ((u).abs().mean, .std, kurtosis-like peakedness, participation ratio) — it is
      magnitude-BLIND by construction, not merely by the loss. This is the architectural version of
      "demand != effect-size proxy".
  (3) PURE GRAVES ACT (no PonderNet geometric KL): halting is trained implicitly through the mixture
      output against the task loss + a ponder-cost. Their finding: the PonderNet KL prior made depth
      effect-COUPLED and failed non-redundancy; plain ACT-4 passed.

Task: predict each perturbation's UNIT response direction (magnitude-free target), cosine-distance
loss. Depth E[N] = sum_t w_t*t over the ACT halting distribution.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


def scale_invariant_feats(u_dir):
    """Shape statistics of a (already unit-norm) direction vector — all magnitude-blind.
    u_dir: (B, G). Returns (B, 4): mean|u|, std|u|, peakedness (max|u|/mean|u|), participation ratio."""
    a = u_dir.abs()
    m = a.mean(dim=1, keepdim=True)
    s = a.std(dim=1, keepdim=True)
    peak = a.max(dim=1, keepdim=True).values / (m + 1e-8)
    # participation ratio (effective # of active genes) / G  -> in (0,1], scale-free
    pr = (u_dir.pow(2).sum(1, keepdim=True) ** 2) / (u_dir.pow(4).sum(1, keepdim=True) + 1e-12)
    pr = pr / u_dir.size(1)
    return torch.cat([m, s, peak, pr], dim=1)          # (B, 4)


class ACTRefiner(nn.Module):
    def __init__(self, n_genes, n_pert, d=128, max_rounds=4, min_rounds=2,
                 epsilon=0.01, ponder_tau=0.05):
        super().__init__()
        self.n_genes = n_genes; self.max_rounds = max_rounds
        self.min_rounds = min_rounds; self.epsilon = epsilon; self.ponder_tau = ponder_tau
        self.pert_emb = nn.Embedding(n_pert, d)
        self.enc = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        # shared recurrent update cell (GRU-style), applied up to max_rounds times
        self.cell = nn.GRUCell(d, d)
        self.to_dir = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, n_genes))
        # halt head sees ONLY scale-invariant shape feats (4) + a round-index scalar
        self.halt_head = nn.Sequential(nn.Linear(4 + 1, 64), nn.SiLU(), nn.Linear(64, 1))

    def _predict_dir(self, h):
        raw = self.to_dir(h)
        return F.normalize(raw, dim=-1, eps=1e-8)       # unit direction

    def forward(self, pert_idx, target_dir):
        """pert_idx:(B,), target_dir:(B,G) unit. Returns dict with mixture pred, ponder cost, E[N]."""
        B = pert_idx.size(0); dev = pert_idx.device
        h = self.enc(self.pert_emb(pert_idx))
        x = torch.zeros(B, self.enc[-1].out_features, device=dev)   # running input to cell
        halt_cum = torch.zeros(B, 1, device=dev)
        weights, dirs, step_losses, E_terms = [], [], [], []
        remainders = torch.ones(B, 1, device=dev)
        still = torch.ones(B, 1, device=dev)                        # 1 while not yet halted
        n_used = torch.zeros(B, 1, device=dev)
        for t in range(self.max_rounds):
            h = self.cell(x, h)
            d_hat = self._predict_dir(h)                            # (B,G)
            dirs.append(d_hat)
            # scale-invariant halt features of the CURRENT direction estimate
            si = scale_invariant_feats(d_hat)
            rfrac = torch.full((B,1), (t+1)/self.max_rounds, device=dev)
            p = torch.sigmoid(self.halt_head(torch.cat([si, rfrac], dim=1)))   # (B,1)
            # Graves ACT: enforce min_rounds by disallowing halt before it
            if (t+1) < self.min_rounds:
                p = torch.zeros_like(p)
            is_last = (t == self.max_rounds-1)
            # would this step push cumulative over 1-eps?
            over = (halt_cum + p) >= (1 - self.epsilon)
            w = torch.where(over | is_last, remainders * still, p * still)
            weights.append(w)
            step_losses.append(1.0 - (d_hat * target_dir).sum(dim=1, keepdim=True))  # cosine dist (B,1)
            n_used = n_used + still
            E_terms.append(w * (t+1))
            halt_cum = halt_cum + p * still
            remainders = remainders - p * still
            # update still-running mask
            newly_halted = (over | is_last).float() * still
            still = still * (1.0 - (over | is_last).float())
            # feed the running direction shape back as next input (detached magnitude)
            x = h
            if still.sum() == 0:
                break
        W = torch.cat(weights, dim=1)                     # (B, T)
        SL = torch.cat(step_losses, dim=1)                # (B, T)
        recon = (W * SL).sum(dim=1).mean()                # mixture cosine loss
        E_N = torch.cat(E_terms, dim=1).sum(dim=1)        # (B,) effective depth
        ponder = (n_used.squeeze(1) + remainders.squeeze(1))    # Graves ponder cost N+R
        total = recon + self.ponder_tau * ponder.mean()
        return dict(recon=recon, ponder=ponder.mean(), total=total, E_N=E_N,
                    W=W, dirs=dirs)

    @torch.no_grad()
    def signature(self, pert_idx):
        """Per-pert E[N] using a dummy target (halting doesn't depend on target, only on state shape)."""
        dummy = torch.zeros(pert_idx.size(0), self.n_genes, device=pert_idx.device)
        dummy[:, 0] = 1.0
        out = self.forward(pert_idx, dummy)
        return out["E_N"].cpu()
