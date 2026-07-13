"""
halting.py — ACT (Graves 2016) and PonderNet (Banino 2021) halting heads + losses.

Self-contained and framework-agnostic: depends only on torch. The two integration
sketches (scdiff_ponder.py, txpert_ponder.py) import from here.

Two mechanisms:
  * PonderNet  — Bernoulli halt node lambda_n per step; halting distribution
                 p_n = lambda_n * prod_{j<n}(1 - lambda_j); loss is the expected
                 per-step task loss under p plus a KL to a Geometric(lambda_p) prior.
                 Better-behaved gradients; the geometric prior controls "laziness".
  * ACT (Graves) — cumulative halt sum until >= 1 - eps; last step takes the
                 "remainder"; ponder cost = N + remainder. Kept for completeness.

Both expose the SAME readout you need for the thesis: a per-example expected number
of refinement steps E[N] (the "refinement rounds" signature component) and a halt
distribution whose entropy is the "halt confidence" component.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class HaltingHead(nn.Module):
    """Reads a running state h_t (B, D) and emits a per-step halt probability lambda_t in (0,1).

    Mirror of TxPert's GatedCombiner.gate form (Linear -> Sigmoid) so the graft is idiomatic.
    Optionally conditions on a step/time embedding so 'how far along' is visible to the head.
    """
    def __init__(self, dim: int, hidden: int | None = None, time_dim: int = 0):
        super().__init__()
        hidden = hidden or dim
        self.time_dim = time_dim
        self.net = nn.Sequential(
            nn.Linear(dim + time_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor | None = None) -> torch.Tensor:
        if self.time_dim and t_emb is not None:
            h = torch.cat([h, t_emb], dim=-1)
        return torch.sigmoid(self.net(h)).squeeze(-1)  # (B,)


def ponder_step_probs(lambdas: torch.Tensor) -> torch.Tensor:
    """Convert per-step halt probs lambda_n (B, N) into the PonderNet halting
    distribution p_n (B, N), with the last step absorbing all remaining mass so
    sum_n p_n == 1 exactly.

        p_n = lambda_n * prod_{j<n} (1 - lambda_j)
    """
    B, N = lambdas.shape
    one_minus = (1.0 - lambdas).clamp(min=1e-8)
    # cumulative product of (1 - lambda_j) STRICTLY before step n
    cum = torch.cumprod(one_minus, dim=1)
    prior_not_halted = torch.cat([torch.ones(B, 1, device=lambdas.device), cum[:, :-1]], dim=1)
    p = lambdas * prior_not_halted
    # absorb leftover mass into the final step (guarantees a proper distribution)
    p[:, -1] = 1.0 - p[:, :-1].sum(dim=1)
    return p.clamp(min=0.0)


def ponder_loss(step_losses: torch.Tensor, lambdas: torch.Tensor,
                lambda_prior: float = 0.2, beta: float = 0.01):
    """PonderNet loss.

    Args:
        step_losses: (B, N) task loss of the readout produced AT each step n
                     (e.g. MSE between the step-n endpoint prediction and the target).
        lambdas:     (B, N) per-step halt probabilities from HaltingHead.
        lambda_prior: geometric prior halting rate lambda_p in (0,1). Smaller => the
                      prior prefers MORE steps (lazier halting).
        beta:        weight on the KL regularizer.

    Returns: (loss, aux) where aux has p (B,N), expected_n (B,), halt_entropy (B,).
    """
    p = ponder_step_probs(lambdas)                    # (B, N)
    L_rec = (p * step_losses).sum(dim=1)              # expected task loss under p

    B, N = lambdas.shape
    steps = torch.arange(1, N + 1, device=lambdas.device).float()
    # geometric prior p_G(n) = (1-lp)^{n-1} * lp, truncated+renormalised to N steps
    pg = (1 - lambda_prior) ** (steps - 1) * lambda_prior
    pg = (pg / pg.sum()).unsqueeze(0).expand(B, N)
    L_kl = (p * (torch.log(p.clamp(min=1e-8)) - torch.log(pg.clamp(min=1e-8)))).sum(dim=1)

    loss = (L_rec + beta * L_kl).mean()
    expected_n = (p * steps.unsqueeze(0)).sum(dim=1)             # E[N] per example
    halt_entropy = -(p * torch.log(p.clamp(min=1e-8))).sum(dim=1)
    return loss, {"p": p, "expected_n": expected_n, "halt_entropy": halt_entropy,
                  "rec": L_rec.mean().detach(), "kl": L_kl.mean().detach()}


def act_halting(lambdas: torch.Tensor, eps: float = 0.01):
    """Graves ACT: cumulative-sum halting. Returns (weights (B,N), ponder_cost (B,), N_used (B,)).

    weights are the per-step mixing weights (halt prob for steps before the halt step,
    remainder at the halt step, 0 after). ponder_cost = N_used + remainder.
    """
    B, N = lambdas.shape
    cum = torch.cumsum(lambdas, dim=1)
    still_running = (cum < 1 - eps).float()                      # 1 while not yet halted
    N_used = still_running.sum(dim=1) + 1                        # index of halt step (1-based)
    weights = lambdas * still_running
    # remainder assigned to the halt step so weights sum to 1
    R = 1.0 - (weights * still_running).sum(dim=1)
    idx = N_used.long().clamp(max=N) - 1
    weights[torch.arange(B), idx] = R
    ponder_cost = N_used + R
    return weights, ponder_cost, N_used
