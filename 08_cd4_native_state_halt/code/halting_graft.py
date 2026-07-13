"""
Joint-calibration halting graft for STATE StateTransitionPerturbationModel.

Fuses the user's §4.4 (oracle stopping round r*) + §4.5 (joint calibration) halting
INTO STATE's training_step, WITHOUT touching the native distributional loss.

Design (all per user's spec):
  * K-round x0-style refinement: run the transformer R times; each round r feeds the
    round-(r-1) PREDICTION back as the basal ("current estimate"), so the stack re-refines
    its own output. Round 1 == STATE's native single pass.
  * Per-round distributional loss D_r = D(pred_r, target)  (STATE energy/sinkhorn loss)
  * Oracle stopping round:  r* = min{ r : D_r <= D_min + tau*(D_1 - D_min) },  tau=0.05
  * Halt head q_r: sequential-hazard halting distribution over rounds. Inputs are
    magnitude-FREE biology/trajectory features + the confidence token's predicted error e_r.
  * Joint loss:
      L = D(Y_hat, Y)                                   # anchor: final round
        + alpha * (1/R) * sum_r D(pred_r, Y)            # deep supervision (all rounds)
        + beta  * ( -log q_{r*} )                       # calibrate stop to oracle round
        + gamma * E[R]/R                                # ponder / expected-depth (efficiency)
        + delta * Huber( e_r , stopgrad(D_r) )          # predicted-error regression
    ponder terms (beta,gamma) ACTIVE ONLY AFTER WARM-UP; KL prior = 0 (no geometric-prior KL).
    Per-round target is magnitude-free: D is on L2-normalized response directions
    (control-relative), so depth cannot collapse to an effect-size proxy.

E[N] (expected number of refinement rounds) = sum_r r * p_r  where p_r is the halting
distribution (sequential hazard). This is the per-perturbation depth signature.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HaltHead(nn.Module):
    """Sequential-hazard halt head. Emits hazard lambda_r in (0,1) per round from
    magnitude-free trajectory features + predicted-error e_r. Halting distribution:
        p_r = lambda_r * prod_{j<r}(1 - lambda_j),  with the last round forced to halt.
    """
    def __init__(self, n_feat: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def hazard(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, n_feat] -> lambda in (0,1): [B]
        return torch.sigmoid(self.net(feats)).squeeze(-1)


def halting_distribution(hazards: torch.Tensor) -> torch.Tensor:
    """hazards: [B, R] sequential hazards. Returns p: [B, R] proper halting distribution,
    last round absorbs remaining mass (forced halt)."""
    B, R = hazards.shape
    # survival prod_{j<r}(1 - lambda_j)
    one_minus = (1.0 - hazards).clamp(min=1e-6, max=1.0)
    surv = torch.cumprod(one_minus, dim=1)               # [B,R] = prod_{j<=r}
    surv_prev = torch.cat([torch.ones(B, 1, device=hazards.device), surv[:, :-1]], dim=1)
    p = hazards * surv_prev                                # p_r = lambda_r * prod_{j<r}(1-lambda_j)
    # force halt at last round: put all leftover mass there
    p_last = surv_prev[:, -1] * 1.0  # = prod_{j<R}(1-lambda_j) (halt for sure at R)
    p = p.clone()
    p[:, -1] = p_last
    # normalize (guards numerical drift)
    p = p / p.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return p


def expected_depth(p: torch.Tensor) -> torch.Tensor:
    """E[N] = sum_r r * p_r  (rounds are 1-indexed). p: [B,R] -> [B]."""
    B, R = p.shape
    rounds = torch.arange(1, R + 1, device=p.device, dtype=p.dtype).unsqueeze(0)  # [1,R]
    return (p * rounds).sum(dim=1)


def oracle_stop_round(D_rounds: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
    """D_rounds: [B, R] per-round distributional losses.
    r* = min{ r : D_r <= D_min + tau*(D_1 - D_min) }  (0-indexed return).
    Captures >= (1-tau) of attainable improvement over the first exit."""
    D1 = D_rounds[:, :1]                                  # [B,1]
    Dmin = D_rounds.min(dim=1, keepdim=True).values       # [B,1]
    thresh = Dmin + tau * (D1 - Dmin)                     # [B,1]
    meets = D_rounds <= thresh                            # [B,R] bool
    # first index where meets is True
    R = D_rounds.shape[1]
    idx = torch.arange(R, device=D_rounds.device).unsqueeze(0).expand_as(meets)
    big = torch.full_like(idx, R)
    rstar = torch.where(meets, idx, big).min(dim=1).values  # [B]
    return rstar.clamp(max=R - 1)


def magnitude_free(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize along feature dim so distributional loss is on DIRECTION, not magnitude.
    x: [..., D] -> unit vectors. This is the magnitude-free target (user's requirement)."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)
