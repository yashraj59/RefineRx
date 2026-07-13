"""
scdiff_ponder_model.py — ScDiffPonder: adds PonderNet adaptive-depth halting to the
REAL scDiff ScDiff LightningModule, wiring the ponder loss into p_losses so it ADDS to
the native diffusion loss and trains end-to-end.

Grounded in OmicsML/scDiff scdiff/model.py (read 2026-07):
  * ScDiff.p_losses (L837)  - native diffusion loss; we call super() then add ponder.
  * ScDiff.prepare_noised_input (L809), q_sample (L782) - reused for the refine rounds.
  * self.model(x_ctxt, x_noised, t=..., conditions=...) -> (pred, mask); under
    parameterization="x0" pred is the endpoint estimate x0_hat directly (config
    eval_perturbation.yaml sets parameterization: x0). That is exactly what the halting
    head reads and what the per-round task loss scores.

Design (faithful to a single-endpoint diffusion model):
  standard training samples ONE random t and denoises once, so there is no refinement
  loop to halt over. We add an explicit K-round x0-refinement recurrence at a decreasing
  t-schedule (coarse->fine): predict x0_hat, score it vs target, emit a halt prob, feed
  x0_hat into the next round. PonderNet turns the round count into a learned,
  per-perturbation quantity. The native p_losses objective is left intact; ponder is added.
"""
from __future__ import annotations
import torch
from scdiff.model import ScDiff
from halting import HaltingHead, ponder_loss, ponder_step_probs


class ScDiffPonder(ScDiff):
    def __init__(self, *args,
                 ponder_weight: float = 0.1,
                 n_refine: int = 5,
                 refine_t_schedule=None,      # list[int] length n_refine, decreasing
                 lambda_prior: float = 0.2,
                 ponder_beta: float = 0.05,
                 halt_hidden: int = 256,
                 refine_mask: bool = False,   # False = all-context (no MAE masking) in refine rounds
                 **kwargs):
        super().__init__(*args, **kwargs)
        G = self.model.diffusion_model.in_dim
        self.halt_head = HaltingHead(dim=G, hidden=halt_hidden)
        self.ponder_weight = ponder_weight
        self.n_refine = n_refine
        self.lambda_prior = lambda_prior
        self.ponder_beta = ponder_beta
        self.refine_mask = refine_mask
        if refine_t_schedule is None:
            # geometric-ish decreasing schedule across the T=timesteps range
            T = self.num_timesteps
            refine_t_schedule = [int(T * f) for f in
                                 torch.linspace(0.8, 0.05, n_refine).tolist()]
        assert len(refine_t_schedule) == n_refine
        self.refine_t_schedule = refine_t_schedule

    # ---- the K-round refinement recurrence with per-round halting ----
    def ponder_refine(self, x_start, conditions, target,
                      input_gene_list=None, aug_graph=None):
        device = x_start.device
        B = x_start.shape[0]
        x_cur = x_start
        # all-context boolean mask (no MAE masking during refinement rounds).
        # NB scDiff.prepare_noised_input's `mask=False` branch builds a FLOAT zeros tensor
        # which then breaks on `~mask`; pass an explicit BOOL mask to avoid that repo bug.
        full_ctx = torch.zeros_like(x_start, dtype=torch.bool)
        step_losses, lambdas = [], []
        for k in range(self.n_refine):
            t = torch.full((B,), self.refine_t_schedule[k], device=device, dtype=torch.long)
            # noise the current endpoint estimate to level t (reuses scDiff machinery)
            x_ctxt, x_noised, mask = self.prepare_noised_input(x_cur, t, mask=full_ctx)
            pred, mask = self.model(x_ctxt, x_noised, t=t, conditions=conditions,
                                    input_gene_list=input_gene_list, aug_graph=aug_graph, mask=mask)
            x0_hat = pred                      # x0 parameterization: pred IS the endpoint estimate
            # per-example task loss of THIS round's endpoint prediction
            sl = self.get_loss(x0_hat, target, mask, mean=False)     # (B, G)
            step_losses.append(sl.mean(dim=1))                       # (B,)
            lambdas.append(self.halt_head(x0_hat))                   # (B,)
            x_cur = x0_hat.detach() if k < self.n_refine - 1 else x0_hat  # stabilize recurrence
        step_losses = torch.stack(step_losses, dim=1)                # (B, N)
        lambdas = torch.stack(lambdas, dim=1)                        # (B, N)
        loss, aux = ponder_loss(step_losses, lambdas,
                                lambda_prior=self.lambda_prior, beta=self.ponder_beta)
        return loss, aux

    # ---- override p_losses: native diffusion loss + ponder loss ----
    def p_losses(self, x_start, t, noise=None, pe_input=None, conditions=None,
                 input_gene_list=None, target_gene_list=None, text_embeddings=None,
                 aug_graph=None, target=None):
        # 1) intact native scDiff loss
        base_loss, loss_dict = super().p_losses(
            x_start, t, noise=noise, pe_input=pe_input, conditions=conditions,
            input_gene_list=input_gene_list, target_gene_list=target_gene_list,
            text_embeddings=text_embeddings, aug_graph=aug_graph, target=target)
        # 2) ponder loss over the refinement rounds
        tgt = target if target is not None else x_start   # x0 param: target defaults to input
        ponder_l, aux = self.ponder_refine(x_start, conditions, tgt,
                                           input_gene_list=input_gene_list, aug_graph=aug_graph)
        total = base_loss + self.ponder_weight * ponder_l
        pfx = 'train' if self.training else 'val'
        loss_dict[f'{pfx}/loss_ponder'] = ponder_l.detach()
        loss_dict[f'{pfx}/E_N'] = aux['expected_n'].mean().detach()
        loss_dict[f'{pfx}/halt_entropy'] = aux['halt_entropy'].mean().detach()
        loss_dict[f'{pfx}/loss'] = total.detach()
        return total, loss_dict

    # ---- per-perturbation signature readout (inference) ----
    @torch.no_grad()
    def signature(self, x_start, conditions, target, **kw):
        self.eval()
        _, aux = self.ponder_refine(x_start, conditions, target, **kw)
        # nonlinear-correction magnitude: change from round-1 to round-N endpoint
        return {
            "refinement_rounds": aux["expected_n"],       # E[N]
            "halt_confidence":  -aux["halt_entropy"],     # sharper halt = higher
            "p": aux["p"],
        }
