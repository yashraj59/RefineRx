"""
StateTransitionHalt — StateTransitionPerturbationModel + joint-calibration adaptive-depth
halting (user's §4.4 oracle r* + §4.5 joint calibration), fused into training_step.

Subclass keeps stock STATE 100% intact; halting is opt-in via model name `state_halt`.
Registered in the model registry so `state tx train model=state_halt ...` works.

New hparams (model.kwargs.*):
  n_refine_rounds      (int,   default 6)   # R refinement rounds
  halt_tau             (float, default 0.05)# oracle stopping tolerance
  halt_alpha           (float, default 0.5) # deep-supervision weight (all rounds)
  halt_beta            (float, default 0.1) # calibration -log q_{r*}
  halt_gamma           (float, default 0.01)# ponder E[R]/R
  halt_warmup_steps    (int,   default 2000)# ponder active only after this
  halt_magnitude_free  (bool,  default True)# D on L2-normalized direction (no effect-size)
  confidence_token     (bool,  set True)    # reuse STATE's confidence token as e_r head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .state_transition import StateTransitionPerturbationModel
from .halting_graft import (
    HaltHead, halting_distribution, expected_depth, oracle_stop_round, magnitude_free,
)


class StateTransitionHalt(StateTransitionPerturbationModel):
    def __init__(self, *args, **kwargs):
        # pull halting knobs BEFORE super() (super forwards **kwargs to hparams)
        self._halt_cfg = dict(
            n_refine_rounds=int(kwargs.get("n_refine_rounds", 6)),
            halt_tau=float(kwargs.get("halt_tau", 0.05)),
            halt_alpha=float(kwargs.get("halt_alpha", 0.5)),
            halt_beta=float(kwargs.get("halt_beta", 0.1)),
            halt_gamma=float(kwargs.get("halt_gamma", 0.01)),
            halt_warmup_steps=int(kwargs.get("halt_warmup_steps", 2000)),
            halt_magnitude_free=bool(kwargs.get("halt_magnitude_free", True)),
        )
        # force the confidence token ON (it is our per-round predicted-error head e_r)
        kwargs["confidence_token"] = True
        super().__init__(*args, **kwargs)
        # expose knobs as attributes
        self.n_refine_rounds = self._halt_cfg["n_refine_rounds"]
        self.halt_tau = self._halt_cfg["halt_tau"]
        self.halt_alpha = self._halt_cfg["halt_alpha"]
        self.halt_beta = self._halt_cfg["halt_beta"]
        self.halt_gamma = self._halt_cfg["halt_gamma"]
        self.halt_warmup_steps = self._halt_cfg["halt_warmup_steps"]
        self.halt_magnitude_free = self._halt_cfg["halt_magnitude_free"]
        # halt head: 4 scale-invariant trajectory features
        self.halt_head = HaltHead(n_feat=4, hidden=64)

    # ---- refinement recurrence (x0-style) ----
    def _refine_forward(self, batch, padded=True):
        R = self.n_refine_rounds
        preds, e_preds = [], []
        cur = dict(batch)
        for r in range(R):
            out = self.forward(cur, padded=padded)
            if isinstance(out, tuple):
                p, e = out
            else:
                p, e = out, None
            preds.append(p); e_preds.append(e)
            if r < R - 1:
                cur = dict(cur)
                cur["ctrl_cell_emb"] = p.reshape(-1, self.input_dim)
        return preds, e_preds

    # ---- fused training step ----
    def training_step(self, batch, batch_idx, padded=True):
        R = self.n_refine_rounds
        preds, e_preds = self._refine_forward(batch, padded=padded)
        target = batch["pert_cell_emb"]

        def _shape(t):
            return t.reshape(-1, self.cell_sentence_len, self.output_dim) if padded \
                   else t.reshape(1, -1, self.output_dim)
        target_s = _shape(target)

        D_per_set, D_scalar = [], []
        for p in preds:
            p_s = _shape(p)
            if self.halt_magnitude_free:
                p_use, t_use = magnitude_free(p_s), magnitude_free(target_s)
            else:
                p_use, t_use = p_s, target_s
            d = self._compute_distribution_loss(p_use, t_use)
            D_per_set.append(d); D_scalar.append(torch.nanmean(d))
        D_rounds = torch.stack(D_per_set, dim=1)          # [B,R]

        final_pred_s = _shape(preds[-1])
        main_loss = torch.nanmean(self._compute_distribution_loss(final_pred_s, target_s))
        self.log("train_loss", main_loss, prog_bar=True)

        deep_sup = torch.stack(D_scalar).mean()
        self.log("train/deep_sup", deep_sup)

        rstar = oracle_stop_round(D_rounds.detach(), tau=self.halt_tau)
        self.log("train/rstar_mean", rstar.float().mean())

        eps = 1e-6
        B = D_rounds.shape[0]
        D1 = D_rounds[:, :1].detach()
        hazards = []
        for r in range(R):
            Dr = D_rounds[:, r:r+1].detach()
            Dprev = D_rounds[:, max(r-1, 0):max(r-1, 0)+1].detach()
            f1 = (D1 - Dr) / (D1 + eps)
            f2 = torch.full_like(f1, (r + 1) / R)
            f3 = (Dprev - Dr) / (D1 + eps)
            if e_preds[r] is not None:
                er = e_preds[r].detach().reshape(B, 1); e1 = e_preds[0].detach().reshape(B, 1)
                f4 = er / (e1 + eps)
            else:
                f4 = torch.zeros_like(f1)
            hazards.append(self.halt_head.hazard(torch.cat([f1, f2, f3, f4], dim=1)))
        hazards = torch.stack(hazards, dim=1)
        p_halt = halting_distribution(hazards)
        EN = expected_depth(p_halt)
        self.log("train/EN_mean", EN.mean(), prog_bar=True)
        self.log("train/EN_std", EN.std())

        logp = torch.log(p_halt.clamp(min=1e-8))
        calib = F.nll_loss(logp, rstar)
        self.log("train/calib", calib)

        ponder = (EN / R).mean()
        self.log("train/ponder", ponder)

        conf_loss = torch.tensor(0.0, device=main_loss.device)
        if e_preds[0] is not None and self.confidence_weight > 0:
            for r in range(R):
                er = e_preds[r].reshape(B)
                tgt = D_per_set[r].detach()
                if self.confidence_target_scale is not None:
                    tgt = tgt * self.confidence_target_scale
                conf_loss = conf_loss + F.smooth_l1_loss(er, tgt.to(er.device))
            conf_loss = conf_loss / R
            self.log("train/conf_loss", conf_loss)

        after_warmup = float(self.global_step >= self.halt_warmup_steps)
        a, b, g = self.halt_alpha, self.halt_beta, self.halt_gamma
        d = self.confidence_weight if self.confidence_weight > 0 else 0.01
        total = (main_loss + a * deep_sup + after_warmup * b * calib
                 + after_warmup * g * ponder + d * conf_loss)

        if getattr(self, "gene_decoder", None) is not None and "pert_cell_counts" in batch:
            gene_targets = batch["pert_cell_counts"]
            latent_preds = preds[-1].detach() if self.detach_decoder else preds[-1]
            gd_pred = self.gene_decoder(latent_preds)
            gt = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_decoder.gene_dim()) if padded \
                 else gene_targets.reshape(1, -1, self.gene_decoder.gene_dim())
            dec_loss = self._compute_distribution_loss(gd_pred, gt).mean()
            self.log("decoder_loss", dec_loss)
            total = total + self.decoder_loss_weight * dec_loss

        self.log("train/total_loss", total)
        return total
