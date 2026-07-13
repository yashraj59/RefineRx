"""adaptive_state_refine.py — AdaptiveStateTransitionPerturbationModel.

Learned adaptive-depth ACT on ARC's frozen ST-SE-Replogle-k562, following the user's refinement-token +
sequential-hazard design. Fixes the failure mode of the pooled-feature halt head (which collapsed to a
global constant): a DEDICATED learnable REFINEMENT TOKEN is appended to the cell sentence, and its hidden
state at each exit layer feeds the halt head — so the halting representation is built per-perturbation by
the transformer itself, not pooled from cell features.

Sequential hazard: hazard_k = sigmoid(halt_logit_k) for k<K, forced 1 at the last round; stop_mass =
hazard * survival sums to 1 over rounds. The PREDICTION is the stop-mass-weighted MIXTURE of per-round
decoded predictions, so the halt distribution receives gradient directly from the task loss (mass flows to
the round that best predicts each perturbation). An auxiliary ERROR HEAD (softplus) predicts each round's
error.

STATE encode/decode preserved exactly:
  seq   = encode_perturbation(pert) + encode_basal_expression(basal)          # (B,S,H)
  decode(h) = gene_decoder(project_out(h + control_cells))                     # residual in hidden space
control_hidden == control_cells == encode_basal_expression(basal); basal kept for signature parity.

halt_confidence := 1 - entropy(stop_mass)/log(num_rounds)  — a property of the STOP DISTRIBUTION ONLY.
It is NOT STATE's ConfidenceToken.confidence_pred (which is inactive here) and must never be conflated
with it.
"""
from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F


class AdaptiveStateRefine(nn.Module):
    def __init__(self, arc_model, exit_layers=None, error_head=True):
        super().__init__()
        self.arc = arc_model
        for p in self.arc.parameters(): p.requires_grad_(False)
        self.H = self.arc.transformer_backbone.config.hidden_size
        self.n_layers = self.arc.transformer_backbone.config.num_hidden_layers
        self.exit_layers = list(exit_layers) if exit_layers is not None else list(range(1, self.n_layers+1))
        self.num_rounds = len(self.exit_layers)
        # dedicated learnable refinement token (the ACT probe), appended after the cell sentence
        self.refinement_token = nn.Parameter(torch.randn(self.H) * 0.02)
        # LayerNorm on the halt-head input is LOAD-BEARING: the frozen refinement-token hidden states have
        # large norm, so without normalization the pre-sigmoid halt logits have std ~100+ and sigmoid
        # saturates to exactly 0/1 -> binary hazards -> one-hot stop-mass -> constant-depth collapse.
        self.halt_head = nn.Sequential(nn.LayerNorm(self.H), nn.Linear(self.H, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.constant_(self.halt_head[-1].bias, -2.0)   # small hazard init -> E[rounds] starts high
        self.error_head = nn.Sequential(nn.LayerNorm(self.H), nn.Linear(self.H, 64), nn.SiLU(), nn.Linear(64, 1)) if error_head else None

    # ---- STATE-faithful encode / decode ----
    def _build_sequence_input(self, pert, basal_state, batch_idx):
        control_cells = self.arc.encode_basal_expression(basal_state)          # (B,S,H)
        seq = self.arc.encode_perturbation(pert) + control_cells
        if getattr(self.arc,"batch_encoder",None) is not None and batch_idx is not None:
            seq = seq + self.arc.batch_encoder(batch_idx.long())
        B,S,H = seq.shape
        rt = self.refinement_token.view(1,1,H).expand(B,1,H)
        seq_ref = torch.cat([seq, rt], dim=1)                                   # (B,S+1,H) token at index S
        self.cell_slice = slice(0,S); self.refinement_token_index = S
        return seq_ref, control_cells

    def _decode_hidden(self, cell_hidden, control_hidden, basal_expression):
        latent = self.arc.project_out(cell_hidden + control_hidden) if self.arc.predict_residual \
                 else self.arc.project_out(cell_hidden)
        return self.arc.gene_decoder(latent)                                    # (B,S,genes)

    # ---- adaptive forward ----
    def _run(self, pert, basal_state, batch_idx):
        seq_ref, control_cells = self._build_sequence_input(pert, basal_state, batch_idx)
        out = self.arc.transformer_backbone(inputs_embeds=seq_ref, output_hidden_states=True, return_dict=True)
        exit_hidden = [out.hidden_states[k] for k in self.exit_layers]
        round_preds, halt_logits, pred_errors = [], [], []
        for hidden in exit_hidden:
            cell_hidden = hidden[:, self.cell_slice, :]
            refinement_hidden = hidden[:, self.refinement_token_index, :]
            round_preds.append(self._decode_hidden(cell_hidden, control_cells, basal_state))
            halt_logits.append(self.halt_head(refinement_hidden).squeeze(-1))
            if self.error_head is not None:
                pred_errors.append(F.softplus(self.error_head(refinement_hidden).squeeze(-1)))
        round_preds = torch.stack(round_preds, dim=1)          # (B,R,S,G)
        halt_logits = torch.stack(halt_logits, dim=1)          # (B,R)
        pred_errors = torch.stack(pred_errors, dim=1) if pred_errors else None
        # sequential hazard; last round forced to stop
        hazard_before_last = torch.sigmoid(halt_logits[:, :-1])
        hazards = torch.cat([hazard_before_last, torch.ones_like(halt_logits[:, -1:])], dim=1)
        survival = torch.cumprod(torch.cat([torch.ones_like(hazards[:, :1]), 1.0 - hazards[:, :-1]], dim=1), dim=1)
        stop_mass = hazards * survival                          # (B,R) sums to 1
        return round_preds, stop_mass, pred_errors

    def forward(self, pert, basal_state, batch_idx, target_dir, basal_genes, entropy_floor=0.0):
        round_preds, stop_mass, pred_errors = self._run(pert, basal_state, batch_idx)
        B,R,S,G = round_preds.shape
        # per-round magnitude-free response-direction cosine distance (B,R)
        pr_dir = F.normalize(round_preds - basal_genes[:,None], dim=-1, eps=1e-8)         # (B,R,S,G)
        step_cd = 1.0 - (pr_dir * target_dir[:,None]).sum(-1).mean(2)                     # (B,R)
        # (1) mixture task loss trains the HALT HEAD's mass allocation: detach the per-round predictions
        #     so the ONLY free parameters shaping mix are stop_mass (halt head), not the frozen decoder.
        mix = (stop_mass[:,:,None,None] * round_preds.detach()).sum(1)                    # (B,S,G)
        mix_dir = F.normalize(mix - basal_genes, dim=-1, eps=1e-8)
        mix_cd = (1.0 - (mix_dir * target_dir).sum(-1).mean(1)).mean()
        # (2) expected per-round task loss under stop_mass (PonderNet-style)
        exp_cd = (stop_mass * step_cd.detach()).sum(1).mean()
        # (3) error-head aux
        err_loss = torch.zeros((), device=mix.device)
        if pred_errors is not None:
            err_loss = F.mse_loss(pred_errors, step_cd.detach())
        rounds = torch.arange(1, R+1, device=mix.device).float()
        expected_rounds = (stop_mass * rounds).sum(1)
        entropy = -(stop_mass.clamp_min(1e-8).log() * stop_mass).sum(1)                   # (B,)
        halt_confidence = 1.0 - entropy/math.log(R)                                       # stop-dist concentration
        # (4) entropy FLOOR: penalize the distribution collapsing below a target entropy (keeps the halt
        #     head exploring per-perturbation structure instead of snapping to one global round early).
        ent_pen = F.relu(entropy_floor - entropy).mean() if entropy_floor > 0 else torch.zeros((),device=mix.device)
        total = mix_cd + exp_cd + 0.1*err_loss + ent_pen
        return dict(total=total, mix_cd=mix_cd, exp_cd=exp_cd, err_loss=err_loss, ent_pen=ent_pen,
                    entropy=entropy.mean().detach(), expected_rounds=expected_rounds,
                    halt_confidence=halt_confidence, stop_mass=stop_mass, step_cd=step_cd.detach())

    @torch.no_grad()
    def eval_signature(self, pert, basal_state, batch_idx, target_dir, basal_genes):
        d = self.forward(pert, basal_state, batch_idx, target_dir, basal_genes)
        return (d["expected_rounds"].cpu().numpy(), d["halt_confidence"].cpu().numpy(),
                d["stop_mass"].cpu().numpy(), d["step_cd"].cpu().numpy())

    @staticmethod
    def oracle_round(step_cd, tau=0.05):
        """Stage-B oracle stopping round. step_cd: (B,R) per-round distributional loss D_r.
        r* = min{ r : D_r <= D_min + tau*(D_1 - D_min) } — first round capturing >=(1-tau) of the
        attainable improvement over the first exit. Returns 0-based index (B,)."""
        D1 = step_cd[:, :1]; Dmin = step_cd.min(1, keepdim=True).values
        thresh = Dmin + tau*(D1 - Dmin)                          # (B,1)
        below = step_cd <= thresh                                # (B,R) bool
        R = step_cd.shape[1]
        idx = torch.arange(R, device=step_cd.device)[None,:].expand_as(below).clone()
        idx[~below] = R                                          # mask non-qualifying to +inf
        return idx.min(1).values                                # (B,) first qualifying round (0-based)

    def oracle_loss(self, pert, basal_state, batch_idx, target_dir, basal_genes,
                    tau=0.05, alpha=0.5, beta=1.0, gamma=0.1, delta=0.1, use_ponder=True):
        """Stage-C jointly-calibrated stopping loss:
          L = D(mix) + alpha * mean_r D(Y_r,Y) + beta * (-log q_{r*}) + gamma * E[R]/R (ponder, post-warmup)
              + delta * Huber(e_hat_r, stopgrad(D_r))
        q_{r*} is the stop-mass at the ORACLE round r* (Stage B, tau). Ponder gated by use_ponder (off during
        warm-up). NOTE (anti-circularity guardrail): there is deliberately NO term coupling guides/perts that
        target the same gene to similar depth — that is left for a later regularizer so the guide-reproducibility
        test stays non-circular."""
        round_preds, stop_mass, pred_errors = self._run(pert, basal_state, batch_idx)
        B,R,S,G = round_preds.shape
        pr_dir = F.normalize(round_preds - basal_genes[:,None], dim=-1, eps=1e-8)
        step_cd = 1.0 - (pr_dir * target_dir[:,None]).sum(-1).mean(2)                 # (B,R) = D_r
        # final/mixture distributional loss
        mix = (stop_mass[:,:,None,None] * round_preds.detach()).sum(1)
        mix_dir = F.normalize(mix - basal_genes, dim=-1, eps=1e-8)
        mix_cd = (1.0 - (mix_dir * target_dir).sum(-1).mean(1)).mean()
        # alpha: deep per-round supervision (all rounds predict well)
        deep = step_cd.mean()
        # beta: oracle cross-entropy — pull stop-mass onto r* (Stage B target, computed on detached D_r)
        rstar = self.oracle_round(step_cd.detach(), tau=tau)                          # (B,)
        q_rstar = stop_mass.gather(1, rstar[:,None]).squeeze(1).clamp_min(1e-8)       # (B,)
        oracle_ce = (-q_rstar.log()).mean()
        # gamma: ponder cost E[R]/R (encourage earlier halting) — only after warm-up
        rounds = torch.arange(1, R+1, device=mix.device).float()
        E_R = (stop_mass * rounds).sum(1)
        ponder = (E_R/R).mean() if use_ponder else torch.zeros((),device=mix.device)
        # delta: error head predicts each round's own D_r (Huber, stop-grad target)
        err = torch.zeros((), device=mix.device)
        if pred_errors is not None:
            err = F.huber_loss(pred_errors, step_cd.detach())
        total = mix_cd + alpha*deep + beta*oracle_ce + gamma*ponder + delta*err
        entropy = -(stop_mass.clamp_min(1e-8).log()*stop_mass).sum(1)
        halt_confidence = 1.0 - entropy/math.log(R)
        return dict(total=total, mix_cd=mix_cd, deep=deep, oracle_ce=oracle_ce, ponder=ponder, err=err,
                    expected_rounds=E_R, halt_confidence=halt_confidence, rstar=(rstar+1).float(),
                    stop_mass=stop_mass, step_cd=step_cd.detach())
