"""bioact_state.py — adaptive early-EXIT-DEPTH ACT on ARC's PRETRAINED State Transition model.

Host = arcinstitute/ST-SE-Replogle (fewshot/k562): real StateTransitionPerturbationModel, an 8-layer
bidirectional-Llama perturbation transformer in SE-600M embedding space, loaded from ARC pretrained
weights (0 missing / 0 unexpected). Predicts GENE EXPRESSION: the bundled gene_decoder maps the
transformer output (2058-d embedding space) -> 2000 HVG genes.

Faithful forward (state_transition.py):
   seq = encode_pert(onehot) + encode_basal(X_state) + batch_emb   # (B,S,328)
   hidden = Llama(seq)                                             # (B,S,328)
   out = project_out(hidden)(->2058) + basal ; genes = gene_decoder(out)(->2000)

GRAFT: run Llama with output_hidden_states=True; at EACH of the 8 layers apply project_out + residual
+ gene_decoder -> a per-layer GENE-EXPRESSION prediction. A PonderNet halt head reads each pooled layer
state -> halting distribution over EXIT DEPTH -> E[N] = expected exit layer. All ARC weights FROZEN; only
the halt head trains, so E[N] reads the pretrained model's own layerwise convergence.

Magnitude-free: per-layer ponder task loss = cosine distance on the L2-normalized response DIRECTION
(pred_genes - basal_genes), so halting cannot key on raw effect size.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


class HaltHead(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
    def forward(self, h): return torch.sigmoid(self.net(h)).squeeze(-1)


def ponder_step_probs(lam):
    B, K = lam.shape
    cont = torch.cumprod(torch.cat([torch.ones(B,1,device=lam.device), 1-lam[:,:-1]], 1), 1)
    p = lam * cont
    return p / p.sum(1, keepdim=True).clamp_min(1e-8)


class BioActState(nn.Module):
    def __init__(self, arc_model, basal_genes_fn, lambda_prior=0.25, ponder_beta=0.02):
        """arc_model: loaded StateTransitionPerturbationModel (FROZEN).
        basal_genes_fn: maps basal X_state (B,S,input_dim) -> basal gene expr (B,S,2000) via gene_decoder
                        of the basal, so the residual is computed in gene space consistently."""
        super().__init__()
        self.arc = arc_model
        for p in self.arc.parameters(): p.requires_grad_(False)
        H = self.arc.transformer_backbone.config.hidden_size
        self.n_layers = self.arc.transformer_backbone.config.num_hidden_layers
        self.halt_head = HaltHead(H)
        self.lambda_prior = lambda_prior; self.ponder_beta = ponder_beta

    def _genes_from_hidden(self, h, control_cells):
        # FAITHFUL to StateTransitionPerturbationModel.forward + predict_step (SE variant, gene_decoder):
        #   predict_residual -> latent = project_out(res_pred + control_cells)  [residual added in HIDDEN
        #   space, 328-d, BEFORE project_out]; no relu (gene_decoder present); genes = gene_decoder(latent).
        # control_cells = encode_basal_expression(basal) -> (B,S,328), same tensor used to build seq.
        if self.arc.predict_residual:
            latent = self.arc.project_out(h + control_cells)     # (B,S,2058)
        else:
            latent = self.arc.project_out(h)
        genes = self.arc.gene_decoder(latent)                    # (B,S,2000)
        return genes

    def _seq_and_ctrl(self, pert, basal_state, batch_idx):
        control_cells = self.arc.encode_basal_expression(basal_state)      # (B,S,328) hidden
        seq = self.arc.encode_perturbation(pert) + control_cells
        if getattr(self.arc, "batch_encoder", None) is not None and batch_idx is not None:
            seq = seq + self.arc.batch_encoder(batch_idx.long())
        return seq, control_cells

    def forward(self, pert, basal_state, batch_idx, target_dir, basal_genes):
        seq, control_cells = self._seq_and_ctrl(pert, basal_state, batch_idx)
        out = self.arc.transformer_backbone(inputs_embeds=seq, output_hidden_states=True)
        lam, step_cd = [], []
        for h in out.hidden_states[1:]:
            genes = self._genes_from_hidden(h, control_cells)
            pred_dir = F.normalize(genes - basal_genes, dim=-1, eps=1e-8)
            step_cd.append(1.0 - (pred_dir * target_dir).sum(-1).mean(1))   # (B,) magnitude-free
            lam.append(self.halt_head(h.mean(1)))
        lam = torch.stack(lam, 1); step_cd = torch.stack(step_cd, 1)
        p = ponder_step_probs(lam)
        steps = torch.arange(1, self.n_layers+1, device=lam.device).float()
        E_N = (p * steps).sum(1)
        task = (p * step_cd).sum(1).mean()               # expected per-layer task loss under halting dist
        # NO KL PRIOR: pure Graves-ACT objective. With ponder_beta=0 the halt distribution is free to
        # concentrate on whichever layer best predicts each perturbation's response direction — E[N]
        # becomes an unconstrained per-perturbation readout of best-exit-depth, not pinned to a prior mean.
        if self.ponder_beta and self.ponder_beta > 0:
            kfac = torch.arange(self.n_layers, device=lam.device).float()
            prior = self.lambda_prior*(1-self.lambda_prior)**kfac; prior = prior/prior.sum()
            kl = (p*(p.clamp_min(1e-8).log()-prior.clamp_min(1e-8).log().unsqueeze(0))).sum(1).mean()
        else:
            kl = torch.zeros((), device=lam.device)      # KL = 0
        total = task + self.ponder_beta*kl
        return dict(total=total, task=task, kl=kl, E_N=E_N, p=p, lam=lam, step_cd=step_cd.detach())

    @torch.no_grad()
    def per_layer_direction_loss(self, pert, basal_state, batch_idx, target_dir, basal_genes):
        seq, control_cells = self._seq_and_ctrl(pert, basal_state, batch_idx)
        out = self.arc.transformer_backbone(inputs_embeds=seq, output_hidden_states=True)
        cds = []
        for h in out.hidden_states[1:]:
            genes = self._genes_from_hidden(h, control_cells)
            pd = F.normalize(genes - basal_genes, dim=-1, eps=1e-8)
            cds.append(1.0 - (pd*target_dir).sum(-1).mean(1))
        return torch.stack(cds, 1).cpu()

    @torch.no_grad()
    def final_genes(self, pert, basal_state, batch_idx):
        # faithful full-model prediction (last layer) for a faithfulness check vs ARC's adata_pred
        seq, control_cells = self._seq_and_ctrl(pert, basal_state, batch_idx)
        out = self.arc.transformer_backbone(inputs_embeds=seq, output_hidden_states=True)
        return self._genes_from_hidden(out.hidden_states[-1], control_cells)
