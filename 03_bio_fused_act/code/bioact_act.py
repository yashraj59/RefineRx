"""bioact_act.py — GENUINE learned ACT on ARC's frozen pretrained STATE model.

The earlier halt head (raw pooled hidden state, default init) collapsed to E[N]=1 because the
aggregate per-layer loss is nearly flat, so the gate got no gradient and drifted to lambda_1 high.

Two fixes here, both faithful to ACT/PonderNet:
  (1) HALT FEATURES = the model's own layer-to-layer CONVERGENCE signals (Graves' canonical
      "am I still changing?" signal), computed from the FROZEN hidden states:
        - drep  = ||h_k - h_{k-1}|| / ||h_{k-1}||     representation movement
        - dpred = ||pred_dir_k - pred_dir_{k-1}||      predicted-direction change (target-free!)
        - hn    = ||h_k|| / sqrt(H)                    magnitude
        - frac  = k / n_layers                         hop fraction
      All available at inference WITHOUT the target -> no leakage.
  (2) lambda INIT SMALL (halt-head bias = -2 -> lambda~0.12): E[N] starts near max so the
      expected-task-loss gradient can pull halting EARLIER where the frozen model has already
      converged, instead of starting collapsed at layer 1.

Loss = expected per-layer task loss under the halting distribution p_t = lambda_t*Prod(1-lambda_j).
KL = 0 (per instruction). Optional tiny entropy bonus to keep the distribution from prematurely
one-hot-collapsing during early training (default 0 = pure PonderNet task loss). E[N]=Sum p_t*t is
a LEARNED quantity: the halt head decides, from convergence features, when to stop.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


class HaltHeadConv(nn.Module):
    def __init__(self, n_feat=4, hidden=32, bias_init=-2.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_feat, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.constant_(self.net[-1].bias, bias_init)     # lambda starts small -> E[N] starts high
    def forward(self, feats):                                # feats: (B, n_feat)
        return torch.sigmoid(self.net(feats)).squeeze(-1)


def ponder_step_probs(lam):
    B, K = lam.shape
    cont = torch.cumprod(torch.cat([torch.ones(B,1,device=lam.device), 1-lam[:,:-1]], 1), 1)
    p = lam * cont
    return p / p.sum(1, keepdim=True).clamp_min(1e-8)


class BioActACT(nn.Module):
    def __init__(self, arc_model, entropy_beta=0.0):
        super().__init__()
        self.arc = arc_model
        for p in self.arc.parameters(): p.requires_grad_(False)
        self.H = self.arc.transformer_backbone.config.hidden_size
        self.n_layers = self.arc.transformer_backbone.config.num_hidden_layers
        self.halt_head = HaltHeadConv(n_feat=4)
        self.entropy_beta = entropy_beta

    def _seq_ctrl(self, pert, basal_state, batch_idx):
        cc = self.arc.encode_basal_expression(basal_state)
        seq = self.arc.encode_perturbation(pert) + cc
        if getattr(self.arc,"batch_encoder",None) is not None and batch_idx is not None:
            seq = seq + self.arc.batch_encoder(batch_idx.long())
        return seq, cc

    def _genes(self, h, cc):
        latent = self.arc.project_out(h + cc) if self.arc.predict_residual else self.arc.project_out(h)
        return self.arc.gene_decoder(latent)

    def _run(self, pert, basal_state, batch_idx, target_dir, basal_genes):
        seq, cc = self._seq_ctrl(pert, basal_state, batch_idx)
        out = self.arc.transformer_backbone(inputs_embeds=seq, output_hidden_states=True)
        hs = out.hidden_states                       # tuple len n_layers+1 (embeddings + each layer)
        lam, step_cd = [], []
        prev_h = hs[0]; prev_pd = None
        for k in range(1, self.n_layers+1):
            h = hs[k]
            genes = self._genes(h, cc)
            pd = F.normalize(genes - basal_genes, dim=-1, eps=1e-8)          # (B,S,G)
            step_cd.append(1.0 - (pd * target_dir).sum(-1).mean(1))           # (B,) magnitude-free
            # convergence features (pooled over cell-sentence), all from frozen states / target-free
            drep = ((h - prev_h).norm(dim=-1) / prev_h.norm(dim=-1).clamp_min(1e-6)).mean(1)   # (B,)
            pd_pool = pd.mean(1)                                              # (B,G)
            if prev_pd is None:
                dpred = pd_pool.norm(dim=-1)
            else:
                dpred = (pd_pool - prev_pd).norm(dim=-1)
            hn = h.norm(dim=-1).mean(1) / (self.H**0.5)
            frac = torch.full_like(drep, k/self.n_layers)
            feats = torch.stack([drep, dpred, hn, frac], dim=1)              # (B,4)
            lam.append(self.halt_head(feats))
            prev_h = h; prev_pd = pd_pool
        lam = torch.stack(lam,1); step_cd = torch.stack(step_cd,1)           # (B,K)
        return lam, step_cd

    def forward(self, pert, basal_state, batch_idx, target_dir, basal_genes):
        lam, step_cd = self._run(pert, basal_state, batch_idx, target_dir, basal_genes)
        p = ponder_step_probs(lam)
        steps = torch.arange(1, self.n_layers+1, device=lam.device).float()
        E_N = (p*steps).sum(1)
        task = (p*step_cd).sum(1).mean()
        ent = -(p*p.clamp_min(1e-8).log()).sum(1).mean()
        total = task - self.entropy_beta*ent            # KL=0; entropy bonus optional (default 0)
        return dict(total=total, task=task, entropy=ent, E_N=E_N, p=p, lam=lam, step_cd=step_cd.detach())

    @torch.no_grad()
    def eval_EN(self, pert, basal_state, batch_idx, target_dir, basal_genes):
        lam, step_cd = self._run(pert, basal_state, batch_idx, target_dir, basal_genes)
        p = ponder_step_probs(lam)
        steps = torch.arange(1, self.n_layers+1, device=lam.device).float()
        return (p*steps).sum(1).cpu().numpy(), step_cd.cpu().numpy()
