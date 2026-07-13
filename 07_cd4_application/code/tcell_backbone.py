"""tcell_backbone.py — fresh 8-layer transition backbone for T-cell HVG response + oracle-ACT halting.

Mirrors the K562 oracle-ACT setup but on a SELF-CONTAINED HVG backbone (no ARC weights — the HVG-first path):
  - a perturbation is encoded from its response's INPUT features? No — inputs are (perturbation identity via a
    learned per-gene embedding) + (basal context vector). We predict the control-relative response DIRECTION.
  - the backbone is a stack of pre-norm transformer-style residual MLP blocks over a length-(1+contexts) token
    sequence: [refinement_token, context_token]; each block is one "round"; per-layer exit decodes a response.
  - trained FIRST to convergence (all layers predict response), then FROZEN; the oracle ACT halt head is trained
    on top exactly as in K562 (refinement token + LayerNorm-unsaturated gate + oracle r* supervision).

Design symmetry with adaptive_state_refine.py: same halt head, same oracle_round / oracle_loss, same
sequential-hazard mixture, same halt_confidence (stop-distribution entropy, NOT any model confidence).
"""
from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F


class Block(nn.Module):
    """one refinement round: pre-norm attention-free residual MLP mixing token + context."""
    def __init__(self, H):
        super().__init__()
        self.ln1=nn.LayerNorm(H); self.mix=nn.Linear(2*H, H)     # mix refinement token with context token
        self.ln2=nn.LayerNorm(H); self.ff=nn.Sequential(nn.Linear(H,2*H), nn.GELU(), nn.Linear(2*H,H))
    def forward(self, tok, ctx):
        tok = tok + self.mix(torch.cat([self.ln1(tok), ctx], -1))
        tok = tok + self.ff(self.ln2(tok))
        return tok


class TcellBackbone(nn.Module):
    """fresh backbone: gene-embedding + context -> 8 residual rounds -> per-round HVG response decode."""
    def __init__(self, n_genes_pert, n_contexts, n_hvg, H=256, n_layers=8):
        super().__init__()
        self.H=H; self.n_layers=n_layers; self.n_hvg=n_hvg
        self.gene_emb=nn.Embedding(n_genes_pert, H)              # perturbation identity
        self.ctx_emb=nn.Embedding(n_contexts, H)                # culture condition
        self.basal_proj=nn.Linear(n_hvg, H)                     # context basal expression -> hidden
        self.refine0=nn.Parameter(torch.randn(H)*0.02)
        self.blocks=nn.ModuleList([Block(H) for _ in range(n_layers)])
        self.decode=nn.Sequential(nn.LayerNorm(H), nn.Linear(H, n_hvg))   # per-round response decode (shared)

    def rounds(self, gene_idx, ctx_idx, basal):
        B=gene_idx.shape[0]
        ctx = self.ctx_emb(ctx_idx) + self.basal_proj(basal)    # (B,H) context token
        tok = self.refine0[None].expand(B,-1) + self.gene_emb(gene_idx)   # (B,H) refinement/pert token
        preds=[]; toks=[]
        for blk in self.blocks:
            tok = blk(tok, ctx); toks.append(tok)
            preds.append(self.decode(tok))                      # (B,n_hvg) response prediction at this round
        return torch.stack(preds,1), torch.stack(toks,1)        # (B,R,n_hvg), (B,R,H)


class OracleACTHead(nn.Module):
    """oracle-ACT halt head over the frozen backbone rounds — identical machinery to the K562 graft."""
    def __init__(self, H, n_layers):
        super().__init__()
        self.n_layers=n_layers
        self.halt=nn.Sequential(nn.LayerNorm(H), nn.Linear(H,64), nn.SiLU(), nn.Linear(64,1))
        nn.init.constant_(self.halt[-1].bias, -2.0)
        self.err=nn.Sequential(nn.LayerNorm(H), nn.Linear(H,64), nn.SiLU(), nn.Linear(64,1))

    def stop_mass(self, toks):
        halt_logits=self.halt(toks).squeeze(-1)                 # (B,R)
        haz=torch.cat([torch.sigmoid(halt_logits[:,:-1]), torch.ones_like(halt_logits[:,-1:])],1)
        surv=torch.cumprod(torch.cat([torch.ones_like(haz[:,:1]), 1-haz[:,:-1]],1),1)
        return haz*surv, halt_logits

    @staticmethod
    def oracle_round(step_cd, tau=0.05):
        D1=step_cd[:, :1]; Dmin=step_cd.min(1,keepdim=True).values
        thr=Dmin+tau*(D1-Dmin); below=step_cd<=thr; R=step_cd.shape[1]
        idx=torch.arange(R,device=step_cd.device)[None,:].expand_as(below).clone(); idx[~below]=R
        return idx.min(1).values

    def loss(self, preds, toks, target_dir, basal_genes, tau=0.05, alpha=0.5, beta=1.0, gamma=0.1, delta=0.1, use_ponder=True):
        B,R,G=preds.shape
        pr_dir=F.normalize(preds-basal_genes[:,None], dim=-1, eps=1e-8)
        step_cd=1.0-(pr_dir*target_dir[:,None]).sum(-1)         # (B,R)
        sm,_=self.stop_mass(toks)
        mix=(sm[:,:,None]*preds.detach()).sum(1)
        mix_dir=F.normalize(mix-basal_genes,dim=-1,eps=1e-8)
        mix_cd=(1.0-(mix_dir*target_dir).sum(-1)).mean()
        deep=step_cd.mean()
        rstar=self.oracle_round(step_cd.detach(),tau=tau)
        q=sm.gather(1,rstar[:,None]).squeeze(1).clamp_min(1e-8)
        oracle_ce=(-q.log()).mean()
        rounds=torch.arange(1,R+1,device=preds.device).float()
        E_R=(sm*rounds).sum(1); ponder=(E_R/R).mean() if use_ponder else torch.zeros((),device=preds.device)
        ehat=F.softplus(self.err(toks).squeeze(-1)); err=F.huber_loss(ehat,step_cd.detach())
        total=mix_cd+alpha*deep+beta*oracle_ce+gamma*ponder+delta*err
        ent=-(sm.clamp_min(1e-8).log()*sm).sum(1); hc=1.0-ent/math.log(R)
        return dict(total=total,E_R=E_R,halt_confidence=hc,rstar=(rstar+1).float(),stop_mass=sm,step_cd=step_cd.detach())

    @torch.no_grad()
    def signature(self, preds, toks, target_dir, basal_genes):
        d=self.loss(preds,toks,target_dir,basal_genes)
        return d["E_R"].cpu().numpy(), d["halt_confidence"].cpu().numpy(), d["step_cd"].cpu().numpy()
