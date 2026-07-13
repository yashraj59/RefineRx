"""act_refiner_conv.py — MY OWN convergence-based adaptive-depth refiner (standalone; no rampert
/ scDiff / TxPert code). Design lessons taken from rampert HALTING_SETUP.md: max_rounds ceiling,
scale-invariance, and NO free learned halt gate (which collapses). But instead of a learned ACT gate,
depth is CONVERGENCE-based: the refiner is a genuine dynamical system and depth = #rounds until the
predicted unit direction STABILIZES. This makes 'depth' an intrinsic property of the dynamics, not a
free parameter the optimizer can trivially minimize.

  state_{t+1} = state_t + tanh(W state_t + pert_drive)     # shared residual operator (fixed point)
  dir_t       = normalize(readout(state_t))                # unit direction each round
  HALT at first t (>= min_rounds) where cos(dir_t, dir_{t-1}) > 1 - eps_converge
  E[N] = that halting round (soft version below for a differentiable training signal)

Training target = perturbation's unit response direction; loss = cosine distance of the FINAL
(converged) direction. The dynamics operator W is shared across rounds and across perturbations, so
convergence rate is the only thing that varies per perturbation.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


class ConvRefiner(nn.Module):
    def __init__(self, n_genes, n_pert, d=128, max_rounds=8, min_rounds=1,
                 eps_converge=0.02, step=0.5):
        super().__init__()
        self.n_genes=n_genes; self.max_rounds=max_rounds; self.min_rounds=min_rounds
        self.eps=eps_converge; self.step=step
        self.pert_emb = nn.Embedding(n_pert, d)
        self.pert_drive = nn.Sequential(nn.Linear(d,d), nn.SiLU(), nn.Linear(d,d))
        # SHARED residual dynamics operator (same weights every round -> genuine iteration)
        self.W = nn.Linear(d, d)
        self.readout = nn.Sequential(nn.Linear(d,d), nn.SiLU(), nn.Linear(d,n_genes))

    def _dir(self, state):
        return F.normalize(self.readout(state), dim=-1, eps=1e-8)

    def forward(self, pert_idx, target_dir=None):
        B=pert_idx.size(0); dev=pert_idx.device
        emb=self.pert_emb(pert_idx); drive=self.pert_drive(emb)
        state=torch.zeros_like(emb)
        dirs=[]; conv=[]                      # per-round direction + cos-to-previous
        depth=torch.full((B,),float(self.max_rounds),device=dev)   # default: never converged
        done=torch.zeros(B,dtype=torch.bool,device=dev)
        prev=None
        for t in range(self.max_rounds):
            state = state + self.step*torch.tanh(self.W(state)+drive)   # shared residual dynamics
            d_t = self._dir(state); dirs.append(d_t)
            if prev is not None:
                cos = F.cosine_similarity(d_t, prev, dim=1)             # (B,)
                conv.append(cos)
                newly = (~done) & (cos > (1-self.eps)) & ((t+1) >= self.min_rounds)
                depth = torch.where(newly, torch.full_like(depth,float(t+1)), depth)
                done = done | newly
            prev = d_t.detach()
        final_dir = dirs[-1]
        out=dict(final_dir=final_dir, depth=depth, dirs=dirs)
        if target_dir is not None:
            # train the DYNAMICS to reach the right direction; converged dir is what we score
            recon = (1.0 - (final_dir*target_dir).sum(dim=1)).mean()
            # also encourage each round to move toward target (deep supervision, magnitude-free)
            aux = torch.stack([(1.0-(d*target_dir).sum(1)) for d in dirs],dim=1).mean()
            out["recon"]=recon; out["aux"]=aux; out["total"]=recon+0.3*aux
        return out

    @torch.no_grad()
    def signature(self, pert_idx):
        return self.forward(pert_idx)["depth"].cpu()
