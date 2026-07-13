"""
adaptive_txpert.py — adaptive message-passing DEPTH over a gene graph, PonderNet halting.

Faithful to valence-labs/TxPert (gspp/models/pert_models/multi_graph.py, read 2026-07):
  * message passing = torch_geometric GATv2Conv, exactly as MGAT builds it (L106-141).
  * per-hop state collection mirrors MGAT.forward's `layer_outputs.append(h)` loop (L154-172).
  * the sigmoid gate form is TxPert's GatedCombiner.gate = Sequential(Linear, Sigmoid) (L17);
    here it becomes the PonderNet halting head over the running NODE state.

DIFFERENCE from MGAT: MGAT stacks DISTINCT per-hop GATv2Conv layers of changing dim (a fixed
depth). For ADAPTIVE depth we use ONE weight-shared, dim-preserving GATv2Conv applied K times
(a recurrent GNN / IterGNN), so a single halting head applies uniformly across hops and
"number of hops" is a free per-node quantity. This is the minimal change that turns TxPert's
fixed message-passing depth into a halting axis.

THESIS POINT: unlike scDiff (re-noises the same endpoint -> depth == distance-to-endpoint ==
effect size), here depth == graph propagation distance. So E[hops] can in principle decouple
from effect size. BUT message-passing depth has its OWN confound: graph connectivity of the
perturbed gene (a hub node reaches the readout in fewer hops regardless of biology). The train
script therefore reports corr(E[hops], effect_size) AND corr(E[hops], graph_degree).
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from halting import HaltingHead, ponder_step_probs, ponder_loss


class AdaptiveTxPert(nn.Module):
    def __init__(self, n_genes, d=128, heads=4, n_hops=8, dropout=0.1,
                 lambda_prior=0.2, ponder_beta=0.05, ponder_weight=0.1):
        super().__init__()
        self.n_genes = n_genes
        self.n_hops = n_hops
        self.lambda_prior = lambda_prior
        self.ponder_beta = ponder_beta
        self.ponder_weight = ponder_weight
        # node init features: a learned embedding per gene (the "identity" of each node)
        self.gene_emb = nn.Embedding(n_genes, d)
        # ONE weight-shared, dim-preserving message-passing step (TxPert's GATv2Conv, concat=False)
        self.conv = GATv2Conv(in_channels=d, out_channels=d, heads=heads,
                              concat=False, dropout=dropout, add_self_loops=True, bias=True)
        self.norm = nn.LayerNorm(d)
        # halting head over the running node state (TxPert GatedCombiner gate form)
        self.halt_head = HaltingHead(dim=d, hidden=d)
        # decoder: a perturbed gene's propagated embedding -> full expression prediction
        self.decoder = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, n_genes))
        # learned basal expression (shared control mean, refined by training)
        self.basal = nn.Parameter(torch.zeros(n_genes))

    def propagate(self, edge_index):
        """Run K weight-shared message-passing hops over ALL gene nodes.
        Returns per-hop states, per-node halt probs p (n_genes, K), and E[hops] per node."""
        h = self.gene_emb.weight                       # (n_genes, d)
        states, lambdas = [], []
        for k in range(self.n_hops):
            h = self.conv(h, edge_index)               # one GATv2 hop (mirrors MGAT layer call)
            h = self.norm(F.silu(h))
            states.append(h)
            lambdas.append(self.halt_head(h))          # halt prob AFTER this hop, per node
        lam = torch.stack(lambdas, dim=1)              # (n_genes, K)
        p = ponder_step_probs(lam)                     # (n_genes, K) PonderNet step probs
        H = torch.stack(states, dim=1)                 # (n_genes, K, d)
        z_halted = (p.unsqueeze(-1) * H).sum(dim=1)    # (n_genes, d) halting-weighted embedding
        steps = torch.arange(1, self.n_hops + 1, device=lam.device).float()
        E_N = (p * steps).sum(dim=1)                   # (n_genes,) expected hops per node
        return dict(states=states, lam=lam, p=p, z_halted=z_halted, E_N=E_N)

    def forward(self, pert_idx, target, edge_index):
        """pert_idx: (B,) node index of the perturbed gene; target: (B, n_genes) expression."""
        prop = self.propagate(edge_index)
        z_g = prop["z_halted"][pert_idx]               # (B, d) halted embedding of perturbed node
        pred = self.basal.unsqueeze(0) + self.decoder(z_g)   # (B, n_genes)
        recon = F.mse_loss(pred, target)
        # PonderNet per-hop task loss AT THE PERTURBED NODE
        step_losses = []
        for k in range(self.n_hops):
            pred_k = self.basal.unsqueeze(0) + self.decoder(prop["states"][k][pert_idx])
            step_losses.append(((pred_k - target) ** 2).mean(dim=-1))   # (B,)
        step_losses = torch.stack(step_losses, dim=1)  # (B, K)
        lam_g = prop["lam"][pert_idx]                  # (B, K)
        ploss, aux = ponder_loss(step_losses, lam_g,
                                 lambda_prior=self.lambda_prior, beta=self.ponder_beta)
        total = recon + self.ponder_weight * ploss
        return dict(pred=pred, recon=recon, ponder=ploss, total=total,
                    E_N_batch=aux["expected_n"], prop=prop)

    @torch.no_grad()
    def signature(self, edge_index):
        """Per-node E[hops] and halt confidence — the reproducible per-perturbation readout."""
        prop = self.propagate(edge_index)
        p = prop["p"]
        ent = -(p.clamp_min(1e-9).log() * p).sum(dim=1)   # (n_genes,)
        return dict(E_N=prop["E_N"].cpu(), halt_conf=(-ent).cpu())
