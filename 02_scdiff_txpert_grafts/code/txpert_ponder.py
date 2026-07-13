"""
txpert_ponder.py — adaptive message-passing DEPTH for TxPert via PonderNet / ACT.

Grounded in valence-labs/TxPert gspp/models/pert_models/multi_graph.py (read 2026-07):
  * MGAT.__init__(num_hidden, ...)  -> num_hidden IS the GNN depth; builds
    self.layers = ModuleList of GATv2Conv (L96-141). num_layers = num_hidden (L95).
  * MGAT.forward(graph_tuple, inputs) (L146) loops `for i, layer in enumerate(self.layers)`
    and ALREADY collects layer_outputs = [h after hop 1, hop 2, ...]  (L154-172).
    -> this per-hop list is exactly the ACT axis: each hop is one refinement round.
  * GatedCombiner.gate = nn.Sequential(Linear, Sigmoid) (L17) -> the idiomatic gate form
    HaltingHead reuses; here we add a SEPARATE halting head over the running node state.

IMPORTANT distinction (from reading the code): TxPert has TWO "layer" axes.
  (a) MGAT depth = sequential message-passing hops  <-- the TRUE refinement axis; halt here.
  (b) MultiGraph per-graph-type layers combined by GatedCombiner = PARALLEL graph channels,
      NOT sequential refinement; do NOT treat these as halting steps.
CAVEAT for the thesis: message-passing depth conflates GRAPH DISTANCE with response
complexity (a graph-distant perturbed gene needs more hops regardless of biology). This is
why scDiff (generative, no graph) is the primary backbone and TxPert is the architecture-
invariance CROSS-CHECK: agreement of the E[N] ranking across the two is the strong result.

Integration SKETCH: a depth-adaptive MGAT that emits a halt prob after each hop and reads
out E[hops]. GEARS note: deepen first (num_go_gnn_layers defaults to 1) before halting is
meaningful.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from halting import HaltingHead, ponder_loss, ponder_step_probs


class AdaptiveDepthMGAT(nn.Module):
    """Wraps TxPert's MGAT so message-passing depth halts adaptively per node/perturbation.

    Assumes the host MGAT exposes (all verified in multi_graph.py):
        self.layers (ModuleList of GATv2Conv), self.num_layers,
        self.aggregation ('concat'|'mean'), self.activation
    We reproduce MGAT.forward's per-hop update and add a halting head over the node state.
    """
    def __init__(self, mgat, hidden_dim: int, lambda_prior: float = 0.3, beta: float = 0.1):
        super().__init__()
        self.mgat = mgat                                   # the original MGAT (layers reused)
        self.halt_head = HaltingHead(dim=hidden_dim, hidden=hidden_dim)
        self.lambda_prior = lambda_prior
        self.ponder_beta = beta

    def forward(self, graph_tuple, inputs, target=None, node_readout=None):
        """Run message passing, emitting a halt prob after each hop.

        node_readout: optional fn(h)->prediction to score per-hop task loss for PonderNet.
        Returns per-hop states, halt probs, and (if target+readout given) ponder loss + E[hops].
        """
        h = inputs
        edge_index, edge_weight, total_nodes = graph_tuple
        edge_index = edge_index.long()

        hop_states, lambdas, step_losses = [], [], []
        for i, layer in enumerate(self.mgat.layers):        # mirror MGAT.forward L154
            h = layer(h, edge_index)
            h = h.flatten(1) if self.mgat.aggregation == "concat" else h.mean(1)
            if self.mgat.activation is not None:
                h = self.mgat.activation(h)
            hop_states.append(h)
            lambdas.append(self.halt_head(h))               # halt prob AFTER this hop
            if target is not None and node_readout is not None:
                pred_i = node_readout(h)
                step_losses.append(((pred_i - target) ** 2).mean(dim=-1))

        lambdas = torch.stack(lambdas, dim=1)               # (nodes, depth)
        out = {"hop_states": hop_states, "lambdas": lambdas}
        if target is not None and node_readout is not None:
            step_losses = torch.stack(step_losses, dim=1)
            loss, aux = ponder_loss(step_losses, lambdas,
                                    lambda_prior=self.lambda_prior, beta=self.ponder_beta)
            out.update({"ponder_loss": loss, **aux})        # expected_n = E[hops] per node
        # halting-weighted node embedding (soft mixture over depth) for downstream decoder
        p = ponder_step_probs(lambdas).unsqueeze(-1)        # (nodes, depth, 1)
        out["h_halted"] = (p * torch.stack(hop_states, dim=1)).sum(dim=1)
        return out


# --- GEARS variant: same idea, but you must deepen the GO-GNN stack first ---
# In snap-stanford/GEARS gears/model.py, num_go_gnn_layers defaults to 1 -> nothing to halt.
# Set num_go_gnn_layers >= 3, then wrap the SGConv loop exactly as above (each SGConv hop = one step).
