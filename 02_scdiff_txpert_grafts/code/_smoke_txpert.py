import sys; sys.path.insert(0,'.')
import torch, torch.nn as nn
from txpert_ponder import AdaptiveDepthMGAT

# realistic stub: GATv2Conv returns (nodes, heads, out_feats); MGAT then mean/concat over heads
class StubGAT(nn.Module):
    def __init__(self, d, heads=2): super().__init__(); self.lin=nn.Linear(d,d); self.heads=heads
    def forward(self, h, edge_index):
        return self.lin(h).unsqueeze(1).repeat(1, self.heads, 1)   # (nodes, heads, d)
class StubMGAT(nn.Module):
    def __init__(self, d, depth):
        super().__init__(); self.layers=nn.ModuleList([StubGAT(d) for _ in range(depth)])
        self.num_layers=depth; self.aggregation="mean"; self.activation=torch.relu

d, depth, Nn = 16, 5, 30
adm = AdaptiveDepthMGAT(StubMGAT(d, depth), hidden_dim=d, lambda_prior=0.3, beta=0.1)
h0 = torch.randn(Nn, d); ei = torch.randint(0, Nn, (2, 60))
target = torch.randn(Nn, d)
out = adm((ei, None, Nn), h0, target=target, node_readout=nn.Identity())
out["ponder_loss"].backward()
gnorm = sum(p.grad.abs().sum() for p in adm.halt_head.parameters() if p.grad is not None)
print("TxPert graft smoke PASSED")
print("  E[hops] mean = %.2f  (in [1,%d])" % (out["expected_n"].mean(), depth))
print("  h_halted shape = %s (nodes x dim)" % (tuple(out["h_halted"].shape),))
print("  lambdas shape  = %s (nodes x depth)" % (tuple(out["lambdas"].shape),))
print("  ponder loss = %.3f | grad flows into halt head: %s" % (float(out["ponder_loss"]), bool(gnorm>0)))
