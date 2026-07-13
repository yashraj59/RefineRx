"""
scdiff_ponder.py — graft PonderNet halting onto scDiff's reverse-diffusion chain.

Grounded in OmicsML/scDiff scdiff/model.py (read 2026-07):
  * ScDiff.p_sample_loop (L728)  = the reverse chain over reversed(range(t_start+1)); @torch.no_grad
  * ScDiff.p_sample     (L701)  = one denoising step -> model_mean (+ optional VLB)
  * ScDiff.predict_start_from_noise (L661) = running x_hat_0 estimate from (x_t, eps_theta)
  * ScDiff.p_mean_variance (L676) = wraps the model; parameterization="eps" by default
  * per-step VLB already computed in p_sample when calculate_vlb=True (L719-723)

WHY scDiff is the clean host: the reverse chain provably moves x_t toward the data
manifold, and x_hat_0 at step t is the model's CURRENT endpoint prediction. "How many
steps until x_hat_0 stops changing" is a per-perturbation complexity scalar with NO
graph-distance confound. PonderNet turns that into a *learned* halting decision.

This is an integration SKETCH: it subclasses ScDiff and overrides the sampling loop with
a grad-enabled, halting-aware variant. Adapt attribute names to the installed version.
"""
from __future__ import annotations
import torch
from halting import HaltingHead, ponder_loss, ponder_step_probs

# from scdiff.model import ScDiff   # <- the real import in-repo


class ScDiffPonder:  # in-repo: class ScDiffPonder(ScDiff)
    """Mixin/subclass adding a learned-halting reverse chain to ScDiff.

    Assumes the host ScDiff instance exposes (all verified in model.py):
        self.num_timesteps, self.betas.device
        self.q_sample(x_start, t, noise)
        self.p_mean_variance(x_start, x, t, clip_denoised, ...)- returns (mean, _, logvar)
        self.predict_start_from_noise(x_t, t, noise)          - running x_hat_0
        self.model(...)                                        - eps_theta network
    """

    def _init_halting(self, feat_dim: int, lambda_prior: float = 0.15, beta: float = 0.1):
        # feat_dim = dimensionality of the pooled state the head reads (e.g. gene dim or a latent)
        self.halt_head = HaltingHead(dim=feat_dim, hidden=256)
        self.lambda_prior = lambda_prior
        self.ponder_beta = beta

    # ---- grad-enabled reverse chain with a halt probability emitted per step ----
    def ponder_sample_loop(self, x_start, shape, t_start, target=None,
                           conditions=None, input_gene_list=None, target_gene_list=None,
                           text_embeddings=None, aug_graph=None, mask=None,
                           max_steps: int | None = None, train: bool = True):
        """Run the reverse chain, collecting per-step endpoint predictions and halt probs.

        Returns dict with x_hat0_steps (list), lambdas (B,N), and (if target given & train)
        the ponder loss + signature readouts.
        """
        device = self.betas.device
        # start from noise at t_start (mirrors p_sample_loop L735-745)
        if isinstance(conditions, dict):
            N = conditions[list(conditions)[0]].shape[0]
        else:
            N = conditions.shape[0]
        x = torch.randn(N, x_start.shape[1], device=device)
        t_iter = list(reversed(range(0, min(t_start, self.num_timesteps)) ))
        if max_steps:  # optionally strided/truncated schedule to bound compute
            t_iter = t_iter[:max_steps]

        lambdas, xhat0_steps, step_losses = [], [], []
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for i in t_iter:
                t = torch.full((N,), i, device=device, dtype=torch.long)
                # one denoising step -> model_mean, logvar  (p_mean_variance, L676)
                model_mean, _, model_logvar = self.p_mean_variance(
                    x_start=x_start, x=x, t=t, clip_denoised=self.clip_denoised,
                    conditions=conditions, input_gene_list=input_gene_list, aug_graph=aug_graph,
                    mask=mask)
                # running endpoint prediction x_hat_0 (predict_start_from_noise, L661)
                eps = self.model(x_start, x, t=t, conditions=conditions,
                                 input_gene_list=input_gene_list, aug_graph=aug_graph, mask=mask)[0]
                x_hat0 = self.predict_start_from_noise(x, t, eps)
                xhat0_steps.append(x_hat0)

                # halt probability reads the CURRENT endpoint estimate
                lambdas.append(self.halt_head(x_hat0))

                # per-step task loss = how far the current endpoint estimate is from target
                if target is not None:
                    step_losses.append(((x_hat0 - target) ** 2).mean(dim=1))

                # advance the chain (ancestral sampling; no noise at t==0)
                noise = torch.randn_like(x) if i > 0 else 0.0
                x = model_mean + (0.5 * model_logvar).exp() * noise

        lambdas = torch.stack(lambdas, dim=1)                 # (B, N)
        out = {"x_hat0_steps": xhat0_steps, "lambdas": lambdas, "x_final": x}
        if target is not None:
            step_losses = torch.stack(step_losses, dim=1)     # (B, N)
            loss, aux = ponder_loss(step_losses, lambdas,
                                    lambda_prior=self.lambda_prior, beta=self.ponder_beta)
            out.update({"ponder_loss": loss, **aux})          # aux: p, expected_n, halt_entropy
        return out

    # ---- the five-part per-perturbation signature, read off one ponder pass ----
    @torch.no_grad()
    def perturbation_signature(self, x_start, t_start, target, **kw):
        o = self.ponder_sample_loop(x_start, x_start.shape, t_start, target=target, train=False, **kw)
        p = ponder_step_probs(o["lambdas"])
        xh = torch.stack(o["x_hat0_steps"], dim=1)            # (B, N, G)
        delta = o["x_final"] - x_start                        # predicted response
        # nonlinear-correction magnitude = how much steps 2..N change the 1-step endpoint
        nonlin = (xh[:, -1] - xh[:, 0]).norm(dim=1)
        return {
            "response_shift":        delta.norm(dim=1),               # effect size (the covariate)
            "refinement_rounds":     o["expected_n"],                 # E[N]
            "nonlinear_correction":  nonlin,
            "halt_confidence":       -o["halt_entropy"],              # higher = sharper halt
            # 'stability' is computed ACROSS seeds/guides/donors by running this repeatedly
        }
