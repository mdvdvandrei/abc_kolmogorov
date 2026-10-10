"""Constant-volatility Brownian base process used by ABC.

With a(t) = 0 and sigma(t) = sigma the Gaussian kernel is a Brownian bridge.
Phi and C follow Appendix L.3 of Guo et al. (arXiv:2604.27443).
"""

from __future__ import annotations

import torch


class ConstantVolatilitySDE:
    """Matches PeriodicVolatilitySDE(alpha=0) in their sde.py: phi = 1, C = sigma^2 * (min(ta, tb) - start)."""

    def __init__(self, sigma: float):
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.sigma0 = float(sigma)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(t, self.sigma0)

    def phi(self, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(start + end)

    def C(self, start: torch.Tensor, t_a: torch.Tensor, t_b: torch.Tensor) -> torch.Tensor:
        """Integrated covariance C_start(t_a, t_b) = sigma^2 * (min(t_a, t_b) - start)."""
        return self.sigma0**2 * (torch.minimum(t_a, t_b) - start)


def sample_bridge(
    sde: ConstantVolatilitySDE,
    x_prev: torch.Tensor,
    x_next: torch.Tensor,
    t: torch.Tensor,
    t_prev: torch.Tensor,
    t_next: torch.Tensor,
) -> torch.Tensor:
    """Sample p_base(x_t | x_prev, x_next). x tensors are (B, 1, H, W), times are (B,)."""
    phi_t = sde.phi(t_prev, t).view(-1, 1, 1, 1)
    phi_n = sde.phi(t_prev, t_next).view(-1, 1, 1, 1)
    c_tn = sde.C(t_prev, t, t_next).view(-1, 1, 1, 1)
    c_nn = sde.C(t_prev, t_next, t_next).view(-1, 1, 1, 1).clamp(min=1e-8)
    c_tt = sde.C(t_prev, t, t).view(-1, 1, 1, 1)
    mu = phi_t * x_prev + (c_tn / c_nn) * (x_next - phi_n * x_prev)
    var = (c_tt - c_tn**2 / c_nn).clamp(min=0.0)
    return mu + torch.sqrt(var) * torch.randn_like(x_prev)


def score_target(
    sde: ConstantVolatilitySDE,
    x_t: torch.Tensor,
    x_next: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
) -> torch.Tensor:
    """grad_{x_t} log p_base(x_next | x_t) = Phi / C * (x_next - Phi x_t)."""
    shrink = sde.phi(t, t_next)
    coeff = shrink / sde.C(t, t_next, t_next).clamp(min=1e-8)
    return coeff.view(-1, 1, 1, 1) * (x_next - shrink.view(-1, 1, 1, 1) * x_t)


def dsm_weight(
    sde: ConstantVolatilitySDE,
    t: torch.Tensor,
    t_next: torch.Tensor,
) -> torch.Tensor:
    """w = C(t, t_next, t_next) / Phi(t, t_next)."""
    return sde.C(t, t_next, t_next) / sde.phi(t, t_next).clamp(min=1e-8)


def euler_step(
    sde: ConstantVolatilitySDE,
    x: torch.Tensor,
    score: torch.Tensor,
    t: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Euler–Maruyama step of dX = sigma(t)^2 score dt + sigma(t) dB."""
    sig = sde.sigma(t).to(dtype=x.dtype).view(-1, 1, 1, 1)
    dt_b = dt.to(dtype=x.dtype).view(-1, 1, 1, 1)
    dB = torch.randn_like(x) * torch.sqrt(dt_b.clamp(min=0.0))
    return x + (sig**2) * score * dt_b + sig * dB


def self_check(n: int = 4000, sigma: float = 0.5, tol: float = 0.02) -> None:
    """Monte Carlo check of the Brownian-bridge mean and variance at t = 1/2."""
    sde = ConstantVolatilitySDE(sigma)
    x0 = torch.zeros(1, 1, 1, 1)
    x1 = torch.ones(1, 1, 1, 1)
    t_prev = torch.zeros(n)
    t_next = torch.ones(n)
    t = torch.full((n,), 0.5)
    samples = sample_bridge(
        sde,
        x0.expand(n, -1, -1, -1),
        x1.expand(n, -1, -1, -1),
        t,
        t_prev,
        t_next,
    )
    mean = samples.mean().item()
    var = samples.var(unbiased=False).item()
    expected_var = sigma**2 * 0.25
    if abs(mean - 0.5) > tol or abs(var - expected_var) > tol:
        raise AssertionError(
            f"bridge check failed: mean={mean:.4f} (want 0.5), var={var:.4f} (want {expected_var:.4f})"
        )
    xt = torch.zeros(2, 1, 1, 1)
    xn = torch.tensor([1.0, -2.0]).view(2, 1, 1, 1)
    tt = torch.tensor([0.2, 0.2])
    tn = torch.tensor([0.7, 0.7])
    target = score_target(sde, xt, xn, tt, tn)
    manual = xn / (sigma**2 * 0.5)
    if not torch.allclose(target, manual, atol=1e-5):
        raise AssertionError("score target does not match the constant-volatility formula")
    weight = dsm_weight(sde, tt, tn)
    if not torch.allclose(weight, torch.full((2,), sigma**2 * 0.5), atol=1e-5):
        raise AssertionError("DSM weight does not match sigma^2 * (t_next - t)")
