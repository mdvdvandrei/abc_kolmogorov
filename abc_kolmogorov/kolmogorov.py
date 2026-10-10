"""Forced 2D Kolmogorov flow on a pseudo-spectral RK4 core.

Vorticity on [0, 2π]²:
    dω/dt + u·∇ω = ν ∇²ω + curl(sin(k_f y), 0)
The nonlinear term and the streamfunction inversion match SpectralFlow.
Newtonian viscosity is hyperviscosity with p = 1.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from abc_kolmogorov.spectral import SpectralFlow


class BatchedKolmogorov(SpectralFlow):
    def __init__(
        self,
        N: int = 64,
        nu: float = 0.02,
        kf: int = 4,
        dt: float = 0.005,
        drag: float = 0.0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__(N=N, nu=nu, p=1, dt=dt, device=device, dtype=dtype)
        self.kf = int(kf)
        self.drag = float(drag)
        y = torch.arange(self.N, device=self.device, dtype=self.real) * (2.0 * math.pi / self.N)
        # curl(sin(kf y), 0) = -kf cos(kf y)
        force = (-float(self.kf) * torch.cos(self.kf * y)).view(1, self.N).expand(self.N, self.N)
        self.force_hat = torch.fft.fft2(force.contiguous()).to(self.complex)

    def ic_hat(self, seeds: list[int], scale: float = 1.0) -> torch.Tensor:
        kx = self.kx.detach().cpu().numpy()
        ky = self.ky.detach().cpu().numpy()
        k = np.sqrt(kx**2 + ky**2)
        amp = np.zeros((self.N, self.N), dtype=np.float64)
        mask = (k > 0) & (k < self.N / 3.0)
        amp[mask] = (k[mask] ** -1) * np.exp(-(k[mask] ** 2) / 50.0)
        hats = np.empty((len(seeds), self.N, self.N), dtype=np.complex128)
        for i, seed in enumerate(seeds):
            rng = np.random.default_rng(int(seed))
            phase = rng.uniform(0.0, 2.0 * np.pi, size=(self.N, self.N))
            q = np.fft.ifft2(amp * np.exp(1j * phase)).real
            q = q / (q.std() + 1e-12) * scale
            hats[i] = np.fft.fft2(q)
        return torch.tensor(hats, device=self.device, dtype=self.complex)

    def rhs(self, q_hat: torch.Tensor) -> torch.Tensor:
        psi = self.inv_lap * q_hat
        deal = self.dealias
        u = torch.fft.ifft2(-1j * self.ky * psi * deal).real
        v = torch.fft.ifft2(1j * self.kx * psi * deal).real
        qx = torch.fft.ifft2(1j * self.kx * q_hat * deal).real
        qy = torch.fft.ifft2(1j * self.ky * q_hat * deal).real
        jac = torch.fft.fft2(u * qx + v * qy) * deal
        return -jac + self.hypvisc * q_hat - self.drag * q_hat + self.force_hat

    def speed(self, q_hat: torch.Tensor) -> torch.Tensor:
        psi = self.inv_lap * q_hat
        deal = self.dealias
        u = torch.fft.ifft2(-1j * self.ky * psi * deal).real
        v = torch.fft.ifft2(1j * self.kx * psi * deal).real
        return torch.sqrt(u * u + v * v)
