"""Pseudo-spectral RK4 on the torus, with 2/3 dealiasing.

A GPU batch is one fft2 over (batch, N, N). Kolmogorov flow subclasses this
and replaces the right-hand side.
"""

from __future__ import annotations

import numpy as np
import torch


class SpectralFlow:
    def __init__(
        self,
        N: int = 64,
        nu: float = 1e-12,
        p: int = 4,
        dt: float = 0.005,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        self.N = int(N)
        self.nu = float(nu)
        self.p = int(p)
        self.dt = float(dt)
        self.device = torch.device(device)
        self.real = dtype
        self.complex = torch.complex128 if dtype == torch.float64 else torch.complex64

        k = torch.fft.fftfreq(self.N, d=1.0 / self.N, device=self.device, dtype=self.real)
        kx = k[:, None].expand(self.N, self.N).contiguous()
        ky = k[None, :].expand(self.N, self.N).contiguous()
        ksq = kx**2 + ky**2
        ksq = ksq.clone()
        ksq[0, 0] = 1.0
        inv = (-1.0 / ksq).clone()
        inv[0, 0] = 0.0
        kmax = self.N // 3
        dealias = torch.ones(self.N, self.N, device=self.device, dtype=self.real)
        dealias[kx.abs() > kmax] = 0.0
        dealias[ky.abs() > kmax] = 0.0

        self.kx = kx
        self.ky = ky
        self.ksq = ksq
        self.inv_lap = inv
        self.hypvisc = -self.nu * ksq**self.p
        self.dealias = dealias

    def ic_hat(self, seeds: list[int]) -> torch.Tensor:
        """Spectral initial condition. The seed fixes the phase."""
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
            q_hat = amp * np.exp(1j * phase)
            q = np.fft.ifft2(q_hat).real
            q = q / (q.std() + 1e-12) * 5.0
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
        return -jac + self.hypvisc * q_hat

    def step(self, q_hat: torch.Tensor) -> torch.Tensor:
        dt = self.dt
        k1 = self.rhs(q_hat)
        k2 = self.rhs(q_hat + 0.5 * dt * k1)
        k3 = self.rhs(q_hat + 0.5 * dt * k2)
        k4 = self.rhs(q_hat + dt * k3)
        return q_hat + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def advance(self, q_hat: torch.Tensor, n_steps: int) -> torch.Tensor:
        for _ in range(n_steps):
            q_hat = self.step(q_hat)
        return q_hat

    @torch.no_grad()
    def rollout(
        self,
        seeds: list[int],
        n_frames: int,
        save_every: int,
        spinup: int,
    ) -> torch.Tensor:
        """Returns float32 frames (batch, n_frames, N, N) on CPU."""
        q_hat = self.ic_hat(seeds)
        if spinup:
            q_hat = self.advance(q_hat, spinup)
        frames = torch.empty(
            (len(seeds), n_frames, self.N, self.N), dtype=torch.float32, device="cpu"
        )
        frames[:, 0] = torch.fft.ifft2(q_hat).real.to(dtype=torch.float32).cpu()
        for i in range(1, n_frames):
            q_hat = self.advance(q_hat, save_every)
            frames[:, i] = torch.fft.ifft2(q_hat).real.to(dtype=torch.float32).cpu()
        return frames
