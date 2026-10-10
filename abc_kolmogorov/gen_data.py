"""GPU cache of forced Kolmogorov flow at nu=0.01, kf=4.

The probe at this viscosity stays unsteady, with vorticity spread across
several wavenumbers rather than a single condensate. Solver dt stays 0.005
(CFL was about 0.26). Frames are 0.2 apart after a spinup of t=30.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from abc_kolmogorov.kolmogorov import BatchedKolmogorov


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    nu = 0.01
    dt = 0.005
    spinup = 6000
    save_every = 40
    n_frames = 128
    n_train = 512
    n_test = 32
    train_seed0 = 0
    test_seed0 = 100_000
    batch = 64
    physical_dt = dt * save_every

    solver = BatchedKolmogorov(N=64, nu=nu, kf=4, dt=dt, device=device, dtype=torch.float64)
    print(
        f"gpu={torch.cuda.get_device_name(0)}  nu={nu} kf=4  train={n_train} test={n_test}  "
        f"T={n_frames}  frame_dt={physical_dt}  spinup_t={dt * spinup:.1f}",
        flush=True,
    )

    def run_split(n_traj: int, seed0: int, label: str) -> np.ndarray:
        out = np.empty((n_traj, n_frames, solver.N, solver.N), dtype=np.float32)
        t_start = time.perf_counter()
        done = 0
        while done < n_traj:
            take = min(batch, n_traj - done)
            seeds = list(range(seed0 + done, seed0 + done + take))
            q_hat = solver.ic_hat(seeds, scale=1.0)
            if spinup:
                q_hat = solver.advance(q_hat, spinup)
            frames = np.empty((take, n_frames, solver.N, solver.N), dtype=np.float32)
            for i in range(n_frames):
                if i:
                    q_hat = solver.advance(q_hat, save_every)
                frames[:, i] = torch.fft.ifft2(q_hat).real.detach().to(dtype=torch.float32).cpu().numpy()
            if not np.isfinite(frames).all():
                raise SystemExit(f"non-finite field in {label} at traj {done}")
            out[done : done + take] = frames
            done += take
            z0 = 0.5 * np.mean(frames[:, 0] ** 2, axis=(1, 2))
            z1 = 0.5 * np.mean(frames[:, -1] ** 2, axis=(1, 2))
            rate = done / max(time.perf_counter() - t_start, 1e-6)
            print(
                f"  {label} {done}/{n_traj}  Z0={z0.mean():.3f} ZT={z1.mean():.3f}  {rate:.2f} traj/s",
                flush=True,
            )
        return out

    train = run_split(n_train, train_seed0, "train")
    test = run_split(n_test, test_seed0, "test")
    if not np.isfinite(train).all():
        raise SystemExit("non-finite train cache")
    mean = float(train.mean())
    std = float(train.std())
    normed = (train - mean) / std
    diff = normed[:, 1:] - normed[:, :-1]
    sigma = float(np.sqrt(np.mean(diff**2)) / np.sqrt(1.0 / (n_frames - 1)))
    path = Path(__file__).resolve().parents[1] / "cache" / "kolmogorov_64.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train=train,
        test=test,
        mean=np.float32(mean),
        std=np.float32(std),
        sigma=np.float32(sigma),
        grid=np.int32(64),
        n_frames=np.int32(n_frames),
        nu=np.float64(nu),
        kf=np.int32(4),
        p=np.int32(1),
        dt=np.float64(dt),
        spinup=np.int32(spinup),
        save_every=np.int32(save_every),
        seed=np.int32(train_seed0),
        test_seed=np.int32(test_seed0),
        physical_dt=np.float64(physical_dt),
    )
    print(f"wrote {path}  mean={mean:.4f} std={std:.4f} sigma={sigma:.4f}", flush=True)


if __name__ == "__main__":
    main()
