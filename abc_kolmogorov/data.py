"""Causal waypoint sampling on a cached Kolmogorov trajectory.

Conditioning is a random subset of frames at or before t. Frame 0 is always
kept. The bridge endpoint is a later frame and is not shown to the network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def estimate_sigma(traj: np.ndarray) -> float:
    """sigma such that sigma * sqrt(dt_frame) matches the RMS frame-to-frame change.

    Time is normalized to [0, 1], so dt_frame = 1 / (T - 1).
    """
    dt = 1.0 / (traj.shape[1] - 1)
    diff = traj[:, 1:] - traj[:, :-1]
    rms = float(np.sqrt(np.mean(diff**2)))
    return rms / np.sqrt(dt)

def load_cache(path: Path) -> dict:
    blob = np.load(path)
    mean = float(blob["mean"])
    std = float(blob["std"])
    train = blob["train"].astype(np.float32)
    test = blob["test"].astype(np.float32)
    return {
        "train": (train - mean) / std,
        "test": (test - mean) / std,
        "train_raw": train,
        "test_raw": test,
        "mean": mean,
        "std": std,
        "sigma": float(blob["sigma"]),
        "grid": int(blob["grid"]),
        "n_frames": int(blob["n_frames"]),
        "dt": float(blob["dt"]),
        "nu": float(blob["nu"]),
        "p": int(blob["p"]),
        "physical_dt": float(blob["physical_dt"]) if "physical_dt" in blob.files else None,
    }


def _sample_one(traj: np.ndarray, l_max: int, rng: np.random.Generator) -> dict:
    """One causal training draw, matching gabeguo/abc_diffusion train.py with --force-causal.

    Conditioning is a random subset of frames at or before t (frame 0 is always kept).
    The bridge endpoint t_next is a random later frame and is not shown to the network.
    """
    n_frames, height, width = traj.shape
    shift_y = int(rng.integers(0, height))
    shift_x = int(rng.integers(0, width))
    traj = np.roll(traj, shift=(shift_y, shift_x), axis=(1, 2))

    frame_t = np.linspace(0.0, 1.0, n_frames, dtype=np.float64)
    # Same exclusion zone as their sanitize_timesteps (margin_eps=7.5e-4).
    margin = 7.5e-4
    t = float(rng.random())
    for _ in range(64):
        if 0.0 < t < 1.0 and np.min(np.abs(frame_t - t)) >= margin:
            break
        t = float(rng.random())
    else:
        gap_i = int(np.argmax(np.diff(frame_t)))
        t = float(0.5 * (frame_t[gap_i] + frame_t[gap_i + 1]))

    past = np.flatnonzero(frame_t <= t)
    future = np.flatnonzero(frame_t > t)
    if len(past) == 0 or len(future) == 0:
        raise RuntimeError(f"empty causal bracket at t={t}")

    max_k = min(l_max, len(past))
    n_cond = int(rng.integers(1, max_k + 1))
    others = past[past != 0]
    if n_cond > 1 and len(others):
        take = min(n_cond - 1, len(others))
        chosen = np.sort(rng.choice(others, size=take, replace=False))
        cond_idx = np.concatenate((np.array([0], dtype=np.int64), chosen))
    else:
        cond_idx = np.array([0], dtype=np.int64)
    prev_idx = int(cond_idx[np.argmax(frame_t[cond_idx])])
    next_idx = int(rng.choice(future))

    cond = np.zeros((l_max, 1, height, width), dtype=np.float32)
    cond_times = np.zeros((l_max,), dtype=np.float32)
    cond_mask = np.zeros((l_max,), dtype=np.float32)
    for slot, idx in enumerate(cond_idx[:l_max]):
        cond[slot, 0] = traj[idx]
        cond_times[slot] = np.float32(frame_t[idx])
        cond_mask[slot] = 1.0

    if frame_t[prev_idx] > t or frame_t[next_idx] <= t:
        raise RuntimeError(
            f"causal bracket failed: t_prev={frame_t[prev_idx]} t={t} t_next={frame_t[next_idx]}"
        )
    if np.any(cond_times[cond_mask > 0] > t + 1e-8):
        raise RuntimeError("future frame leaked into the causal conditioning set")

    return {
        "x_prev": traj[prev_idx][None].astype(np.float32),
        "x_next": traj[next_idx][None].astype(np.float32),
        "t": np.float32(t),
        "t_prev": np.float32(frame_t[prev_idx]),
        "t_next": np.float32(frame_t[next_idx]),
        "cond": cond,
        "cond_times": cond_times,
        "cond_mask": cond_mask,
    }


def sample_batch(
    traj: np.ndarray,
    batch_size: int,
    l_max: int,
    rng: np.random.Generator,
) -> dict[str, torch.Tensor]:
    """`traj` is (N, T, H, W) normalized. Returns a CPU batch."""
    n_traj = traj.shape[0]
    picks = rng.integers(0, n_traj, size=batch_size)
    examples = [_sample_one(traj[int(i)], l_max, rng) for i in picks]
    batch = {}
    for key in examples[0]:
        batch[key] = torch.from_numpy(np.stack([ex[key] for ex in examples], axis=0))
    return batch


def sample_batch_cuda(
    traj: torch.Tensor,
    batch_size: int,
    l_max: int,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Vectorized copy of `_sample_one` on the same device as `traj`.

    `traj` is (N, T, H, W), normalized. Same causal rule: frame 0 is always
    conditioned, other conditions are a uniform subset of frames at or before
    t, and t_next is a uniform frame strictly after t (not shown to the net).
    """
    if l_max < 1:
        raise ValueError("l_max must be at least 1")
    device = traj.device
    n_traj, n_frames, height, width = traj.shape
    b = batch_size
    g = generator

    idx = torch.randint(0, n_traj, (b,), device=device, generator=g)
    sy = torch.randint(0, height, (b,), device=device, generator=g)
    sx = torch.randint(0, width, (b,), device=device, generator=g)

    frame_t = torch.linspace(0, 1, n_frames, device=device, dtype=traj.dtype)
    margin = 7.5e-4
    t = torch.rand(b, device=device, generator=g)
    # Fixed number of redraws, no host sync. Leftover collisions use the first-gap midpoint.
    for _ in range(8):
        dist = (t[:, None] - frame_t[None, :]).abs().amin(dim=1)
        bad = (t <= 0) | (t >= 1) | (dist < margin)
        t = torch.where(bad, torch.rand(b, device=device, generator=g), t)
    dist = (t[:, None] - frame_t[None, :]).abs().amin(dim=1)
    bad = (t <= 0) | (t >= 1) | (dist < margin)
    t = torch.where(bad, 0.5 * (frame_t[0] + frame_t[1]), t)

    # Random observation stride. 1,2,4,8 frames → physical gaps 0.2, 0.4, 0.8, 1.6.
    # Times stay on the full clip clock. Frame 0 and the last frame are always kept.
    stride = torch.tensor([1, 2, 4, 8], device=device)[
        torch.randint(0, 4, (b,), device=device, generator=g)
    ]
    frame_idx = torch.arange(n_frames, device=device)
    allowed = (frame_idx[None, :] % stride[:, None]) == 0
    allowed[:, 0] = True
    allowed[:, -1] = True
    past = (frame_t[None, :] <= t[:, None]) & allowed
    future = (frame_t[None, :] > t[:, None]) & allowed
    n_past = past.sum(dim=1)
    max_k = torch.clamp(n_past, max=l_max)
    n_cond = 1 + (torch.rand(b, device=device, generator=g) * max_k).long()
    n_cond = torch.minimum(n_cond, max_k)

    k_take = l_max - 1
    if k_take > 0:
        prio = torch.rand(b, n_frames, device=device, generator=g)
        blocked = ~past
        blocked[:, 0] = True
        prio = prio.masked_fill(blocked, -1.0)
        vals, extra_idx = prio.topk(min(k_take, n_frames), dim=1)
        if extra_idx.shape[1] < k_take:
            raise RuntimeError("not enough frames to fill the conditioning slots")
        rank = torch.arange(k_take, device=device).view(1, -1)
        extra_valid = (rank < (n_cond - 1).clamp(min=0).view(b, 1)) & (vals[:, :k_take] > -0.5)
        extra_idx = extra_idx[:, :k_take]
    else:
        extra_idx = torch.zeros(b, 0, dtype=torch.long, device=device)
        extra_valid = torch.zeros(b, 0, dtype=torch.bool, device=device)

    fut_prio = torch.rand(b, n_frames, device=device, generator=g).masked_fill(~future, -1.0)
    next_idx = fut_prio.argmax(dim=1)

    # Slots: frame 0, up to l_max-1 past extras, then the unobserved future endpoint.
    need = torch.cat(
        [
            torch.zeros(b, 1, dtype=torch.long, device=device),
            extra_idx,
            next_idx.view(b, 1),
        ],
        dim=1,
    )
    ys = torch.arange(height, device=device)
    xs = torch.arange(width, device=device)
    iy = (ys.view(1, 1, height, 1) - sy.view(b, 1, 1, 1)) % height
    ix = (xs.view(1, 1, 1, width) - sx.view(b, 1, 1, 1)) % width
    frames = traj[
        idx.view(b, 1, 1, 1),
        need[:, :, None, None],
        iy,
        ix,
    ]

    cond = torch.zeros(b, l_max, 1, height, width, device=device, dtype=traj.dtype)
    cond_times = torch.zeros(b, l_max, device=device, dtype=traj.dtype)
    cond_mask = torch.zeros(b, l_max, device=device, dtype=traj.dtype)
    cond[:, 0, 0] = frames[:, 0]
    cond_times[:, 0] = frame_t[0]
    cond_mask[:, 0] = 1
    for slot in range(k_take):
        valid = extra_valid[:, slot].to(dtype=traj.dtype)
        cond[:, slot + 1, 0] = frames[:, slot + 1] * valid[:, None, None]
        cond_times[:, slot + 1] = frame_t[extra_idx[:, slot]] * valid
        cond_mask[:, slot + 1] = valid

    prev_times = cond_times.masked_fill(cond_mask == 0, -1)
    prev_slot = prev_times.argmax(dim=1)
    bb = torch.arange(b, device=device)
    x_prev = cond[bb, prev_slot, 0].unsqueeze(1)
    t_prev = cond_times[bb, prev_slot]
    x_next = frames[:, -1].unsqueeze(1)
    t_next = frame_t[next_idx]
    return {
        "x_prev": x_prev,
        "x_next": x_next,
        "t": t,
        "t_prev": t_prev,
        "t_next": t_next,
        "cond": cond,
        "cond_times": cond_times,
        "cond_mask": cond_mask,
    }

