"""Pinned Heun forecast.

Between saved frames the bridge is split into Heun steps that share one
Brownian increment. The step that lands on the next frame writes
x + sigma^2 * score * h and does not add noise. That frame is what the next
bridge sees.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from abc_kolmogorov.metrics import score_leads
from abc_kolmogorov.model import ABCUNet
from abc_kolmogorov.sde import ConstantVolatilitySDE

MARGIN = 7.5e-4


def visited_indices(n_frames: int, stride: int) -> list[int]:
    idx = [0]
    i = 0
    while i < n_frames - 1:
        j = min(i + stride, n_frames - 1)
        idx.append(j)
        i = j
    return idx


def pack_history(frames: list[torch.Tensor], frame_times: list[float], l_max: int):
    """Most recent l_max frames, time-ordered. Frame 0 stays visible while it fits."""
    take = frames[-l_max:]
    take_t = frame_times[-l_max:]
    ref = take[0]
    batch, _, height, width = ref.shape
    cond = ref.new_zeros(batch, l_max, 1, height, width)
    times = ref.new_zeros(batch, l_max)
    mask = ref.new_zeros(batch, l_max)
    for i, (frame, tt) in enumerate(zip(take, take_t)):
        cond[:, i] = frame
        times[:, i] = tt
        mask[:, i] = 1
    return cond, times, mask


def _score(model, x, t_scalar, t_next, cond, cond_t, cond_m, amp: bool):
    batch = x.shape[0]
    device = x.device
    t_b = torch.full((batch,), t_scalar, device=device, dtype=x.dtype)
    tn_b = torch.full((batch,), t_next, device=device, dtype=x.dtype)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        return model(x, t_b, tn_b, cond, cond_t, cond_m).float()


@torch.no_grad()
def rollout(
    model: torch.nn.Module,
    sde: ConstantVolatilitySDE,
    x0: torch.Tensor,
    n_frames: int,
    frame_stride: int,
    n_substeps: int,
    l_max: int,
) -> torch.Tensor:
    """x0 (B, 1, H, W) -> (B, n_frames, H, W). Frames the stride skips stay zero."""
    if frame_stride < 1 or n_substeps < 1:
        raise ValueError("frame_stride and n_substeps must be positive")
    batch = x0.shape[0]
    device = x0.device
    amp = device.type == "cuda"
    times = torch.linspace(0, 1, n_frames, device=device)
    sigma = sde.sigma0
    x = x0
    frames = [x0]
    frame_times = [0.0]
    out = x0.new_zeros(batch, n_frames, x0.shape[-2], x0.shape[-1])
    out[:, 0] = x0[:, 0]
    i = 0
    while i < n_frames - 1:
        j = min(i + frame_stride, n_frames - 1)
        t_left = float(times[i])
        t_right = float(times[j])
        dt = (t_right - t_left) / n_substeps
        cond, cond_t, cond_m = pack_history(frames, frame_times, l_max)
        for _sub in range(n_substeps):
            t = t_left + _sub * dt
            t_hat = t + dt
            score = _score(model, x, t, t_right, cond, cond_t, cond_m, amp)
            drift = (sigma**2) * score
            if t_hat <= t_right - MARGIN:
                noise = torch.randn_like(x) * (dt**0.5)
                pred = x + drift * dt + sigma * noise
                score2 = _score(model, pred, t_hat, t_right, cond, cond_t, cond_m, amp)
                drift2 = (sigma**2) * score2
                x = x + 0.5 * (drift + drift2) * dt + sigma * noise
            else:
                x = x + drift * dt
        frames.append(x)
        frame_times.append(t_right)
        out[:, j] = x[:, 0]
        i = j
    return out


@torch.no_grad()
def evaluate_forecast(
    model: torch.nn.Module,
    sde: ConstantVolatilitySDE,
    test_norm: np.ndarray,
    mean: float,
    std: float,
    frame_stride: int,
    n_substeps: int,
    l_max: int,
    n_members: int,
    device: torch.device,
) -> dict:
    n_traj, n_frames = test_norm.shape[:2]
    leads = [k for k in visited_indices(n_frames, frame_stride) if k > 0]
    x0 = torch.from_numpy(test_norm[:, 0, None]).to(device)
    x0 = x0.repeat_interleave(n_members, dim=0)
    pred = rollout(model, sde, x0, n_frames, frame_stride, n_substeps, l_max)
    pred_np = pred.detach().float().cpu().numpy().reshape(
        n_traj, n_members, n_frames, *test_norm.shape[-2:]
    )
    metrics = score_leads(pred_np, test_norm, leads, mean, std)
    metrics["frame_stride"] = frame_stride
    metrics["n_substeps"] = n_substeps
    metrics["n_members"] = n_members
    metrics["n_traj"] = n_traj
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pinned Heun forecast on a Kolmogorov cache")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--substeps", type=int, default=16)
    p.add_argument("--n-traj", type=int, default=4)
    p.add_argument("--n-members", type=int, default=8)
    p.add_argument("--seed", type=int, default=1000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    blob = np.load(args.cache)
    mean = float(blob["mean"])
    std = float(blob["std"])
    physical_dt = float(blob["physical_dt"]) if "physical_dt" in blob.files else 0.2
    n_traj = min(args.n_traj, blob["test"].shape[0])
    test = blob["test"][:n_traj].astype(np.float32)
    test_n = (test - mean) / std
    model = ABCUNet(base=int(ckpt["base"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    sde = ConstantVolatilitySDE(float(ckpt["sigma"]))
    torch.manual_seed(args.seed)
    print(
        "pinned Heun  stride=%d  substeps=%d  traj=%d  members=%d  sigma=%.4f"
        % (args.stride, args.substeps, n_traj, args.n_members, sde.sigma0),
        flush=True,
    )
    metrics = evaluate_forecast(
        model,
        sde,
        test_n,
        mean,
        std,
        args.stride,
        args.substeps,
        int(ckpt["l_max"]),
        args.n_members,
        device,
    )
    metrics["lead_time"] = [k * physical_dt for k in metrics["lead_frames"]]
    metrics["physical_dt"] = physical_dt
    metrics["sigma"] = float(ckpt["sigma"])
    metrics["step"] = int(ckpt.get("step", -1))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "forecast.json").write_text(json.dumps(metrics))
    for frame, t, rmse, crps, hk, z in zip(
        metrics["lead_frames"],
        metrics["lead_time"],
        metrics["rmse_ensemble_mean"],
        metrics["crps"],
        metrics["high_k_member"],
        metrics["enstrophy_member"],
    ):
        if frame not in (8, 32, 127) and frame != metrics["lead_frames"][-1]:
            continue
        print(
            "  lead %.1f  RMSE %.3f  CRPS %.3f  k>8 %.3f  Z %.2f"
            % (t, rmse, crps, hk, z),
            flush=True,
        )
    print("wrote %s" % (out / "forecast.json"), flush=True)


if __name__ == "__main__":
    main()
