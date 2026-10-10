"""Train the causal ABC score on a cached Kolmogorov flow."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from abc_kolmogorov.data import load_cache, sample_batch, sample_batch_cuda
from abc_kolmogorov.model import ABCUNet, count_parameters
from abc_kolmogorov.sample import evaluate_forecast
from abc_kolmogorov.sde import (
    ConstantVolatilitySDE,
    dsm_weight,
    sample_bridge,
    score_target,
    self_check,
)

ROOT = Path(__file__).resolve().parent


def shell_power_weight(n: int, k_split: float, high: float, device, dtype) -> torch.Tensor:
    """Multiplier on Fourier power: 1 at k <= k_split, `high` above it."""
    freq = torch.fft.fftfreq(n, d=1.0 / n, device=device)
    rad = torch.sqrt(freq[:, None] ** 2 + freq[None, :] ** 2)
    weight = torch.ones(n, n, device=device, dtype=dtype)
    return torch.where(rad > k_split, torch.full_like(weight, high), weight)


def dsm_loss(
    model: ABCUNet,
    sde: ConstantVolatilitySDE,
    batch: dict[str, torch.Tensor],
    amp: bool = False,
    parts: bool = False,
    k_split: float = 8.0,
    high_k_weight: float = 1.0,
):
    """Same objective as non_markov_diffusion/loss.py::dsm_loss.

    The network output is the score. No Brownian-bridge residual is added:
    that flag is off in their causal runs, and a future frame is not observed.
    The bridge sample and the target stay in fp32; autocast covers the network.
    """
    x_t = sample_bridge(
        sde,
        batch["x_prev"],
        batch["x_next"],
        batch["t"],
        batch["t_prev"],
        batch["t_next"],
    )
    target = score_target(sde, x_t, batch["x_next"], batch["t"], batch["t_next"])
    weight = dsm_weight(sde, batch["t"], batch["t_next"]).view(-1, 1, 1, 1)
    with torch.autocast(device_type=x_t.device.type, dtype=torch.bfloat16, enabled=amp):
        pred = model(
            x_t,
            batch["t"],
            batch["t_next"],
            batch["cond"],
            batch["cond_times"],
            batch["cond_mask"],
        )
    residual = pred.float() - target
    if high_k_weight != 1.0:
        shells = shell_power_weight(
            residual.shape[-1], k_split, high_k_weight, residual.device, residual.dtype
        )
        spec = torch.fft.fft2(residual[:, 0]) * torch.sqrt(shells)
        residual = torch.fft.ifft2(spec).real.unsqueeze(1)
    per = (weight * residual.square()).mean(dim=(1, 2, 3))
    loss = per.mean()
    if parts:
        return loss, per.detach()
    return loss


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def scheduled_lr(step: int, total: int, base: float, warmup: int, floor: float) -> float:
    if warmup > 0 and step <= warmup:
        return base * step / warmup
    span = max(1, total - warmup)
    progress = min(1.0, (step - warmup) / span)
    return floor + 0.5 * (base - floor) * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(path: Path, model: torch.nn.Module, data: dict, args: argparse.Namespace, step: int) -> None:
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(
        {
            "model": raw.state_dict(),
            "mean": data["mean"],
            "std": data["std"],
            "sigma": data["sigma"],
            "base": args.base,
            "l_max": args.l_max,
            "grid": data["grid"],
            "n_frames": data["n_frames"],
            "step": step,
            "cache": str(args.cache),
            "args": vars(args),
        },
        path,
    )


def fit_batch(
    model, sde, traj, requested: int, l_max: int, generator, amp: bool, k_split: float, high_k_weight: float
) -> int:
    """Largest power-of-two batch at or below `requested` that survives one step."""
    trial = int(requested)
    while trial >= 8:
        try:
            torch.cuda.empty_cache()
            batch = sample_batch_cuda(traj, trial, l_max, generator)
            loss = dsm_loss(
                model, sde, batch, amp=amp, k_split=k_split, high_k_weight=high_k_weight
            )
            loss.backward()
            model.zero_grad(set_to_none=True)
            del batch, loss
            torch.cuda.empty_cache()
            used = torch.cuda.max_memory_allocated() / 2**30
            print(f"batch {trial} fits  peak={used:.1f} GiB", flush=True)
            return trial
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print(f"batch {trial} does not fit, trying {trial // 2}", flush=True)
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            trial //= 2
    raise SystemExit("no batch size fit in GPU memory")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train causal ABC on Kolmogorov flow")
    p.add_argument("--smoke", action="store_true", help="200 steps, batch 4")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--compile", action="store_true", help="torch.compile the score network")
    p.add_argument("--no-amp", action="store_true", help="disable bfloat16 autocast on CUDA")
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--val-every", type=int, default=0)
    p.add_argument("--val-traj", type=int, default=2)
    p.add_argument("--val-members", type=int, default=8)
    p.add_argument("--l-max", type=int, default=8)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--cache", type=str, required=True)
    p.add_argument("--stride", type=int, default=4, help="frames per bridge at validation")
    p.add_argument("--substeps", type=int, default=16, help="Heun steps per validation bridge")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--sigma", type=float, default=0.0, help="override cache sigma; 0 keeps the estimate")
    p.add_argument("--k-split", type=float, default=8.0)
    p.add_argument("--high-k-weight", type=float, default=1.0, help="Fourier power weight above k-split")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.steps = 200
        args.batch_size = 4
        args.log_every = 20
        if not args.out:
            args.out = str(ROOT.parent / "outputs" / "smoke")
    if not args.out:
        args.out = str(ROOT.parent / "outputs" / "run")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    if hasattr(torch.backends, "nnpack"):
        torch.backends.nnpack.enabled = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    self_check()
    print("SDE self-check passed", flush=True)

    data = load_cache(Path(args.cache))
    if args.sigma > 0:
        print(f"sigma override {data['sigma']:.4f} -> {args.sigma:.4f}", flush=True)
        data["sigma"] = float(args.sigma)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    sde = ConstantVolatilitySDE(data["sigma"])
    model = ABCUNet(base=args.base).to(device)
    n_params = count_parameters(model)
    if args.compile and device.type == "cuda":
        model = torch.compile(model, mode="reduce-overhead")
    train_gpu = None
    cuda_gen = None
    train_shape = tuple(data["train"].shape)
    if device.type == "cuda":
        train_gpu = torch.from_numpy(np.ascontiguousarray(data["train"])).to(device)
        del data["train"]
        data.pop("train_raw", None)
        data.pop("test_raw", None)
        cuda_gen = torch.Generator(device=device)
        cuda_gen.manual_seed(args.seed)
    print(
        f"device={device}  params={n_params:,}  "
        f"sigma={data['sigma']:.4f}  train={train_shape}  "
        f"steps={args.steps}  batch={args.batch_size}  lr={args.lr}  "
        f"amp={use_amp}  compile={bool(args.compile and device.type == 'cuda')}",
        flush=True,
    )
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
        args.batch_size = fit_batch(
            model,
            sde,
            train_gpu,
            args.batch_size,
            args.l_max,
            cuda_gen,
            use_amp,
            args.k_split,
            args.high_k_weight,
        )
        # Previous runs used 2e-4 at batch 128. Scale with the square root of the batch.
        args.lr = args.lr * math.sqrt(args.batch_size / 128.0)
        print(f"using batch={args.batch_size}  lr={args.lr:.3e}  steps={args.steps}", flush=True)
    opt_kwargs = dict(lr=args.lr, betas=(0.9, 0.999))
    if device.type == "cuda":
        opt_kwargs["fused"] = True
    opt = torch.optim.Adam(model.parameters(), **opt_kwargs)
    val_traj = None
    if args.val_every > 0:
        n_val = min(args.val_traj, data["test"].shape[0])
        val_traj = data["test"][:n_val]

    model.train()
    logged: list[float] = []
    val_history: list[dict] = []
    gap_sum = torch.zeros(4, device=device)
    gap_count = torch.zeros(4, device=device)
    loss_window = torch.zeros((), device=device)
    window_n = 0
    floor_lr = args.lr * 0.1
    t_start = time.perf_counter()
    t_mark = t_start
    for step in range(1, args.steps + 1):
        lr = scheduled_lr(step, args.steps, args.lr, args.warmup, floor_lr)
        for group in opt.param_groups:
            group["lr"] = lr
        if train_gpu is not None:
            batch = sample_batch_cuda(train_gpu, args.batch_size, args.l_max, cuda_gen)
        else:
            batch = move_batch(
                sample_batch(data["train"], args.batch_size, args.l_max, rng),
                device,
            )
        opt.zero_grad(set_to_none=True)
        loss, per = dsm_loss(
            model,
            sde,
            batch,
            amp=use_amp,
            parts=True,
            k_split=args.k_split,
            high_k_weight=args.high_k_weight,
        )
        gap = (batch["t_next"] - batch["t"]).detach()
        gap_bin = torch.zeros_like(gap, dtype=torch.long)
        gap_bin = torch.where(gap >= 0.01, torch.ones_like(gap_bin), gap_bin)
        gap_bin = torch.where(gap >= 0.05, torch.full_like(gap_bin, 2), gap_bin)
        gap_bin = torch.where(gap >= 0.2, torch.full_like(gap_bin, 3), gap_bin)
        if not torch.isfinite(loss):
            print(f"step {step}: non-finite loss, skipping", flush=True)
            opt.zero_grad(set_to_none=True)
            continue
        gap_sum.scatter_add_(0, gap_bin, per)
        gap_count.scatter_add_(0, gap_bin, torch.ones_like(per))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        loss_window = loss_window + loss.detach()
        window_n += 1
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            avg = float((loss_window / max(window_n, 1)).item())
            logged.append(avg)
            now = time.perf_counter()
            rate = args.log_every / max(now - t_mark, 1e-6) if step > 1 else 1.0 / max(now - t_mark, 1e-6)
            t_mark = now
            gap_mean = (gap_sum / gap_count.clamp(min=1)).tolist()
            gap_n = gap_count.tolist()
            print(
                f"step {step:6d}  loss={avg:.5f}  grad={float(grad_norm):.3f}  "
                f"lr={lr:.2e}  {rate:.1f} step/s  {rate * args.batch_size:.0f} ex/s  "
                f"gap_loss={[round(v, 3) for v in gap_mean]} n={[int(v) for v in gap_n]}",
                flush=True,
            )
            loss_window = torch.zeros((), device=device)
            window_n = 0
            gap_sum.zero_()
            gap_count.zero_()
        if val_traj is not None and args.val_every > 0 and (step % args.val_every == 0 or step == args.steps):
            model.eval()
            frame_dt = data.get("physical_dt") or 0.2
            metrics = evaluate_forecast(
                model,
                sde,
                val_traj,
                data["mean"],
                data["std"],
                args.stride,
                args.substeps,
                args.l_max,
                args.val_members,
                device,
            )
            metrics["step"] = step
            metrics["lead_time"] = [lead * frame_dt for lead in metrics["lead_frames"]]
            val_history.append(metrics)
            print(
                f"  val step {step}  stride={args.stride}  substeps={args.substeps}  "
                f"RMSE@first={metrics['rmse_ensemble_mean'][0]:.3f}  "
                f"RMSE@end={metrics['rmse_ensemble_mean'][-1]:.3f}  "
                f"CRPS@end={metrics['crps'][-1]:.3f}  "
                f"persist@end={metrics['rmse_persistence'][-1]:.3f}  "
                f"k>8 member@end={metrics['high_k_member'][-1]:.3f}  "
                f"k>8 truth@end={metrics['high_k_truth'][-1]:.3f}  "
                f"Z member@end={metrics['enstrophy_member'][-1]:.2f}  "
                f"Z truth@end={metrics['enstrophy_truth'][-1]:.2f}",
                flush=True,
            )
            model.train()
            (out_dir / "val_log.json").write_text(json.dumps(val_history))
        if args.ckpt_every > 0 and (step % args.ckpt_every == 0 or step == args.steps):
            save_checkpoint(out_dir / "ckpt.pt", model, data, args, step)

    ckpt_path = out_dir / "ckpt.pt"
    history = {
        "loss_first": logged[0] if logged else None,
        "loss_last": logged[-1] if logged else None,
        "loss_mean_last_20": float(np.mean(logged[-20:])) if logged else None,
        "logged_loss": logged,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seconds": time.perf_counter() - t_start,
        "ckpt": str(ckpt_path),
    }
    (out_dir / "train_log.json").write_text(json.dumps(history, indent=2))
    print(f"saved {ckpt_path}", flush=True)
    print(json.dumps(history), flush=True)


if __name__ == "__main__":
    main()
