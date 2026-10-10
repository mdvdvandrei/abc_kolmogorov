"""Small conditional U-Net for the ABC path-dependent score.

FiLM comes from the average of independent embeddings of t and t_next.
Conditioning frames are encoded to tokens, tagged with their own time, and
read by cross-attention at the bottleneck. Invalid slots are masked.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding. `t` is rescaled by 1000, matching the paper."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args = t.flatten().to(dtype=t.dtype)[:, None] * 1000.0 * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return emb.view(*t.shape, dim)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_ch: int):
        super().__init__()
        groups_in = min(8, in_ch)
        groups_out = min(8, out_ch)
        self.norm1 = nn.GroupNorm(groups_in, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="circular")
        self.norm2 = nn.GroupNorm(groups_out, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="circular")
        self.emb = nn.Linear(emb_ch, out_ch * 2)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb)).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        """query (B, Nq, D), context (B, Nk, D), context_mask (B, Nk) with 1 = valid."""
        b, nq, dim = query.shape
        nk = context.shape[1]
        q = self.q(self.norm_q(query)).view(b, nq, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(self.norm_kv(context)).chunk(2, dim=-1)
        k = k.view(b, nk, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, nk, self.n_heads, self.head_dim).transpose(1, 2)
        mask = context_mask[:, None, None, :].to(dtype=torch.bool).expand(b, self.n_heads, nq, nk)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = torch.nan_to_num(out, nan=0.0)
        valid_any = context_mask.reshape(b, -1).any(dim=-1).to(dtype=out.dtype)
        out = out * valid_any[:, None, None, None]
        out = out.transpose(1, 2).reshape(b, nq, dim)
        return self.proj(out)


class CondEncoder(nn.Module):
    def __init__(self, sin_dim: int, out_ch: int):
        super().__init__()
        self.sin_dim = sin_dim
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv2d(64, out_ch, 4, stride=2, padding=1, padding_mode="circular"),
        )
        self.time_proj = nn.Linear(sin_dim, out_ch)

    def forward(self, frames: torch.Tensor, times: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """frames (B, L, 1, H, W), times (B, L) -> tokens (B, L*h*w, D) and spatial size."""
        b, length, channels, height, width = frames.shape
        flat = frames.reshape(b * length, channels, height, width)
        if flat.is_cuda:
            flat = flat.contiguous(memory_format=torch.channels_last)
        h = self.net(flat)
        _, dim, gh, gw = h.shape
        h = h.view(b, length, dim, gh, gw)
        temb = sinusoidal_embedding(times, self.sin_dim)
        temb = self.time_proj(temb).view(b, length, dim, 1, 1)
        h = h + temb
        tokens = h.permute(0, 1, 3, 4, 2).reshape(b, length * gh * gw, dim)
        return tokens, gh, gw


def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.SiLU(),
        nn.Linear(out_dim, out_dim),
    )


class ABCUNet(nn.Module):
    def __init__(self, base: int = 32, sin_dim: int = 128, n_heads: int = 4):
        super().__init__()
        self.sin_dim = sin_dim
        emb_ch = sin_dim * 4
        c1, c2, c3 = base, base * 2, base * 4
        self.t_mlp = _mlp(sin_dim, emb_ch)
        self.tn_mlp = _mlp(sin_dim, emb_ch)

        self.conv_in = nn.Conv2d(1, c1, 3, padding=1, padding_mode="circular")
        self.rb0 = ResBlock(c1, c1, emb_ch)
        self.down1 = nn.Conv2d(c1, c1, 4, stride=2, padding=1, padding_mode="circular")
        self.rb1 = ResBlock(c1, c2, emb_ch)
        self.down2 = nn.Conv2d(c2, c2, 4, stride=2, padding=1, padding_mode="circular")
        self.rb2 = ResBlock(c2, c3, emb_ch)
        self.down3 = nn.Conv2d(c3, c3, 4, stride=2, padding=1, padding_mode="circular")
        self.mid1 = ResBlock(c3, c3, emb_ch)
        self.cond_encoder = CondEncoder(sin_dim, c3)
        self.cross = CrossAttention(c3, n_heads=n_heads)
        self.mid2 = ResBlock(c3, c3, emb_ch)

        self.up3 = nn.Conv2d(c3, c3, 3, padding=1, padding_mode="circular")
        self.rb_up2 = ResBlock(c3 + c3, c2, emb_ch)
        self.up2 = nn.Conv2d(c2, c2, 3, padding=1, padding_mode="circular")
        self.rb_up1 = ResBlock(c2 + c2, c1, emb_ch)
        self.up1 = nn.Conv2d(c1, c1, 3, padding=1, padding_mode="circular")
        self.rb_up0 = ResBlock(c1 + c1, c1, emb_ch)
        self.out_norm = nn.GroupNorm(min(8, c1), c1)
        self.out_conv = nn.Conv2d(c1, 1, 3, padding=1, padding_mode="circular")
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _time_emb(self, t: torch.Tensor, t_next: torch.Tensor) -> torch.Tensor:
        et = self.t_mlp(sinusoidal_embedding(t, self.sin_dim))
        en = self.tn_mlp(sinusoidal_embedding(t_next, self.sin_dim))
        return 0.5 * (et + en)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        cond: torch.Tensor,
        cond_times: torch.Tensor,
        cond_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the residual score.

        x: (B, 1, H, W)
        t, t_next: (B,)
        cond: (B, L, 1, H, W)
        cond_times, cond_mask: (B, L)
        """
        emb = self._time_emb(t, t_next)
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        h = self.conv_in(x)
        s0 = self.rb0(h, emb)
        h = self.rb1(self.down1(s0), emb)
        s1 = h
        h = self.rb2(self.down2(h), emb)
        s2 = h
        h = self.mid1(self.down3(h), emb)

        ctx, gh, gw = self.cond_encoder(cond, cond_times)
        token_mask = (
            cond_mask[:, :, None]
            .expand(-1, -1, gh * gw)
            .reshape(cond.shape[0], cond.shape[1] * gh * gw)
        )
        h_shape = h.shape
        tokens = h.flatten(2).transpose(1, 2)
        tokens = tokens + self.cross(tokens, ctx, token_mask)
        h = tokens.transpose(1, 2).reshape(h_shape)

        h = self.mid2(h, emb)
        h = F.interpolate(h, size=s2.shape[-2:], mode="nearest")
        h = self.rb_up2(torch.cat([self.up3(h), s2], dim=1), emb)
        h = F.interpolate(h, size=s1.shape[-2:], mode="nearest")
        h = self.rb_up1(torch.cat([self.up2(h), s1], dim=1), emb)
        h = F.interpolate(h, size=s0.shape[-2:], mode="nearest")
        h = self.rb_up0(torch.cat([self.up1(h), s0], dim=1), emb)
        return self.out_conv(F.silu(self.out_norm(h)))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
