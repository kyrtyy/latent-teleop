"""C1: action-conditioned latent predictor over frozen DINOv2 tokens, and a probe that reads the scene back out.

The predictor is a dynamics model in *robot time*: frame t plus the command executed during tick t -> frame t+1.
Network delay enters through which actions are fed and how many steps are rolled out (see README, "Where delay
enters"), not as an input to the network."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sim import AGENT, ANGLE, BLOCK


def amp():
    """bf16 autocast on Ampere (RTX 3050); plain fp32 on cards without native bf16 (GTX 1660 Super)."""
    return torch.autocast("cuda", torch.bfloat16, enabled=torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False))


def norm(z):
    return F.layer_norm(z.float(), z.shape[-1:])


def norm_xy(x):
    return (x - 256) / 128  # PushT pixels [0, 512] -> about [-2, 2]


def episode_split(profiles):
    """0 = test, 1 = val (later: conformal calibration), 2-9 = train. Split on the per-profile episode index:
    collect.py interleaves profiles, and splitting on the raw index would put only some profiles in each split."""
    n_prof = len(np.unique(profiles))
    split = (np.arange(len(profiles)) // n_prof) % 10
    for k in (0, 1):
        assert set(profiles[split == k]) == set(profiles), f"split {k} is missing a delay profile"
    return split


def probe_targets(states, contact):
    th = states[:, ANGLE:ANGLE + 1]
    return np.concatenate([norm_xy(states[:, AGENT]), norm_xy(states[:, BLOCK]), np.sin(th), np.cos(th),
                           contact[:, None]], 1).astype(np.float32)


def decode(p):
    """probe output (..., 7) -> agent xy px, block xy px, block angle rad, contact probability"""
    return p[..., 0:2] * 128 + 256, p[..., 2:4] * 128 + 256, torch.atan2(p[..., 4], p[..., 5]), torch.sigmoid(p[..., 6])


class Probe(nn.Module):
    """Spatial readout: per-patch scores and offsets, softmax-weighted patch centres -> agent and block position;
    block angle and contact from features pooled under the block's attention. Translation-equivariant, so it needs far
    less pose coverage than a flatten-MLP (probe_check.py: 6.4 vs 7.9 px block error on the diagnostic set)."""

    def __init__(self, n_tok, d_emb=384, h=256):
        super().__init__()
        P = int(round((n_tok - 1) ** 0.5))
        g = (torch.arange(P) + 0.5) / P
        self.register_buffer("c", torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).view(-1, 2))
        self.f = nn.Sequential(nn.Linear(d_emb, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.score, self.off, self.ang, self.contact = nn.Linear(h, 2), nn.Linear(h, 4), nn.Linear(h, 2), nn.Linear(h, 1)
        self.aff = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])  # image (row, col) -> workspace axes

    def forward(self, z):
        lead = z.shape[:-2]
        h = self.f(z.reshape(-1, *z.shape[-2:])[:, 1:])  # patch tokens only (CLS dropped)
        w = self.score(h).softmax(1)  # (B, N, 2): one attention map per object
        pts = self.c[None, :, None] + self.off(h).view(*h.shape[:2], 2, 2) / 4
        pos = torch.einsum("bnk,bnkc->bkc", w, pts)
        hb = torch.einsum("bn,bnh->bh", w[..., 1], h)
        out = torch.cat([self.aff[0](pos[:, 0]), self.aff[1](pos[:, 1]), self.ang(hb), self.contact(hb)], -1)
        return out.view(*lead, 7)


def probe_loss(out, tgt):
    return F.mse_loss(out[..., :6], tgt[..., :6]) + 0.1 * F.binary_cross_entropy_with_logits(out[..., 6], tgt[..., 6])


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask, cache=None):
        B, N, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        if cache is not None:  # rollout: earlier ticks never change (block-causal), so reuse their keys/values
            if cache:
                k, v = torch.cat([cache[0], k], 2), torch.cat([cache[1], v], 2)
            cache[:] = [k, v]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(y.transpose(1, 2).reshape(B, N, d))
        return x + self.mlp(self.n2(x))


class Predictor(nn.Module):
    """Each tick is a block of frame tokens + one action token. Attention is block-causal (a token sees its own tick
    and earlier ones), so actions stay in order. Outputs at tick t's frame tokens predict tick t+1's frame tokens."""

    def __init__(self, n_tok, d_emb=384, d=256, depth=6, heads=8, max_t=32):
        super().__init__()
        self.n_tok = n_tok
        self.inp, self.act, self.out = nn.Linear(d_emb, d), nn.Linear(2, d), nn.Linear(d, d_emb)
        self.tok_pos = nn.Parameter(torch.randn(n_tok + 1, d) * 0.02)
        self.time_pos = nn.Parameter(torch.randn(max_t, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(d)

    def forward(self, z, a, t0=0, caches=None):
        """z (B, T, n, D) normalized frame tokens for ticks t0..t0+T-1, a (B, T, 2) normalized commands
        -> (B, T, n, D) next-frame tokens. With caches (one list per block), earlier ticks come from the cache."""
        B, T, n, _ = z.shape
        x = torch.cat([self.inp(z), self.act(a)[:, :, None]], 2) + self.tok_pos + self.time_pos[t0:t0 + T, None]
        q_tick = torch.arange(t0, t0 + T, device=z.device).repeat_interleave(n + 1)
        k_tick = torch.arange(0 if caches is not None else t0, t0 + T, device=z.device).repeat_interleave(n + 1)
        mask = k_tick[None, :] <= q_tick[:, None]  # True = may attend
        x = x.flatten(1, 2)
        for i, blk in enumerate(self.blocks):
            x = blk(x, mask, None if caches is None else caches[i])
        return self.out(self.norm(x).view(B, T, n + 1, -1)[:, :, :n])


def rollout(model, z_ctx, a, k):
    """z_ctx (B, H, n, D) true frames up to the anchor; a (B, >= H+k-1, 2) commands executed at each tick.
    Rolls k ticks past the anchor, feeding predictions back in. Returns (B, k, n, D).
    Past the trained horizon (max_t), the rollout is chained: the newest H frames (predicted) become the next context."""
    H, seg = z_ctx.shape[1], model.time_pos.shape[0] - z_ctx.shape[1] + 1
    if k <= seg:
        return _rollout(model, z_ctx, a, k)
    first = _rollout(model, z_ctx, a, seg)
    ctx = torch.cat([z_ctx, first], 1)[:, -H:]
    return torch.cat([first, rollout(model, ctx, a[:, seg:], k - seg)], 1)


def _rollout(model, z_ctx, a, k):
    H = z_ctx.shape[1]
    caches = [[] for _ in model.blocks]
    nxt = model(z_ctx, a[:, :H], 0, caches)[:, -1:]
    preds = [nxt]
    for t in range(H, H + k - 1):
        nxt = model(nxt, a[:, t:t + 1], t, caches)
        preds.append(nxt)
    return torch.cat(preds, 1)


if __name__ == "__main__":
    # cached rollout must equal naive rollout (re-running the whole sequence every step)
    torch.manual_seed(0)
    m = Predictor(5, max_t=12).eval()
    z, a = torch.randn(3, 4, 5, 384), torch.randn(3, 11, 2)
    with torch.no_grad():
        zs = z
        for _ in range(8):
            zs = torch.cat([zs, m(zs, a[:, :zs.shape[1]])[:, -1:]], 1)
        assert torch.allclose(rollout(m, z, a, 8), zs[:, 4:], atol=1e-4), "KV-cached rollout differs from naive"
        # chained past max_t: same as rolling the first segment, then restarting from the newest 4 frames
        a = torch.randn(3, 40, 2)
        long, first = rollout(m, z, a, 20), rollout(m, z, a, 9)
        assert torch.allclose(long[:, :9], first, atol=1e-4)
        assert torch.allclose(long[:, 9:18], rollout(m, torch.cat([z, first], 1)[:, -4:], a[:, 9:], 9), atol=1e-4)
    print("world_model.py checks passed")
