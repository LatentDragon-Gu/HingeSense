# -*- coding: utf-8 -*-
"""模型：DriftNet（通道级漂移修正）→ 共享嵌入头 → 门控融合 → soft-kNN 读出。

子模块的构造顺序（heads → fuse → logtau → fusegate → drift）决定参数初始化消耗随机数的顺序，
决定结果的确定性，不可调整。

Model: DriftNet (channel-wise drift correction) → shared embedding head → gated fusion → soft-kNN read-out.
The construction order of the submodules (heads → fuse → logtau → fusegate → drift) fixes the order in which parameter
initialisation consumes random numbers and therefore the results; it must not be changed.
"""
import torch
from torch import nn

from . import config as C


class MlpHead(nn.Module):
    """嵌入头：当前帧的 raw 通路（n 维）→ LAT 维嵌入，末端 LayerNorm。 / Embedding head: raw pathway of the current frame (n-D) → LAT-D embedding with a final LayerNorm."""

    def __init__(self, in_dim, lat=C.LAT):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, C.HEAD_HID), nn.GELU(), nn.Linear(C.HEAD_HID, lat))
        self.ln = nn.LayerNorm(lat)

    def forward(self, xw):  # xw: [B,1,n]
        return self.ln(self.net(xw[:, -1, :]))


class DriftNet(nn.Module):
    """自零漂移修正网络：由当前帧的 raw+ccn 通路估计每通道乘性增益与加性修正。

    输入按 cat[x, 0, x, x, x] 展开为 2n×POOLN 维；加性头、增益头与差分投影零初始化，
    训练起点为恒等映射。差分通路（diff16|diff64）经零初始化线性层加到隐状态。

    Zero-initialised drift-correction network: estimates a per-channel multiplicative gain and additive correction from the
    raw+ccn pathways of the current frame. The input is expanded as cat[x, 0, x, x, x] to 2n×POOLN dimensions; the additive head,
    the gain head and the differential projection start at zero, so training starts from the identity. The differential pathways
    (diff16|diff64) enter the hidden state through a zero-initialised linear layer.
    """

    def __init__(self, n_channel=C.N_CHANNEL, hid=C.DRIFT_HID):
        super().__init__()
        self.rawccn = 2 * n_channel
        in_dim = self.rawccn * C.POOLN
        self.enc = nn.Sequential(nn.Linear(in_dim, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, self.rawccn))
        self.delta = nn.Linear(hid, self.rawccn)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        self.gainh = nn.Linear(hid, self.rawccn)
        nn.init.zeros_(self.gainh.weight)
        nn.init.zeros_(self.gainh.bias)
        self.diffproj = nn.Linear(self.rawccn, hid)
        nn.init.zeros_(self.diffproj.weight)
        nn.init.zeros_(self.diffproj.bias)

    def forward(self, xw):  # xw: [B,1,4n]
        x = xw[:, -1, 0:self.rawccn]
        z = torch.zeros_like(x)
        feats = torch.cat([x, z, x, x, x], -1)
        h = self.enc(feats)
        h = h + self.diffproj(xw[:, -1, self.rawccn:])
        g = torch.sigmoid(self.gate(h))
        d = self.delta(h)
        gain = 1.0 + torch.tanh(self.gainh(h))
        return g * d, gain


class HingeNet(nn.Module):
    """完整解码网络。encode() 输出 [B,1,FUSE_OUT] 嵌入与 raw 臂嵌入；角度由 soft_readout 读出。

    drift_correction=False 时把修正固定为恒等（消融 "w/o drift-correction network"）。

    The complete decoder. encode() returns the [B,1,FUSE_OUT] embedding and the raw-arm embeddings; the angle is read out by
    soft_readout. drift_correction=False fixes the correction to the identity (ablation "w/o drift-correction network").
    """

    def __init__(self, n_channel=C.N_CHANNEL, drift_correction=True):
        super().__init__()
        self.n_channel = n_channel
        self.rawccn = 2 * n_channel
        self.drift_correction = drift_correction
        self.heads = nn.ModuleList([MlpHead(n_channel)])
        self.fuse = nn.Sequential(nn.Linear(C.LAT, C.FUSE_HID), nn.GELU(), nn.Linear(C.FUSE_HID, C.FUSE_OUT))
        self.logtau = nn.Parameter(torch.tensor(0.0))
        self.fusegate = nn.Linear(2 * C.LAT, C.LAT)
        self.drift = DriftNet(n_channel)

    def encode(self, xw):
        corr, gain = self.drift(xw)
        if not self.drift_correction:
            corr = corr * 0.0
            gain = gain * 0.0 + 1.0
        base = xw[:, -1, 0:self.rawccn]
        xw_corr = xw.clone()
        xw_corr[:, -1, 0:self.rawccn] = gain * base + corr

        def arms(xin):
            zs = [self.heads[0](xin[:, :, 0:self.n_channel])]
            return torch.cat(zs, 1), zs

        cat_corr, zs = arms(xw_corr)
        cat_raw, _ = arms(xw)
        gf = torch.sigmoid(self.fusegate(torch.cat([cat_raw, cat_corr], -1)))
        fused = gf * cat_corr + (1.0 - gf) * cat_raw
        return self.fuse(fused)[:, None, :], zs


def soft_readout(zq, zs, ys, logtau):
    """soft-kNN 读出：query 与锚嵌入的平方距离经 softmax(−d²/τ) 加权锚角度。 / soft-kNN read-out: anchor angles weighted by softmax(−d²/τ) of the squared query–anchor distances."""
    d2 = torch.cdist(zq, zs) ** 2
    tau = torch.exp(logtau) + 1e-3
    return torch.softmax(-d2 / tau, 1) @ ys


def count_parameters(model):
    """可训练参数总数。 / Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
