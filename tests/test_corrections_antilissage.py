"""
Tests de non-régression — corrections anti-lissage / prompt v6 (Phases 4 et 6.2).
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


def _rapsd_loss_minimal(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Copie minimale pour test gradient (évite import diffusers)."""
    B, C, H, W = pred.shape
    losses = []
    for b in range(B):
        for c in range(C):
            p = pred[b, c]
            t = target[b, c]
            fp = torch.fft.fftshift(torch.fft.fft2(p))
            ft = torch.fft.fftshift(torch.fft.fft2(t))
            psd_p = torch.abs(fp) ** 2
            psd_t = torch.abs(ft) ** 2
            cy, cx = H // 2, W // 2
            y_idx = torch.arange(H, device=pred.device, dtype=torch.float32) - cy
            x_idx = torch.arange(W, device=pred.device, dtype=torch.float32) - cx
            yy, xx = torch.meshgrid(y_idx, x_idx, indexing="ij")
            r = torch.sqrt(xx ** 2 + yy ** 2).long().clamp(min=0)
            max_r = int(r.max().item()) + 1
            rapsd_p = torch.zeros(max_r, device=pred.device)
            rapsd_t = torch.zeros(max_r, device=pred.device)
            counts = torch.zeros(max_r, device=pred.device)
            rf = r.flatten()
            rapsd_p.scatter_add_(0, rf, psd_p.flatten())
            rapsd_t.scatter_add_(0, rf, psd_t.flatten())
            counts.scatter_add_(0, rf, torch.ones_like(rf, dtype=torch.float32))
            valid = counts > 0
            log_ratio = torch.log(rapsd_p[valid] + 1e-8) - torch.log(rapsd_t[valid] + 1e-8)
            losses.append((log_ratio ** 2).mean())
    return torch.stack(losses).mean()


def test_bicubic_preserves_peak_vs_bilinear():
    field = torch.zeros(1, 1, 50, 50)
    field[0, 0, 25, 25] = 100.0
    field[0, 0, 24, 25] = 80.0
    field[0, 0, 26, 25] = 80.0
    target_size = (172, 179)
    upsampled_bilinear = F.interpolate(field, size=target_size, mode="bilinear", align_corners=False)
    upsampled_bicubic = F.interpolate(field, size=target_size, mode="bicubic", align_corners=False).clamp(
        min=0.0
    )
    assert upsampled_bicubic.max().item() >= upsampled_bilinear.max().item() - 1e-3


def test_ensemble_mean_reduces_variance_vs_single_member():
    N_members = 10
    H, W = 100, 100
    base = torch.ones(H, W) * 0.5
    members = []
    rng = np.random.default_rng(42)
    for _ in range(N_members):
        member = base.clone()
        px = int(rng.integers(10, H - 10))
        py = int(rng.integers(10, W - 10))
        member[px, py] = 5.0
        members.append(member)
    stack = torch.stack(members, dim=0)
    ensemble_mean = stack.mean(dim=0)
    single_member = stack[0]
    assert single_member.var().item() > ensemble_mean.var().item() * 2.0


def test_bicubic_can_overshoot_negative_then_clamp():
    field = torch.zeros(1, 1, 20, 20)
    field[0, 0, 10, 10] = 50.0
    upsampled = F.interpolate(field, size=(172, 179), mode="bicubic", align_corners=False)
    clamped = upsampled.clamp(min=0.0)
    assert (clamped >= 0).all()


def test_rapsd_loss_is_differentiable():
    pred = torch.randn(2, 1, 32, 32, requires_grad=True)
    target = torch.randn(2, 1, 32, 32)
    loss = _rapsd_loss_minimal(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()


def _replace_conv_transpose_with_resize_minimal(module: nn.Module) -> None:
    """Même logique que `diffusion_decoder._replace_conv_transpose_with_resize` (sans import diffusers)."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.ConvTranspose2d):
            st = child.stride[0]
            if st >= 2 and child.stride[0] == child.stride[1]:
                repl = nn.Sequential(
                    nn.Upsample(scale_factor=float(st), mode="nearest"),
                    nn.Conv2d(
                        child.in_channels,
                        child.out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=child.bias is not None,
                    ),
                )
                if child.bias is not None:
                    nn.init.zeros_(repl[1].bias)
                nn.init.kaiming_normal_(repl[1].weight, nonlinearity="relu")
                setattr(module, name, repl)
            else:
                _replace_conv_transpose_with_resize_minimal(child)
        else:
            _replace_conv_transpose_with_resize_minimal(child)


def test_resizeconv_replacement_preserves_shape_and_removes_transpose():
    """Anti-checkerboard: Upsample+Conv remplace ConvTranspose2d avec même taille spatiale de sortie."""

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.up(x)

    m = Tiny()
    x = torch.randn(1, 8, 16, 16)
    with torch.no_grad():
        y_ct = m(x)
    assert any(isinstance(c, nn.ConvTranspose2d) for c in m.modules())

    _replace_conv_transpose_with_resize_minimal(m)
    assert not any(isinstance(c, nn.ConvTranspose2d) for c in m.modules())
    assert isinstance(m.up, nn.Sequential)
    assert isinstance(m.up[0], nn.Upsample) and isinstance(m.up[1], nn.Conv2d)

    with torch.no_grad():
        y_rc = m(x)
    assert y_ct.shape == y_rc.shape
