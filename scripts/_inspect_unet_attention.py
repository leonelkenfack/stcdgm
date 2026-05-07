"""Build the CorrDiff-Mini UNet locally and inspect every attention layer
to find which spatial resolution the attention is actually computed at.

Goal: confirm or refute the hypothesis that self-attention runs at the
FULL 172×179 resolution (matches the 457 GiB OOM).
"""
from __future__ import annotations

import torch
from diffusers import UNet2DConditionModel


def build_unet():
    """Post-BS32d CorrDiff Mini config — attn at smallest level only,
    explicit num_attention_heads to neutralize the diffusers attention_head_dim trap."""
    return UNet2DConditionModel(
        sample_size=(172, 179),
        in_channels=3,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 128),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",                 # BS32d : was CrossAttn → no attn at 86×90
            "CrossAttnDownBlock2D",        # attn only at smallest 43×45
        ),
        up_block_types=(
            "CrossAttnUpBlock2D",          # attn at 43×45
            "UpBlock2D",                   # BS32d : was CrossAttn → no attn at 86×90
            "UpBlock2D",
        ),
        mid_block_type="UNetMidBlock2D",
        cross_attention_dim=128,
        norm_num_groups=32,
        attention_head_dim=4,              # BS32d : in diffusers ce param = num_heads (issue #2011)
        only_cross_attention=[False, False, False],
        class_embed_type="projection",
        projection_class_embeddings_input_dim=768,
    )


def main():
    print("Building UNet (CorrDiff Mini config)...")
    unet = build_unet()
    n_params = sum(p.numel() for p in unet.parameters())
    print(f"  total params : {n_params/1e6:.2f} M")
    print()

    # Count attention layers and report their channel dim and likely resolution.
    print("Attention layers found in module tree :")
    n_attn = 0
    for name, module in unet.named_modules():
        # Diffusers Attention is named ``Attention`` (with attn1/attn2 inside Transformer2DModel).
        cls = type(module).__name__
        if cls == "Attention":
            # heads_dim_per_head = module.dim_head if hasattr(module, "dim_head") else None
            h = getattr(module, "heads", None)
            inner_dim = getattr(module, "inner_dim", None)
            print(f"  {name:80s} | heads={h} inner_dim={inner_dim}")
            n_attn += 1
    print(f"\n  total Attention modules : {n_attn}")
    print()

    # Trace through a forward to see actual spatial sizes at each block.
    # Hook every Transformer2DModel input to print its hidden_states.shape.
    print("Forward pass with bs=2 to trace tensor shapes :")
    sample = torch.randn(2, 3, 172, 179)
    timestep = torch.tensor([100, 200], dtype=torch.long)
    enc_hidden = torch.zeros(2, 1, 128)
    class_labels = torch.zeros(2, 768)

    hooks = []
    seen_shapes = []

    def make_hook(name):
        def hook(module, inp, out):
            try:
                s = inp[0].shape
                seen_shapes.append((name, tuple(s)))
            except Exception:
                pass
        return hook

    for name, module in unet.named_modules():
        if type(module).__name__ in ("Transformer2DModel",):
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        try:
            _ = unet(sample, timestep, encoder_hidden_states=enc_hidden, class_labels=class_labels).sample
        except Exception as e:
            print(f"  forward failed (OK on CPU for big shapes): {type(e).__name__}: {e}")
    for h in hooks:
        h.remove()

    print(f"  Transformer2DModel inputs (B, C, H, W) :")
    for name, s in seen_shapes:
        print(f"    {name:60s} | {s}")

    # Estimate attention matrix size at each transformer
    print()
    print(f"At batch_size=64, bf16 attention matrix size per Transformer2DModel :")
    for name, s in seen_shapes:
        if len(s) == 4:
            B, C, H, W = s
            N = H * W
            heads = C // 32  # attention_head_dim=32
            mem = 64 * heads * N * N * 2
            print(f"    {name:60s} | N={N} ({H}×{W}), heads={heads}, attn={mem/1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
