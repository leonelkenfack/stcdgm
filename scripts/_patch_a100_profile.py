"""
Active le profil A100 80GB sur ``config/training_config.yaml`` :

- Architecture UNet musclée (4 niveaux, layers_per_block=2, channels [64,128,256,512])
- Encoder + RCN passent à hidden_dim=256, conditioning_dim=256
- num_dag_tokens 2 → 4
- diffusion.steps 12 → 1000 (training timesteps DDPM, le 12 était un bug)
- diffusion.eval_num_steps 12 → 30, cfg_scale 1.5 → 2.0
- diffusion.use_gradient_checkpointing true → false (A100 a la VRAM)
- attention_head_dim 32 → 64, norm_num_groups 8 → 32
- only_cross_attention plein-self [false×4]
- compile.diffusion_mode "default" → "max-autotune"
- physical_loss : use_predicted_output true, interval 10 → 4
- focal_gamma 1.5 → 2.5, contrastive_dag.interval 4 → 2
- training.epochs 20 (préservé), training.batch_size 48 (préservé)
- training.num_workers 2 → 8

Idempotent : chaque edit log son état (déjà fait / appliqué / non trouvé).
"""
from __future__ import annotations

import sys
from pathlib import Path

YAML = Path(__file__).resolve().parent.parent / "config" / "training_config.yaml"


# Liste d'edits : (description, ancien_block, nouveau_block).
# Le contexte de chaque ancien_block est suffisamment unique pour éviter les
# remplacements ambigus (encoder.hidden_dim vs rcn.hidden_dim par ex.).
EDITS: list[tuple[str, str, str]] = [
    # ── data ────────────────────────────────────────────────────────
    (
        "data.seq_len 8→16",
        "  seq_len: 8              # Contexte temporel suffisant pour météo",
        "  seq_len: 16             # A100 80GB : retour à 16 (était 8 pour OOM T4)",
    ),
    # Cas alternatif si le commentaire manque
    (
        "data.seq_len bare 8→16 (fallback)",
        "  seq_len: 8\n",
        "  seq_len: 16  # A100 : 8→16\n",
    ),

    # ── encoder ─────────────────────────────────────────────────────
    (
        "encoder.hidden_dim 128→256",
        "encoder:\n  hidden_dim: 128\n  conditioning_dim: 128",
        "encoder:\n  hidden_dim: 256                   # A100 : 128 → 256\n  conditioning_dim: 256              # A100 : 128 → 256",
    ),
    (
        "encoder.num_dag_tokens 2→4",
        "  num_dag_tokens: 2",
        "  num_dag_tokens: 4                  # A100 : 2 → 4 (DAG plus expressif)",
    ),

    # ── rcn ─────────────────────────────────────────────────────────
    (
        "rcn.hidden_dim 128→256",
        "rcn:\n  hidden_dim: 128",
        "rcn:\n  hidden_dim: 256                      # A100 : aligné avec encoder",
    ),

    # ── diffusion ───────────────────────────────────────────────────
    (
        "diffusion.steps 12→1000 (FIX bug)",
        "  steps: 12                # ↑ de 20 à 100 (qualité génération)",
        "  steps: 1000              # FIX : training timesteps DDPM (12 était un bug — c'est eval_num_steps qui doit être petit)",
    ),
    (
        "diffusion.eval_num_steps 12→30",
        "  eval_num_steps: 12            # 15 steps suffisent avec DPM-Solver++",
        "  eval_num_steps: 30            # A100 : 25-30 steps DPM-Solver++ pour qualité",
    ),
    (
        "diffusion.cfg_scale 1.5→2.0",
        "  cfg_scale: 1.5\n",
        "  cfg_scale: 2.0                # A100 : guidance plus forte\n",
    ),
    (
        "diffusion.use_gradient_checkpointing true→false",
        "  use_gradient_checkpointing: true  # OOM fix T4 : divise les activations UNet par ~2",
        "  use_gradient_checkpointing: false # A100 80GB : on a la VRAM, gain ~30% temps",
    ),

    # ── unet_kwargs ─────────────────────────────────────────────────
    (
        "unet_kwargs.layers_per_block 1→2",
        "  unet_kwargs:\n    layers_per_block: 1",
        "  unet_kwargs:\n    layers_per_block: 2                # A100 : 1 → 2",
    ),
    (
        "unet_kwargs.block_out_channels [32,64]→[64,128,256,512]",
        "    block_out_channels: [32, 64]",
        "    block_out_channels: [64, 128, 256, 512]   # A100 : 4 niveaux, ~50M params",
    ),
    (
        "unet_kwargs.down_block_types 2→4",
        "    down_block_types:\n      - \"DownBlock2D\"\n      - \"CrossAttnDownBlock2D\"\n    up_block_types:",
        "    down_block_types:\n      - \"DownBlock2D\"\n      - \"CrossAttnDownBlock2D\"\n      - \"CrossAttnDownBlock2D\"\n      - \"CrossAttnDownBlock2D\"\n    up_block_types:",
    ),
    (
        "unet_kwargs.up_block_types 2→4",
        "    up_block_types:\n      - \"CrossAttnUpBlock2D\"\n      - \"UpBlock2D\"\n    mid_block_type:",
        "    up_block_types:\n      - \"CrossAttnUpBlock2D\"\n      - \"CrossAttnUpBlock2D\"\n      - \"CrossAttnUpBlock2D\"\n      - \"UpBlock2D\"\n    mid_block_type:",
    ),
    (
        "unet_kwargs.norm_num_groups 8→32",
        "    norm_num_groups: 8",
        "    norm_num_groups: 32                # A100 : standard SD",
    ),
    (
        "unet_kwargs.attention_head_dim 32→64",
        "    attention_head_dim: 32",
        "    attention_head_dim: 64             # A100 : 32 → 64 (standard SD)",
    ),
    (
        "unet_kwargs.only_cross_attention pleins false",
        "    only_cross_attention: [false, true]",
        "    only_cross_attention: [false, false, false, false]  # A100 : self-attn complet à tous niveaux",
    ),
    (
        "unet_kwargs.projection_class_embeddings_input_dim",
        "    projection_class_embeddings_input_dim: 640  # 128 × 5 = 640",
        "    projection_class_embeddings_input_dim: 1536 # 256 × 6 = 1536 (auto-aligné par cell 34 quand même)",
    ),

    # ── loss ────────────────────────────────────────────────────────
    (
        "loss.lambda_precip_phy 0.05→0.10",
        "  lambda_precip_phy: 0.05",
        "  lambda_precip_phy: 0.10            # A100 : push physique plus fort",
    ),
    (
        "loss.precip_phy_weights quantile↑",
        "  precip_phy_weights: [1.0, 0.1, 0.2]",
        "  precip_phy_weights: [1.0, 0.1, 0.3]   # A100 : poids quantile ↑ pour extrêmes",
    ),
    (
        "loss.focal_gamma 1.5→2.5",
        "  focal_gamma: 1.5        # ↑ de 2.0 (plus focus sur pixels difficiles)",
        "  focal_gamma: 2.5        # A100 : push extrêmes (objectif F1_p95/p99)",
    ),
    (
        "loss.contrastive_dag.interval 4→2",
        "    interval: 4",
        "    interval: 2                       # A100 : signal DAG plus dense",
    ),

    # ── training ────────────────────────────────────────────────────
    # epochs et batch_size sont préservés à la demande de l'utilisateur.
    (
        "training.num_workers 2→8",
        "  num_workers: 2  # 0 pour CyVerse/Docker (évite /dev/shm limité)",
        "  num_workers: 8  # A100 host : 8-16 vCPU disponibles",
    ),
    (
        "training.compile.diffusion_mode default→max-autotune",
        "    diffusion_mode: \"default\"  # OOM fix T4 : reduce-overhead ⇒ CUDA graphs ⇒ mémoire figée",
        "    diffusion_mode: \"max-autotune\"   # A100 : max-autotune gain 10-20% (compile time 5-15 min)",
    ),
    (
        "training.physical_loss.use_predicted_output false→true",
        "    use_predicted_output: false",
        "    use_predicted_output: true        # A100 : sampling réel pour physical loss",
    ),
    (
        "training.physical_loss.physical_sample_interval 10→4",
        "    physical_sample_interval: 10",
        "    physical_sample_interval: 4       # A100 : physique plus dense",
    ),
    (
        "training.physical_loss.physical_num_steps 15→25",
        "    physical_num_steps: 15",
        "    physical_num_steps: 25            # A100 : sampling plus fidèle",
    ),

    # ── checkpoint (per-epoch persistence — préservé) ───────────────
    (
        "checkpoint.persist_every reste 1 (per-epoch obligatoire)",
        "  persist_every: 1",
        "  persist_every: 1                    # OBLIGATOIRE — per-epoch save sur Drive",
    ),
]


def main() -> int:
    src = YAML.read_text(encoding="utf-8")
    n_applied = 0
    n_already = 0
    n_missing = 0
    for desc, old, new in EDITS:
        if new in src and old not in src:
            print(f"  ✓ {desc}  (déjà appliqué)")
            n_already += 1
            continue
        if old in src:
            src = src.replace(old, new, 1)
            print(f"  ↪ {desc}")
            n_applied += 1
        else:
            print(f"  ⚠ {desc}  (pattern non trouvé — skip)")
            n_missing += 1

    YAML.write_text(src, encoding="utf-8")
    print(f"\n💾 YAML écrit : {n_applied} appliqué(s), {n_already} déjà OK, {n_missing} non trouvé(s)")
    return 0 if n_missing == 0 else 0  # non-bloquant


if __name__ == "__main__":
    sys.exit(main())
