# >>> BS31b_CONFIG_SELECT
# Configuration loader with optional override-merge.
#
# Pick which YAML governs the run:
#   1. CONFIG_FILE env var if set → that file under ``config/``
#   2. else, ``training_config_corrdiff_mini.yaml`` if present
#      (BS31b CorrDiff-Mini override, ~12-15M params Stage 2 UNet)
#   3. fallback ``training_config.yaml`` (base, ~1-2M Stage 2 UNet)
#
# When an override is selected, it is *merged on top of* the base, so
# only the diffs need to be in the override file. See
# ``architecture_journey.md §6`` for context.
import os
from pathlib import Path
from omegaconf import OmegaConf

_base_path = Path("config/training_config.yaml")
# BS43 NON-CAUSAL TRAINING NOTEBOOK — defaut = noncausal (CorrDiff vanilla) (= le "v4", Pearson 0.815,
# checkpoint ckpt_v2_corrdiff_normal). Meilleur modele a ce jour ; v5 (FACL+SW+
# EMA+cond_drop, ckpt_v5_pearson_090) n'a PAS battu v4 -> abandonne.
# Colab : pour evaluer v4 sans reentrainement, garder FORCE_S2_RESTART=False ;
# le cache BS32B auto-detecte stage1_cache_v2.pt via save_dir=ckpt_v2_corrdiff_normal.
# Override via env : os.environ['CONFIG_FILE'] = 'training_config_v3.yaml'
_default_override = "training_config_noncausal.yaml"
_override_name = os.environ.get("CONFIG_FILE", _default_override)
_override_path = Path("config") / _override_name

CONFIG = OmegaConf.load(_base_path)

if _override_path.exists() and _override_path.resolve() != _base_path.resolve():
    _override = OmegaConf.load(_override_path)
    CONFIG = OmegaConf.merge(CONFIG, _override)
    print(f"📂 Config base + override : training_config.yaml + {_override_path.name}")
elif _override_path.exists():
    print(f"📂 Config (single file)    : {_override_path.name}")
else:
    print(f"📂 Config (no override)    : training_config.yaml")
    print(f"   (override {_override_path.name} introuvable — utilisé si placé dans config/)")

print(f"  - Device: {CONFIG.training.device}")
print(f"  - Epochs: {CONFIG.training.epochs}")
print(f"  - Lambda gen: {CONFIG.loss.lambda_gen}, Beta rec: {CONFIG.loss.beta_rec}, Gamma DAG: {CONFIG.loss.gamma_dag}")
print(f"  - Stage 2 UNet block_out_channels: {list(CONFIG.diffusion.unet_kwargs.block_out_channels)}")
