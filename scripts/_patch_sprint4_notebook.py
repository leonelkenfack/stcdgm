"""Sprint 4 patches for the training notebook.

Edits two cells:
  - Cell with id ``447179fa`` — model construction. Switch the hardcoded
    UNet kwargs to read ``CONFIG.diffusion.unet_kwargs``, add the
    ``HRTargetIdentifiabilityHead`` creation, and include it in the
    optimizer.
  - Cell with id ``d1f463ad`` — training loop. Plumb the new
    ``lambda_contrastive_dag`` / ``contrastive_dag_margin`` /
    ``contrastive_dag_interval`` knobs into ``train_epoch``, plus the
    pre-existing ``hr_ident_head`` / ``beta_hr_ident`` / ``lambda_precip_phy``
    / ``precip_phy_weights`` / ``physical_sample_interval`` knobs that
    were silently missing.

Run from the repo root:

    python scripts/_patch_sprint4_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NB = Path("st_cdgm_training_evaluation.ipynb")

CELL_MODELS = "447179fa"
CELL_TRAIN = "d1f463ad"

NEW_MODELS_SOURCE = '''# 1. Intelligible Variable Encoder (metapaths depuis CONFIG.encoder.metapaths)
# Ne garder que les metapaths dont les nœuds existent dans le graphe (aligné avec include_mid_layer)
allowed_nodes = set(builder.dynamic_node_types) | set(builder.static_node_types)
encoder_configs = [
    IntelligibleVariableConfig(
        name=mp.name,
        meta_path=(mp.src, mp.relation, mp.target),
        pool=mp.get("pool", "mean"),
    )
    for mp in CONFIG.encoder.metapaths
    if mp.src in allowed_nodes and mp.target in allowed_nodes
]
if pipeline.get_static_dataset() is not None:
    encoder_configs.append(
        IntelligibleVariableConfig(
            name="static",
            meta_path=("SP_HR", "causes", "GP850"),
            pool="mean"
        )
    )

encoder = IntelligibleVariableEncoder(
    configs=encoder_configs,
    hidden_dim=HIDDEN_DIM,
    conditioning_dim=CONDITIONING_DIM,
).to(DEVICE)

num_vars = len(encoder_configs)
print(f"✅ Encoder créé avec {num_vars} variables intelligibles")

# 2. Causal RCN
# reconstruction_dim doit égaler driver_dim pour que L_rec compare recon et driver (même forme)
rcn_cell = RCNCell(
    num_vars=num_vars,
    hidden_dim=CONFIG.rcn.hidden_dim,
    driver_dim=RCN_DRIVER_DIM,
    reconstruction_dim=RCN_DRIVER_DIM,
    dropout=CONFIG.rcn.dropout,
).to(DEVICE)

rcn_runner = RCNSequenceRunner(rcn_cell, detach_interval=CONFIG.rcn.get("detach_interval"))
print(f"✅ RCN créé")

# 3. Diffusion Decoder — Sprint 4: read UNet kwargs from CONFIG so the
# rewiring (cross-attention at every resolution level + UNetMidBlock2DCrossAttn)
# is actually picked up. Previously this cell hardcoded a 2-block UNet
# with cross-attention on only one down + one up block, which silently
# overrode the YAML and left the bottleneck blind to the DAG tokens —
# the root cause of the "DAG non-conditioning" intervention test.
hr_channels = sample['residual'].shape[1]

from omegaconf import OmegaConf as _OC
UNET_KWARGS = _OC.to_container(CONFIG.diffusion.unet_kwargs, resolve=True)
for _k in ("down_block_types", "up_block_types"):
    if _k in UNET_KWARGS and isinstance(UNET_KWARGS[_k], list):
        UNET_KWARGS[_k] = tuple(UNET_KWARGS[_k])
# Auto-align class_embeddings input dim with the real num_vars × conditioning_dim
# so changing the metapath list in YAML doesn't require editing the projection
# dim by hand.
UNET_KWARGS["projection_class_embeddings_input_dim"] = (
    num_vars * CONFIG.diffusion.conditioning_dim
)
print(
    f"🧠 UNet kwargs (from CONFIG.diffusion.unet_kwargs):\\n"
    f"   - down_block_types = {UNET_KWARGS.get('down_block_types')}\\n"
    f"   - up_block_types   = {UNET_KWARGS.get('up_block_types')}\\n"
    f"   - mid_block_type   = {UNET_KWARGS.get('mid_block_type')}\\n"
    f"   - projection_class_embeddings_input_dim = {UNET_KWARGS['projection_class_embeddings_input_dim']}"
)

diffusion = CausalDiffusionDecoder(
    in_channels=hr_channels,
    conditioning_dim=CONFIG.diffusion.conditioning_dim,
    height=CONFIG.diffusion.height,
    width=CONFIG.diffusion.width,
    num_diffusion_steps=CONFIG.diffusion.steps,
    unet_kwargs=UNET_KWARGS,
).to(DEVICE)

print(f"✅ Diffusion decoder créé avec {CONFIG.diffusion.steps} steps")

# 4. Sprint 2: CausalConditioningProjector (DAG tokens → UNet cross-attention)
from st_cdgm.models.intelligible_encoder import (
    SpatialConditioningProjector,
    CausalConditioningProjector,
    HRTargetIdentifiabilityHead,
)

use_causal_proj = bool(CONFIG.encoder.get("causal_conditioning", False))
_spatial_target_shape = tuple(CONFIG.diffusion.get("spatial_target_shape", [6, 7]))

if use_causal_proj:
    spatial_projector = CausalConditioningProjector(
        num_vars=num_vars,
        hidden_dim=CONFIG.rcn.hidden_dim,
        conditioning_dim=CONFIG.diffusion.conditioning_dim,
        lr_shape=lr_shape,
        target_shape=_spatial_target_shape,
        num_dag_tokens=int(CONFIG.encoder.get("num_dag_tokens", 1)),
    ).to(DEVICE)
    print(f"✅ CausalConditioningProjector créé (num_dag_tokens={CONFIG.encoder.get('num_dag_tokens', 1)})")
else:
    spatial_projector = SpatialConditioningProjector(
        num_vars=num_vars,
        hidden_dim=CONFIG.rcn.hidden_dim,
        conditioning_dim=CONFIG.diffusion.conditioning_dim,
        lr_shape=lr_shape,
        target_shape=_spatial_target_shape,
    ).to(DEVICE)
    print("✅ SpatialConditioningProjector créé (Sprint 1 fallback)")

# 4b. Sprint 2: HR-target identifiability head.
# Predicts summary statistics of the HR field from the pooled causal state,
# so gradient through L_hr_ident forces A_dag to carry information that
# *actually matters* for HR precipitation (not just LR driver recon).
_hr_ident_cfg = CONFIG.loss.get("hr_ident", {}) or {}
_hr_ident_enabled = bool(_hr_ident_cfg.get("enabled", False))
_beta_hr_ident = float(_hr_ident_cfg.get("beta", 0.0))
if _hr_ident_enabled and _beta_hr_ident > 0.0:
    hr_ident_head = HRTargetIdentifiabilityHead(
        num_vars=num_vars,
        hidden_dim=CONFIG.rcn.hidden_dim,
        stats=list(_hr_ident_cfg.get("stats", ["mean", "std", "p95", "p99"])),
    ).to(DEVICE)
    print(
        f"🎯 Sprint 2: HR identifiability head enabled "
        f"(beta={_beta_hr_ident}, stats={hr_ident_head.stats})"
    )
else:
    hr_ident_head = None

# 5. Optimizer — includes spatial_projector + HR ident head parameters
params = (
    list(encoder.parameters())
    + list(rcn_cell.parameters())
    + list(diffusion.parameters())
    + list(spatial_projector.parameters())
)
if hr_ident_head is not None:
    params += list(hr_ident_head.parameters())
optimizer = torch.optim.Adam(params, lr=CONFIG.training.lr)

print(f"✅ Optimizer créé (lr={CONFIG.training.lr})")

# Compter les paramètres (en gérant les paramètres non initialisés/lazy)
def count_parameters(model):
    """Compte les paramètres d'un modèle, en gérant les paramètres non initialisés."""
    total = 0
    trainable = 0
    for p in model.parameters():
        try:
            num = p.numel()
            total += num
            if p.requires_grad:
                trainable += num
        except (ValueError, RuntimeError):
            # Paramètre non initialisé (lazy module), on l'ignore pour le moment
            pass
    return total, trainable

total_params = 0
trainable_params = 0
_models_to_count = [encoder, rcn_cell, diffusion, spatial_projector]
if hr_ident_head is not None:
    _models_to_count.append(hr_ident_head)
for model in _models_to_count:
    total, trainable = count_parameters(model)
    total_params += total
    trainable_params += trainable

print(f"\\n📊 Nombre de paramètres:")
print(f"   Total: {total_params:,}")
print(f"   Trainable: {trainable_params:,}")
'''


# Old/new substring inside the training cell. We surgically replace just
# the train_epoch(...) call so that the rest of the cell (split, val loop,
# early stopping, etc.) stays intact.
OLD_TRAIN_CALL = """    train_metrics = train_epoch(
        encoder=encoder,
        rcn_runner=rcn_runner,
        diffusion_decoder=diffusion,
        optimizer=optimizer,
        data_loader=iterate_batches(train_dataloader, builder, DEVICE),
        lambda_gen=CONFIG.loss.lambda_gen,
        beta_rec=CONFIG.loss.beta_rec,
        gamma_dag=CONFIG.loss.gamma_dag,
        conditioning_fn=None,
        device=DEVICE,
        use_amp=CONFIG.training.get("use_amp", True),
        gradient_clipping=CONFIG.training.gradient_clipping,
        log_interval=CONFIG.training.log_every,
        dag_method=CONFIG.loss.get("dag_method", "dagma"),
        dagma_s=CONFIG.loss.get("dagma_s", 1.0),
        reconstruction_loss_type=CONFIG.loss.get("reconstruction_loss_type", "mse"),
        spatial_projector=spatial_projector,
        conditioning_dropout_prob=float(CONFIG.training.get("conditioning_dropout_prob", 0.0)),
    )"""

NEW_TRAIN_CALL = """    # Sprint 4: pass *all* config-driven knobs through to train_epoch.
    # Previously this cell silently dropped hr_ident, precip_phy, and the
    # contrastive DAG terms — meaning enabling them in the YAML had no
    # effect from the notebook entry point. The combination of
    # ``conditioning_dropout_prob`` from CONFIG.diffusion (not training,
    # which never had it set) + ``contrastive_dag.*`` is what makes the
    # UNet *use* the DAG tokens at training time.
    train_metrics = train_epoch(
        encoder=encoder,
        rcn_runner=rcn_runner,
        diffusion_decoder=diffusion,
        optimizer=optimizer,
        data_loader=iterate_batches(train_dataloader, builder, DEVICE),
        lambda_gen=CONFIG.loss.lambda_gen,
        beta_rec=CONFIG.loss.beta_rec,
        gamma_dag=CONFIG.loss.gamma_dag,
        conditioning_fn=None,
        device=DEVICE,
        use_amp=CONFIG.training.get("use_amp", True),
        gradient_clipping=CONFIG.training.gradient_clipping,
        log_interval=CONFIG.training.log_every,
        dag_method=CONFIG.loss.get("dag_method", "dagma"),
        dagma_s=CONFIG.loss.get("dagma_s", 1.0),
        reconstruction_loss_type=CONFIG.loss.get("reconstruction_loss_type", "mse"),
        spatial_projector=spatial_projector,
        conditioning_dropout_prob=float(
            CONFIG.diffusion.get(
                "conditioning_dropout_prob",
                CONFIG.training.get("conditioning_dropout_prob", 0.0),
            )
        ),
        hr_ident_head=hr_ident_head,
        beta_hr_ident=_beta_hr_ident,
        lambda_precip_phy=float(CONFIG.loss.get("lambda_precip_phy", 0.0)),
        precip_phy_weights=tuple(CONFIG.loss.get("precip_phy_weights", [1.0, 0.1, 0.2])),
        physical_sample_interval=int(
            CONFIG.training.get("physical_loss", {}).get("physical_sample_interval", 10)
        ),
        # Sprint 4: contrastive DAG-conditioning loss.
        lambda_contrastive_dag=(
            float(CONFIG.loss.get("contrastive_dag", {}).get("weight", 0.0))
            if bool(CONFIG.loss.get("contrastive_dag", {}).get("enabled", False))
            else 0.0
        ),
        contrastive_dag_margin=float(
            CONFIG.loss.get("contrastive_dag", {}).get("margin", 0.02)
        ),
        contrastive_dag_interval=int(
            CONFIG.loss.get("contrastive_dag", {}).get("interval", 4)
        ),
    )"""


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    n_models_patched = 0
    n_train_patched = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        cid = cell.get("id")
        if cid == CELL_MODELS:
            cell["source"] = NEW_MODELS_SOURCE.splitlines(keepends=True)
            n_models_patched += 1
        elif cid == CELL_TRAIN:
            src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            if OLD_TRAIN_CALL not in src:
                # Try a tolerant lookup: match on the leading 3 lines to
                # confirm we're at the right call site.
                anchor = "    train_metrics = train_epoch(\n        encoder=encoder,\n"
                if anchor not in src:
                    raise SystemExit(
                        f"[ERROR] Could not locate train_epoch(...) call inside cell {CELL_TRAIN}. "
                        f"Manual review required."
                    )
                # If the anchor is present but the closing differs, re-derive
                # the OLD substring by finding the matching `    )` after it.
                start = src.index(anchor)
                # find the closing line; iterate forward
                depth = 0
                end = None
                for i, ch in enumerate(src[start:], start=start):
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                        if depth == 0:
                            # skip to end of line
                            end = src.index("\n", i) + 1
                            break
                if end is None:
                    raise SystemExit(
                        f"[ERROR] Failed to find end of train_epoch(...) in cell {CELL_TRAIN}."
                    )
                old_call = src[start:end].rstrip("\n")
                src = src.replace(old_call, NEW_TRAIN_CALL)
            else:
                src = src.replace(OLD_TRAIN_CALL, NEW_TRAIN_CALL)
            cell["source"] = src.splitlines(keepends=True)
            n_train_patched += 1

    if n_models_patched != 1:
        raise SystemExit(f"[ERROR] models cell patched {n_models_patched} times (expected 1)")
    if n_train_patched != 1:
        raise SystemExit(f"[ERROR] training cell patched {n_train_patched} times (expected 1)")

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Patched {NB}: models cell + training cell.")


if __name__ == "__main__":
    main()
