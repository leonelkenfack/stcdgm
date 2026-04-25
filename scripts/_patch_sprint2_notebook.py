"""
Patch ``st_cdgm_training_evaluation.ipynb`` to wire Sprint 2 components:
  1. Cell 34 (model init): add CausalConditioningProjector + include in optimizer
  2. Cell 43 (training loop): add dag_grad_gate ramp + pass spatial_projector to train_epoch
  3. Cell 47 (checkpoint save): persist spatial_projector_state_dict

Idempotent: skips if markers already present.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NB_PATH = Path(r"c:/Users/reall/Desktop/climate_data/st_cdgm_training_evaluation.ipynb")


def _to_nb_source(text: str) -> list[str]:
    lines = [line + "\n" for line in text.splitlines()]
    if lines and text.endswith("\n"):
        pass
    elif lines:
        lines[-1] = lines[-1].rstrip("\n")
    return lines


def patch_cell34(nb: dict) -> bool:
    """Add CausalConditioningProjector after diffusion decoder, include in optimizer."""
    cell = nb["cells"][34]
    src = "".join(cell.get("source", []))

    if "CausalConditioningProjector" in src:
        print("Cell 34: already patched (CausalConditioningProjector found)")
        return False

    if "optimizer = torch.optim.Adam(params" not in src:
        print("ERROR: Cell 34 doesn't have expected optimizer line", file=sys.stderr)
        return False

    # 1. Add import + projector creation after diffusion block
    old_optimizer = """# 4. Optimizer (CONFIG.training.lr)
params = list(encoder.parameters()) + list(rcn_cell.parameters()) + list(diffusion.parameters())
optimizer = torch.optim.Adam(params, lr=CONFIG.training.lr)

print(f"✅ Optimizer créé (lr={CONFIG.training.lr})")"""

    new_optimizer = """# 4. Sprint 2: CausalConditioningProjector (DAG tokens → UNet cross-attention)
from st_cdgm.models.intelligible_encoder import (
    SpatialConditioningProjector,
    CausalConditioningProjector,
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

# 5. Optimizer — includes spatial_projector parameters
params = (
    list(encoder.parameters())
    + list(rcn_cell.parameters())
    + list(diffusion.parameters())
    + list(spatial_projector.parameters())
)
optimizer = torch.optim.Adam(params, lr=CONFIG.training.lr)

print(f"✅ Optimizer créé (lr={CONFIG.training.lr})")"""

    if old_optimizer not in src:
        print("ERROR: Cell 34 optimizer block not found verbatim", file=sys.stderr)
        return False

    new_src = src.replace(old_optimizer, new_optimizer)
    cell["source"] = _to_nb_source(new_src)
    cell["outputs"] = []
    cell["execution_count"] = None
    print("Cell 34: patched (CausalConditioningProjector + optimizer)")
    return True


def patch_cell43(nb: dict) -> bool:
    """Add dag_grad_gate ramp and pass spatial_projector to train_epoch."""
    cell = nb["cells"][43]
    src = "".join(cell.get("source", []))

    if "set_dag_grad_gate" in src:
        print("Cell 43: already patched (set_dag_grad_gate found)")
        return False

    # 1. Add gate ramp schedule before the epoch loop
    old_epoch_loop = "for epoch in range(int(CONFIG.training.epochs)):"

    gate_ramp_block = """# Sprint 2: dag_grad_gate warm-up schedule
dag_gate_cfg = CONFIG.rcn.get("dag_grad_gate", {}) or {}
dag_gate_enabled = bool(dag_gate_cfg.get("enabled", False))
dag_gate_cold_epochs = int(dag_gate_cfg.get("cold_epochs", 5))
dag_gate_ramp_epochs = max(1, int(dag_gate_cfg.get("ramp_epochs", 20)))
dag_gate_max = float(dag_gate_cfg.get("max", 1.0))

def _dag_gate_value(epoch_idx):
    if not dag_gate_enabled:
        return 0.0
    if epoch_idx < dag_gate_cold_epochs:
        return 0.0
    progress = (epoch_idx - dag_gate_cold_epochs) / dag_gate_ramp_epochs
    return min(dag_gate_max, progress * dag_gate_max)

if dag_gate_enabled:
    print(f"🧠 Sprint 2: dag_grad_gate enabled (cold={dag_gate_cold_epochs}, ramp={dag_gate_ramp_epochs}, max={dag_gate_max})")

for epoch in range(int(CONFIG.training.epochs)):"""

    if old_epoch_loop not in src:
        print("ERROR: Cell 43 epoch loop start not found", file=sys.stderr)
        return False

    src = src.replace(old_epoch_loop, gate_ramp_block, 1)

    # 2. Add gate update + spatial_projector train mode after the train/eval toggles
    old_train_mode = """    encoder.train()
    rcn_runner.cell.train()
    diffusion.train()

    train_metrics = train_epoch("""

    new_train_mode = """    encoder.train()
    rcn_runner.cell.train()
    diffusion.train()
    if spatial_projector is not None:
        spatial_projector.train()

    # Sprint 2: update dag_grad_gate for this epoch
    _gate = _dag_gate_value(epoch)
    if hasattr(rcn_cell, "set_dag_grad_gate"):
        _rcn_target = rcn_cell.module if hasattr(rcn_cell, "module") else rcn_cell
        _rcn_target = getattr(_rcn_target, "_orig_mod", _rcn_target)
        _rcn_target.set_dag_grad_gate(_gate)
    if dag_gate_enabled:
        print(f"  [Epoch {epoch + 1}] dag_grad_gate = {_gate:.3f}")

    train_metrics = train_epoch("""

    if old_train_mode not in src:
        print("ERROR: Cell 43 train mode block not found", file=sys.stderr)
        return False

    src = src.replace(old_train_mode, new_train_mode, 1)

    # 3. Add spatial_projector kwarg to train_epoch call
    old_train_call_end = """        reconstruction_loss_type=CONFIG.loss.get("reconstruction_loss_type", "mse"),
    )"""

    new_train_call_end = """        reconstruction_loss_type=CONFIG.loss.get("reconstruction_loss_type", "mse"),
        spatial_projector=spatial_projector,
        conditioning_dropout_prob=float(CONFIG.training.get("conditioning_dropout_prob", 0.0)),
    )"""

    if old_train_call_end not in src:
        print("ERROR: Cell 43 train_epoch call end not found", file=sys.stderr)
        return False

    src = src.replace(old_train_call_end, new_train_call_end, 1)

    # 4. Add conditioning_spatial to compute_validation_loss
    old_val_conditioning = """            conditioning = encoder.project_state_tensor(seq_output.states[-1]).to(DEVICE)
            generated = diffusion.sample(
                conditioning,
                num_steps=int(CONFIG.diffusion.get("val_num_steps", 15)),
                scheduler_type=CONFIG.diffusion.get("scheduler_type", "ddpm"),
                apply_constraints=False,
            )"""

    new_val_conditioning = """            H_last = seq_output.states[-1]
            conditioning = encoder.project_state_tensor(H_last).to(DEVICE)
            conditioning_spatial = None
            if spatial_projector is not None:
                spatial_projector.eval()
                _sp = spatial_projector.module if hasattr(spatial_projector, "module") else spatial_projector
                if hasattr(_sp, "dag_mlp"):
                    A_dag = rcn_runner.cell.A_dag if hasattr(rcn_runner.cell, "A_dag") else rcn_runner.cell.module.A_dag
                    A_masked = A_dag - torch.diag(torch.diagonal(A_dag))
                    conditioning_spatial = _sp(H_last, A_masked)
                else:
                    conditioning_spatial = _sp(H_last)
                conditioning_spatial = conditioning_spatial.to(DEVICE)
            generated = diffusion.sample(
                conditioning,
                num_steps=int(CONFIG.diffusion.get("val_num_steps", 15)),
                scheduler_type=CONFIG.diffusion.get("scheduler_type", "ddpm"),
                apply_constraints=False,
                conditioning_spatial=conditioning_spatial,
            )"""

    if old_val_conditioning not in src:
        print("ERROR: Cell 43 val conditioning block not found", file=sys.stderr)
        return False

    src = src.replace(old_val_conditioning, new_val_conditioning, 1)

    # 5. Add spatial_projector to BEST_MODEL_STATES
    old_best_states = """        BEST_MODEL_STATES = {
            "encoder_state_dict": copy.deepcopy(_extract_state_dict(encoder)),
            "rcn_cell_state_dict": copy.deepcopy(_extract_state_dict(rcn_cell)),
            "diffusion_state_dict": copy.deepcopy(_extract_state_dict(diffusion)),
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        }"""

    new_best_states = """        BEST_MODEL_STATES = {
            "encoder_state_dict": copy.deepcopy(_extract_state_dict(encoder)),
            "rcn_cell_state_dict": copy.deepcopy(_extract_state_dict(rcn_cell)),
            "diffusion_state_dict": copy.deepcopy(_extract_state_dict(diffusion)),
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        }
        if spatial_projector is not None:
            BEST_MODEL_STATES["spatial_projector_state_dict"] = copy.deepcopy(
                _extract_state_dict(spatial_projector)
            )"""

    if old_best_states not in src:
        print("ERROR: Cell 43 BEST_MODEL_STATES block not found", file=sys.stderr)
        return False

    src = src.replace(old_best_states, new_best_states, 1)

    cell["source"] = _to_nb_source(src)
    cell["outputs"] = []
    cell["execution_count"] = None
    print("Cell 43: patched (dag_grad_gate ramp + spatial_projector + conditioning_spatial)")
    return True


def patch_cell47(nb: dict) -> bool:
    """Add spatial_projector_state_dict to checkpoint save."""
    cell = nb["cells"][47]
    src = "".join(cell.get("source", []))

    if "spatial_projector_state_dict" in src:
        print("Cell 47: already patched (spatial_projector_state_dict found)")
        return False

    # Add spatial_projector to last_payload
    old_last_payload = """    last_payload = {
        "epoch": len(history.get("loss", [])),
        "encoder_state_dict": _extract_state_dict(encoder),
        "rcn_cell_state_dict": _extract_state_dict(rcn_cell),
        "diffusion_state_dict": _extract_state_dict(diffusion),
        "optimizer_state_dict": optimizer.state_dict(),
        **checkpoint_common,
    }
    torch.save(last_payload, last_checkpoint_path)"""

    new_last_payload = """    last_payload = {
        "epoch": len(history.get("loss", [])),
        "encoder_state_dict": _extract_state_dict(encoder),
        "rcn_cell_state_dict": _extract_state_dict(rcn_cell),
        "diffusion_state_dict": _extract_state_dict(diffusion),
        "optimizer_state_dict": optimizer.state_dict(),
        **checkpoint_common,
    }
    if "spatial_projector" in dir() and spatial_projector is not None:
        last_payload["spatial_projector_state_dict"] = _extract_state_dict(spatial_projector)
    torch.save(last_payload, last_checkpoint_path)"""

    if old_last_payload not in src:
        print("ERROR: Cell 47 last_payload block not found", file=sys.stderr)
        return False

    src = src.replace(old_last_payload, new_last_payload, 1)

    # Add spatial_projector to best_payload
    old_best_save = """        torch.save(best_payload, best_checkpoint_path)
        torch.save(best_payload, standard_checkpoint_path)"""

    new_best_save = """        if "BEST_MODEL_STATES" in globals() and BEST_MODEL_STATES is not None and "spatial_projector_state_dict" in BEST_MODEL_STATES:
            best_payload["spatial_projector_state_dict"] = BEST_MODEL_STATES["spatial_projector_state_dict"]
        torch.save(best_payload, best_checkpoint_path)
        torch.save(best_payload, standard_checkpoint_path)"""

    if old_best_save not in src:
        print("ERROR: Cell 47 best_save block not found", file=sys.stderr)
        return False

    src = src.replace(old_best_save, new_best_save, 1)

    cell["source"] = _to_nb_source(src)
    cell["outputs"] = []
    cell["execution_count"] = None
    print("Cell 47: patched (spatial_projector_state_dict in checkpoint)")
    return True


def main() -> int:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    patched_any = False
    patched_any |= patch_cell34(nb)
    patched_any |= patch_cell43(nb)
    patched_any |= patch_cell47(nb)

    if not patched_any:
        print("\nNotebook already fully patched; nothing to do.")
        return 0

    NB_PATH.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nOK: patched {NB_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
