"""
BS28 — fix ``generate_prediction`` for the two-stage causal_concat
inference path.

Symptom (cell 57 → cell 58 smoke test):
    ValueError: sample(causal_concat=True) requires mu_HR and baseline_log

Cause: ``generate_prediction`` was authored before Stage 1 / regression
head was introduced. With ``causal_concat=True``, the diffusion UNet
takes ``[δ_noisy, μ_HR, baseline_log]`` as input channels, so
``diffusion.sample(...)`` requires ``mu_HR`` and ``baseline_log``
explicitly.

Patch (cell 57): mirror Stage-2 training construction —
- ``mu_HR = regression_head(H_T)`` (interp to target shape if needed)
- ``baseline_log = batch["baseline"][-1]`` (already in log1p space)
- ``torch.nan_to_num`` both (BS17 ocean-void protection)
- pass to ``diffusion.sample(..., mu_HR=mu_HR, baseline_log=baseline_log)``

Also degrade gracefully when ``regression_head`` doesn't exist (legacy
single-stage checkpoint loaded for a smoke test): fall back to the old
behaviour.

Idempotent — sentinel ``BS28_GENERATE_PREDICTION``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_NB = ROOT / "st_cdgm_training_evaluation.ipynb"


def _find_cell(cells, predicate):
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        if predicate("".join(c.get("source", []))):
            return i
    return None


NEW_FN = '''@torch.no_grad()
def generate_prediction(encoder, rcn_runner, diffusion, sample, builder, device):
    """Génère une prédiction haute résolution à partir d'un échantillon.

    BS28 — supports the two-stage causal_concat path: when the diffusion
    UNet was built with ``causal_concat=True``, ``sample(...)`` requires
    ``mu_HR`` and ``baseline_log`` (the extra input channels). We
    reconstruct them exactly as Stage-2 training does:
    - ``mu_HR = regression_head(H_T)``  (interpolated to target shape)
    - ``baseline_log = batch["baseline"][-1]``  (already in log1p space)
    - both ``nan_to_num``ed to 0 at ocean voids.

    Falls back to the legacy single-channel path when ``regression_head``
    is absent or the diffusion is not in causal_concat mode.
    """
    encoder.eval()
    rcn_runner.cell.eval()
    diffusion.eval()

    # Convertir le sample
    batch = convert_sample_to_batch(sample, builder, device)

    lr_data = batch["lr"].to(device)
    hetero_data = batch["hetero"]

    # Forward pass
    H_init = encoder.init_state(hetero_data).to(device)
    drivers = [lr_data[t] for t in range(lr_data.shape[0])]
    seq_output = rcn_runner.run(H_init, drivers, reconstruction_sources=None)

    # Conditioning pour la diffusion
    H_T = seq_output.states[-1]
    conditioning = encoder.project_state_tensor(H_T).to(device)

    # >>> BS28_GENERATE_PREDICTION
    # Detect causal_concat mode by inspecting the eager (uncompiled) core.
    _diff_core = getattr(diffusion, "_orig_mod", diffusion)
    _diff_core = getattr(_diff_core, "module", _diff_core)
    _causal_concat = bool(getattr(_diff_core, "causal_concat", False))

    # In causal_concat mode only ``edm_karras`` (and ``dpm_solver++``)
    # support the 3-channel input ``[δ_noisy, μ_HR, baseline_log]``;
    # the DDPM fall-through path does not. Pick edm_karras then.
    sample_kwargs = dict(
        num_steps=18 if _causal_concat else 25,
        scheduler_type="edm_karras" if _causal_concat else "ddpm",
        apply_constraints=False,
    )

    if _causal_concat and "regression_head" in dir() and regression_head is not None:
        # Build mu_HR + baseline_log just like Stage-2 training.
        target_shape = batch["residual"][-1].to(device).shape  # (C, H, W) or (B, C, H, W)
        mu_HR = regression_head(H_T)
        if mu_HR.shape[-2:] != target_shape[-2:]:
            mu_HR = torch.nn.functional.interpolate(
                mu_HR, size=target_shape[-2:],
                mode="bilinear", align_corners=False,
            )
        baseline_t = batch["baseline"][-1].to(device)
        if baseline_t.dim() == mu_HR.dim() - 1:
            baseline_t = baseline_t.unsqueeze(0)
        baseline_log = baseline_t  # already log1p-encoded by the data pipeline
        mu_HR = torch.nan_to_num(mu_HR, nan=0.0, posinf=0.0, neginf=0.0)
        baseline_log = torch.nan_to_num(baseline_log, nan=0.0, posinf=0.0, neginf=0.0)
        sample_kwargs["mu_HR"] = mu_HR
        sample_kwargs["baseline_log"] = baseline_log

    generated = diffusion.sample(conditioning, **sample_kwargs)

    return generated.residual, batch["residual"][-1].to(device), batch["baseline"][-1].to(device)

print("✅ Fonction d'inférence définie")'''


def patch_cell_57() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    idx = _find_cell(cells, lambda s: "def generate_prediction" in s
                     and "diffusion.sample" in s)
    if idx is None:
        print("  ! generate_prediction cell not found")
        return 0
    # Always overwrite — the NEW_FN body is the canonical implementation.
    cells[idx]["source"] = NEW_FN.splitlines(keepends=True)
    cells[idx]["outputs"] = []
    cells[idx]["execution_count"] = None
    TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  ~ cell {idx}: generate_prediction rewritten for causal_concat")
    return 1


def main() -> int:
    print("=== BS28 : generate_prediction supports causal_concat ===")
    n = patch_cell_57()
    print(f"\n{n} edit(s) applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
