"""
BS34 — tail-aware EDM loss (Phase 1 of paper-gap closure plan).

Motivation
----------
Standard score-matching MSE weights every pixel equally, so a 200 mm/day
event contributes the same as a 2 mm/day event. This is the direct
mechanism behind F1@p99 ≈ 0.004 in BS33b's evaluation. The literature
that addresses this gap:

* Ravuri et al. 2021 (DGMR, *Nature*) — intensity-weighted loss in
  generative precipitation nowcasting.
* Liu et al. 2024 (WassDiff, arXiv:2410.00381) — Wasserstein-regularised
  diffusion for extreme precipitation. Uses a weight = f(intensity)
  before the score-matching MSE.
* Harris et al. 2022 (arXiv:2204.02028, JAMES) — Bernoulli-Gamma loss
  for precipitation.

This patch installs a much simpler intervention than BG-NLL: a
multiplicative threshold-weighted denoising loss. Pixels whose full
HR_log reconstruction (= ``baseline_log + μ_HR + δ_target``) exceeds
the p95/p99 quantile of the training CHIRPS distribution receive a
larger weight. Cost: ≈30 lines of code; expected impact: F1@p99 ×10–30
on the same data, larger Pearson improvement when combined with t-EDM
(BS35) and ERA5 conditioning (BS36).

What this patch changes
-----------------------
1. ``src/st_cdgm/models/edm_preconditioner.py``
   - Adds ``TailWeightConfig`` dataclass.
   - Adds optional ``tail_weight`` field on ``EDMConfig``.

2. ``src/st_cdgm/models/diffusion_decoder.py`` (``compute_loss_edm``)
   - Computes the full HR reconstruction in log space, then a per-pixel
     threshold weight, then multiplies the existing ``λ(σ) · sq_err``.

3. ``config/training_config_corrdiff_mini.yaml``
   - Adds a ``diffusion.edm.tail_weight`` block (defaults: 1.0/5.0/10.0).

4. ``st_cdgm_training_evaluation.ipynb`` cell 35 (EDM_BUILD_CONFIG)
   - Reads ``tail_weight`` from YAML and passes it to ``EDMConfig``.

Default thresholds (mm/day, BEFORE log1p transform):
   p95 = 15.0   (multiplier 5.0)
   p99 = 35.0   (multiplier 10.0)

These defaults are calibrated for Bénin / West African daily CHIRPS
precipitation. If you change region or transform, recompute via
``scripts/measure_chirps_quantiles.py`` (next patch).

Idempotent — sentinel ``BS34_TAIL_LOSS``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EDM_PRECOND = ROOT / "src" / "st_cdgm" / "models" / "edm_preconditioner.py"
DIFFUSION_DEC = ROOT / "src" / "st_cdgm" / "models" / "diffusion_decoder.py"
OVERRIDE_YAML = ROOT / "config" / "training_config_corrdiff_mini.yaml"
TRAIN_NB = ROOT / "st_cdgm_training_evaluation.ipynb"


# ----------------------------------------------------------------------
# 1) edm_preconditioner.py — add TailWeightConfig + EDMConfig field
# ----------------------------------------------------------------------

OLD_EDM_DATACLASS = '''@dataclass
class EDMConfig:
    """EDM hyper-parameters. Defaults are Karras 2022 Table 5 (CIFAR-10)
    except sigma_data, which MUST be calibrated to the target dataset.

    For the precipitation residual setting we expect sigma_data ~= 0.1
    after log1p; the user calibrates this with
    ``scripts/measure_residual_std.py`` before training and overrides
    via the YAML config.
    """

    sigma_data: float = 0.1
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    P_mean: float = -1.2
    P_std: float = 1.2
    # Stochastic sampling (set S_churn>0 for stochastic Heun)
    S_churn: float = 0.0
    S_tmin: float = 0.0
    S_tmax: float = float("inf")
    S_noise: float = 1.0'''


NEW_EDM_DATACLASS = '''# >>> BS34_TAIL_LOSS — tail-aware MSE config (Ravuri 2021, WassDiff 2024).
@dataclass
class TailWeightConfig:
    """Threshold-weighted denoising loss for heavy-tailed targets.

    The weight applied to ``(D - x0)²`` at each pixel is::

        w(x) = 1 + (w95 - 1) · 1[x > τ95] + (w99 - w95) · 1[x > τ99]

    where ``x`` is the FULL HR field reconstructed in log1p(mm/day) space
    (``baseline_log + μ_HR + target``). Defaults match Bénin/West African
    daily CHIRPS climatology; recompute via
    ``scripts/measure_chirps_quantiles.py`` after data swap.

    References
    ----------
    Ravuri et al. 2021, *Nature* (DGMR) — intensity-weighted nowcasting loss.
    Liu et al. 2024, arXiv:2410.00381 (WassDiff) — extreme-precipitation
    diffusion regularisation; this is the simplified weight-only ablation.
    """

    enabled: bool = False                 # OFF by default — opt-in via YAML
    tau95_mmday: float = 15.0             # log1p applied internally
    tau99_mmday: float = 35.0
    weight_p95: float = 5.0
    weight_p99: float = 10.0


@dataclass
class EDMConfig:
    """EDM hyper-parameters. Defaults are Karras 2022 Table 5 (CIFAR-10)
    except sigma_data, which MUST be calibrated to the target dataset.

    For the precipitation residual setting we expect sigma_data ~= 0.1
    after log1p; the user calibrates this with
    ``scripts/measure_residual_std.py`` before training and overrides
    via the YAML config.
    """

    sigma_data: float = 0.1
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    P_mean: float = -1.2
    P_std: float = 1.2
    # Stochastic sampling (set S_churn>0 for stochastic Heun)
    S_churn: float = 0.0
    S_tmin: float = 0.0
    S_tmax: float = float("inf")
    S_noise: float = 1.0
    # >>> BS34_TAIL_LOSS — optional tail-weighting (None = legacy MSE).
    tail_weight: TailWeightConfig | None = None'''


def patch_edm_preconditioner() -> int:
    src = EDM_PRECOND.read_text(encoding="utf-8")
    if "BS34_TAIL_LOSS" in src:
        print(f"  = {EDM_PRECOND.name} already patched (BS34)")
        return 0
    if OLD_EDM_DATACLASS not in src:
        print(f"  ! {EDM_PRECOND.name}: EDMConfig anchor not found")
        return 0
    src = src.replace(OLD_EDM_DATACLASS, NEW_EDM_DATACLASS, 1)
    EDM_PRECOND.write_text(src, encoding="utf-8")
    print(f"  ~ {EDM_PRECOND.name}: TailWeightConfig + EDMConfig.tail_weight added")
    return 1


# ----------------------------------------------------------------------
# 2) diffusion_decoder.py — patch compute_loss_edm
# ----------------------------------------------------------------------

OLD_LOSS_TAIL = '''        weights = lambda_weight(sigma, cfg.sigma_data)  # [B, 1, 1, 1]
        sq_err = (D_y - target_clean) ** 2  # [B, C, H, W]

        # Apply mask + weight, then mean over valid elements only.
        # Broadcast weights against [B, C, H, W].
        weighted = weights * sq_err
        masked = weighted * valid_mask.float()'''


NEW_LOSS_TAIL = '''        weights = lambda_weight(sigma, cfg.sigma_data)  # [B, 1, 1, 1]
        sq_err = (D_y - target_clean) ** 2  # [B, C, H, W]

        # >>> BS34_TAIL_LOSS — multiplicative threshold weight on extreme
        # pixels of the FULL HR reconstruction (Ravuri 2021, WassDiff 2024).
        # Reconstructed field = target + μ_HR + baseline_log (log1p(mm/d)).
        # When config disabled → identity, no behaviour change.
        tw_cfg = getattr(cfg, "tail_weight", None)
        if tw_cfg is not None and getattr(tw_cfg, "enabled", False):
            with torch.no_grad():
                hr_log_recon = target_clean
                if mu_HR is not None:
                    hr_log_recon = hr_log_recon + mu_HR
                if baseline_log is not None:
                    hr_log_recon = hr_log_recon + baseline_log
                tau95 = math.log1p(float(tw_cfg.tau95_mmday))
                tau99 = math.log1p(float(tw_cfg.tau99_mmday))
                w95 = float(tw_cfg.weight_p95)
                w99 = float(tw_cfg.weight_p99)
                tail_w = (
                    1.0
                    + (w95 - 1.0) * (hr_log_recon > tau95).float()
                    + (w99 - w95) * (hr_log_recon > tau99).float()
                )
            weighted = weights * tail_w * sq_err
        else:
            weighted = weights * sq_err
        masked = weighted * valid_mask.float()'''


def patch_diffusion_decoder() -> int:
    src = DIFFUSION_DEC.read_text(encoding="utf-8")
    if "BS34_TAIL_LOSS" in src:
        print(f"  = {DIFFUSION_DEC.name} already patched (BS34)")
        return 0
    if OLD_LOSS_TAIL not in src:
        print(f"  ! {DIFFUSION_DEC.name}: compute_loss_edm anchor not found")
        return 0
    # Ensure ``import math`` is present (it is, line 41 of edm_preconditioner;
    # diffusion_decoder.py imports math too — verify):
    if "import math" not in src:
        # Inject after the first ``import torch``.
        src = src.replace("import torch\n", "import math\nimport torch\n", 1)
        print(f"  ~ {DIFFUSION_DEC.name}: added 'import math'")
    src = src.replace(OLD_LOSS_TAIL, NEW_LOSS_TAIL, 1)
    DIFFUSION_DEC.write_text(src, encoding="utf-8")
    print(f"  ~ {DIFFUSION_DEC.name}: tail-weighted MSE wired in compute_loss_edm")
    return 1


# ----------------------------------------------------------------------
# 3) training_config_corrdiff_mini.yaml — add tail_weight override block
# ----------------------------------------------------------------------

YAML_APPEND = '''
# =============================================================================
# BS34 — tail-aware EDM loss (Phase 1 of the paper-gap closure plan).
#
# Multiplies the per-pixel score-matching MSE by a step function of
# the FULL HR field reconstructed in log1p(mm/day) space:
#
#     w(x) = 1 + (w95-1) · 1[x > τ95] + (w99-w95) · 1[x > τ99]
#
# Counters the standard EDM loss's flat per-pixel weighting which
# treats a 200 mm/day event identically to a 2 mm/day event — the
# direct mechanism behind F1@p99 = 0.0037 in BS33b/v3.
#
# Refs : Ravuri et al. 2021 (DGMR Nature), Liu et al. 2024 (WassDiff
# arXiv:2410.00381), Harris et al. 2022 (arXiv:2204.02028).
#
# Default thresholds (CHIRPS daily climatology, Bénin / West Africa):
#   p95 ≈ 15 mm/day, p99 ≈ 35 mm/day. Recompute with
#   ``scripts/measure_chirps_quantiles.py`` if domain or data swap.
# =============================================================================
diffusion:
  edm:
    tail_weight:
      enabled: true
      tau95_mmday: 15.0
      tau99_mmday: 35.0
      weight_p95: 5.0
      weight_p99: 10.0
'''


def patch_override_yaml() -> int:
    src = OVERRIDE_YAML.read_text(encoding="utf-8")
    if "BS34_TAIL_LOSS" in src or "tail_weight:" in src:
        print(f"  = {OVERRIDE_YAML.name} already has tail_weight (BS34)")
        return 0
    src = src.rstrip() + "\n" + YAML_APPEND
    OVERRIDE_YAML.write_text(src, encoding="utf-8")
    print(f"  ~ {OVERRIDE_YAML.name}: appended tail_weight override")
    return 1


# ----------------------------------------------------------------------
# 4) Notebook cell 35 (EDM_BUILD_CONFIG) — read tail_weight from YAML
# ----------------------------------------------------------------------

OLD_NB_EDM = '''_edm_cfg_raw = CONFIG.diffusion.get("edm", {})
    _edm_config = EDMConfig(
        sigma_data=float(_edm_cfg_raw.get("sigma_data", 0.1)),
        sigma_min=float(_edm_cfg_raw.get("sigma_min", 0.002)),
        sigma_max=float(_edm_cfg_raw.get("sigma_max", 80.0)),
        rho=float(_edm_cfg_raw.get("rho", 7.0)),
        P_mean=float(_edm_cfg_raw.get("P_mean", -1.2)),
        P_std=float(_edm_cfg_raw.get("P_std", 1.2)),
        S_churn=float(_edm_cfg_raw.get("S_churn", 0.0)),
        S_tmin=float(_edm_cfg_raw.get("S_tmin", 0.0)),
        S_tmax=float(_edm_cfg_raw.get("S_tmax", float("inf"))),
        S_noise=float(_edm_cfg_raw.get("S_noise", 1.0)),
    )'''


NEW_NB_EDM = '''_edm_cfg_raw = CONFIG.diffusion.get("edm", {})
    # >>> BS34_TAIL_LOSS — read tail_weight block (None if absent).
    from st_cdgm.models.edm_preconditioner import TailWeightConfig as _TWConfig
    _tw_raw = _edm_cfg_raw.get("tail_weight", None)
    if _tw_raw is not None and bool(_tw_raw.get("enabled", False)):
        _tail_w = _TWConfig(
            enabled=True,
            tau95_mmday=float(_tw_raw.get("tau95_mmday", 15.0)),
            tau99_mmday=float(_tw_raw.get("tau99_mmday", 35.0)),
            weight_p95=float(_tw_raw.get("weight_p95", 5.0)),
            weight_p99=float(_tw_raw.get("weight_p99", 10.0)),
        )
        print(f"   📈 BS34 tail-weight ON : τ95={_tail_w.tau95_mmday}mm "
              f"(w={_tail_w.weight_p95}), τ99={_tail_w.tau99_mmday}mm "
              f"(w={_tail_w.weight_p99})")
    else:
        _tail_w = None
    _edm_config = EDMConfig(
        sigma_data=float(_edm_cfg_raw.get("sigma_data", 0.1)),
        sigma_min=float(_edm_cfg_raw.get("sigma_min", 0.002)),
        sigma_max=float(_edm_cfg_raw.get("sigma_max", 80.0)),
        rho=float(_edm_cfg_raw.get("rho", 7.0)),
        P_mean=float(_edm_cfg_raw.get("P_mean", -1.2)),
        P_std=float(_edm_cfg_raw.get("P_std", 1.2)),
        S_churn=float(_edm_cfg_raw.get("S_churn", 0.0)),
        S_tmin=float(_edm_cfg_raw.get("S_tmin", 0.0)),
        S_tmax=float(_edm_cfg_raw.get("S_tmax", float("inf"))),
        S_noise=float(_edm_cfg_raw.get("S_noise", 1.0)),
        tail_weight=_tail_w,
    )'''


def _find_cell(cells, predicate):
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        if predicate("".join(c.get("source", []))):
            return i
    return None


def patch_notebook() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    idx = _find_cell(cells, lambda s: "EDM_BUILD_CONFIG" in s and "EDMConfig(" in s)
    if idx is None:
        print("  ! EDM_BUILD_CONFIG cell not found")
        return 0
    src = "".join(cells[idx]["source"])
    if "BS34_TAIL_LOSS" in src:
        print(f"  = cell {idx} already patched (BS34)")
        return 0
    if OLD_NB_EDM not in src:
        print(f"  ! cell {idx}: EDMConfig anchor not found")
        return 0
    src = src.replace(OLD_NB_EDM, NEW_NB_EDM, 1)
    cells[idx]["source"] = src.splitlines(keepends=True)
    cells[idx]["outputs"] = []
    cells[idx]["execution_count"] = None
    TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  ~ cell {idx}: EDM_BUILD_CONFIG reads tail_weight from YAML")
    return 1


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main() -> int:
    print("=== BS34 : tail-aware EDM loss (Phase 1) ===")
    n = 0
    n += patch_edm_preconditioner()
    n += patch_diffusion_decoder()
    n += patch_override_yaml()
    n += patch_notebook()
    print(f"\n{n} edit(s) applied")
    print(
        "Sanity check : after running this, training will pick up "
        "tail_weight from YAML automatically. Default thresholds are "
        "for Bénin daily CHIRPS — adjust via override YAML if needed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
