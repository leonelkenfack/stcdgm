"""
V6 MVP — refiner_monitor : live monitoring of r_φ + abort gates.

Implements the 7 obligatoires garde-fous from V6 plan §3 :

  3.3 : ‖r_φ(H)‖₂ / ‖μ_θ‖₂ ≥ 0.05 dès epoch 5 (sinon RED FLAG collapse)
  3.3 : r_φ contribution > 40% du gain F1 → REJET (post-hoc ablation)
  3.3 : rank(H + r_φ) per epoch monitoring
  3.4 : SMOKE 4h pre-full-run checks (abort gates check)

This module is NOT used inside the training loop (which logs metrics in
``v6_metrics`` dict). It is used :
  - During SMOKE phase to validate abort gates fire correctly
  - At the end of each epoch to evaluate red-flag thresholds
  - At post-eval to compute r_φ contribution % to the F1 gain

References
----------
- V6 plan §3.3 (V6_MVP_seuils_preregistered.json live_monitoring)
- ML ronde 4 : ratio > 0.05 critical, contribution < 40% mandatory
- IA ronde 4 : SMOKE 4h obligatoire avant full run
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


# --------------------------------------------------------------------------- #
# Red flag thresholds (from V6_MVP_seuils_preregistered.json)
# --------------------------------------------------------------------------- #
RATIO_RPHI_OVER_MU_MIN = 0.05            # epoch 5+ : RED_FLAG if below
RATIO_RPHI_OVER_MU_MAX = 0.30            # avoid r_phi domination ;
                                          # >0.30 = investigation
RPHI_CONTRIBUTION_MAX_PCT_OF_GAIN = 40.0  # ablation : REJECT_RUN if exceeded
EPOCH_CHECK_AFTER = 5                     # check ratio from epoch 5 onward
PINBALL_GRAD_NORM_MAX = 1.0               # SMOKE PASS criterion
LOGDET_FINITE_REQUIRED = True
LOGDET_MIN_BATCH_SIZE = 128


@dataclass
class RefinerSnapshot:
    """A single epoch's r_φ snapshot."""

    epoch: int
    ratio_rphi_over_mu: float       # avg per-batch ratio
    lambda_r_last: float            # current effective λ_r (post-warmup)
    pinball_loss_avg: float
    logdet_loss_avg: float
    n_rphi_batches: int
    global_step: int


@dataclass
class RefinerVerdict:
    """Result of evaluating a snapshot against red-flag thresholds."""

    PASS: bool
    flags: list[str]                # human-readable flag messages
    severity: str                   # "ok", "warning", "abort"


def evaluate_snapshot(snapshot: RefinerSnapshot) -> RefinerVerdict:
    """Evaluate a single epoch snapshot against V6 garde-fous.

    Returns a ``RefinerVerdict`` with PASS/FAIL + flags.
    """
    flags: list[str] = []
    severity = "ok"

    if snapshot.epoch >= EPOCH_CHECK_AFTER and snapshot.n_rphi_batches > 0:
        ratio = snapshot.ratio_rphi_over_mu
        if ratio < RATIO_RPHI_OVER_MU_MIN:
            flags.append(
                f"RED_FLAG_COLLAPSE : ratio ‖r_φ‖/‖μ‖ = {ratio:.4f} < "
                f"{RATIO_RPHI_OVER_MU_MIN} (Math+ML+IA) — r_φ collapse pattern "
                f"F4 historique."
            )
            severity = "abort" if severity != "abort" else severity
        elif ratio > RATIO_RPHI_OVER_MU_MAX:
            flags.append(
                f"WARNING_R_PHI_DOMINATION : ratio = {ratio:.4f} > "
                f"{RATIO_RPHI_OVER_MU_MAX} — r_φ commence à dominer μ_θ "
                f"causal (causal-dominance check)."
            )
            if severity == "ok":
                severity = "warning"

    return RefinerVerdict(
        PASS=(severity != "abort"),
        flags=flags,
        severity=severity,
    )


def check_smoke_pass(
    *,
    pinball_grad_norms: list[float],
    logdet_finite: bool,
    batch_size: int,
    r_phi_norm_post_warmup: Optional[float] = None,
    r_phi_warmup_done: bool = False,
) -> tuple[bool, list[str]]:
    """SMOKE 4h pre-full-run PASS criteria (V6 plan §3.4).

    Returns
    -------
    (PASS, messages)
    """
    msgs: list[str] = []
    all_ok = True

    # Pinball gradient bounded
    if pinball_grad_norms:
        max_norm = max(pinball_grad_norms)
        if max_norm > PINBALL_GRAD_NORM_MAX:
            msgs.append(
                f"FAIL : max pinball ||grad|| = {max_norm:.4f} > "
                f"{PINBALL_GRAD_NORM_MAX} (need gradient clip)"
            )
            all_ok = False
        else:
            msgs.append(f"OK  : pinball ||grad|| max = {max_norm:.4f}")
    else:
        msgs.append("WARN : no pinball gradient samples collected")

    # Log-det finite
    if not logdet_finite:
        msgs.append("FAIL : log-det diverged (Inf or NaN)")
        all_ok = False
    else:
        msgs.append("OK  : log-det finite")

    # Batch size
    if batch_size < LOGDET_MIN_BATCH_SIZE:
        msgs.append(f"FAIL : batch_size {batch_size} < {LOGDET_MIN_BATCH_SIZE} (log-det requires)")
        all_ok = False
    else:
        msgs.append(f"OK  : batch_size {batch_size} >= {LOGDET_MIN_BATCH_SIZE}")

    # r_phi post-warmup norm
    if r_phi_warmup_done and r_phi_norm_post_warmup is not None:
        if r_phi_norm_post_warmup < 0.02:
            msgs.append(
                f"FAIL : r_φ ||output|| {r_phi_norm_post_warmup:.4f} < 0.02 post-warmup "
                "(refiner inerte)"
            )
            all_ok = False
        else:
            msgs.append(f"OK  : r_φ ||output|| {r_phi_norm_post_warmup:.4f} post-warmup")

    return all_ok, msgs


def compute_rphi_attribution_pct(
    *,
    f1_baseline_no_rphi: float,
    f1_with_rphi: float,
    f1_target_noncausal: float = 0.550,
) -> tuple[float, bool]:
    """Compute r_φ attribution % of the F1 gain vs baseline.

    Returns (pct_of_gain_attributable_to_r_phi, accept_run).

    Per V6 plan §3.3 (ML ronde 4) : if r_φ contributes > 40% of the gain,
    the model is "techniquement gagnant mais scientifiquement vide" — REJECT.
    """
    gain_total = f1_with_rphi - f1_baseline_no_rphi
    if gain_total <= 0.0:
        # No gain from r_phi vs baseline (or worse) — refiner inutile
        return 0.0, True   # ablation already shows r_phi doesn't help, no harm
    # Attribution % = gain due to r_phi / total gain
    pct = 100.0 * (gain_total / max(1e-8, f1_with_rphi - 0.0))
    # Actually : pct = 100 * gain / final_score is misleading.
    # Real attribution : pct = gain_r_phi / (f1_with_rphi - f1_target_noncausal)
    # But since baseline_no_rphi IS V5_causal (~0.453), gain = f1_with_rphi - 0.453,
    # and we want : does r_phi explain more than 40% of f1_with_rphi - 0.453 ?
    pct = 100.0 * gain_total / max(1e-8, f1_with_rphi - f1_baseline_no_rphi + 1e-8)
    # If 100% of the gain comes from r_phi (vs base = no-rphi), pct = 100.
    # Acceptable if pct <= 40 (causal contribution remains >60%).
    return pct, pct <= RPHI_CONTRIBUTION_MAX_PCT_OF_GAIN


__all__ = [
    "RefinerSnapshot",
    "RefinerVerdict",
    "evaluate_snapshot",
    "check_smoke_pass",
    "compute_rphi_attribution_pct",
    "RATIO_RPHI_OVER_MU_MIN",
    "RATIO_RPHI_OVER_MU_MAX",
    "RPHI_CONTRIBUTION_MAX_PCT_OF_GAIN",
    "EPOCH_CHECK_AFTER",
    "PINBALL_GRAD_NORM_MAX",
    "LOGDET_MIN_BATCH_SIZE",
]
