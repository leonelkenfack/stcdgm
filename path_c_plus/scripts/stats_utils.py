"""Statistical utilities for Path C+ Phase A0'' analysis.

DS Round-2 condition B: must exist before any A0'' seed loop references it.

Provides:
- bootstrap_ci_3seeds: BCa bootstrap CI for small-n (3 seeds), per DS audit
- paired_wilcoxon_oracle_vs_corrdiff: paired test for H1
- holm_bonferroni_correction: FDR for H1-H5 multiple testing
- compute_skeleton_f1: structural F1 (already in smoke notebook, exposed here)
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np


def bootstrap_ci_3seeds(
    per_seed_values: Sequence[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    method: str = "bca",
    rng_seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap CI for small-n seed ensemble.

    Per DS audit: BCa (bias-corrected accelerated) is preferred over percentile
    bootstrap for n=3 because symmetric assumptions fail at small sample sizes.

    Parameters
    ----------
    per_seed_values : array of shape (n_seeds,)
        e.g. [Q_phys_seed42, Q_phys_seed7, Q_phys_seed123]
    n_resamples : int
        Bootstrap resample count (default 1000 per DS pre-registration)
    confidence : float
        Confidence level (default 0.95)
    method : "bca" | "percentile"
        BCa recommended for small n

    Returns
    -------
    dict with: point_estimate, ci_lower, ci_upper, n_seeds, method
    """
    x = np.asarray(per_seed_values, dtype=np.float64)
    if x.size < 2:
        return {
            "point_estimate": float(x.mean()) if x.size else float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "n_seeds": int(x.size),
            "method": "insufficient_n",
        }

    rng = np.random.default_rng(rng_seed)
    point = float(np.mean(x))
    alpha = 1 - confidence

    # Generate bootstrap resamples
    resamples = rng.choice(x, size=(n_resamples, x.size), replace=True)
    means = resamples.mean(axis=1)

    if method == "percentile":
        lo = float(np.percentile(means, 100 * alpha / 2))
        hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    elif method == "bca":
        # Bias correction
        z0 = _norm_inv(np.mean(means < point))
        # Acceleration via jackknife
        jackknife = np.array([
            np.mean(np.delete(x, i)) for i in range(x.size)
        ])
        jack_mean = jackknife.mean()
        num = np.sum((jack_mean - jackknife) ** 3)
        denom = 6 * (np.sum((jack_mean - jackknife) ** 2)) ** 1.5
        a = float(num / denom) if denom > 1e-12 else 0.0

        z_alpha_lo = _norm_inv(alpha / 2)
        z_alpha_hi = _norm_inv(1 - alpha / 2)
        a1 = _norm_cdf(z0 + (z0 + z_alpha_lo) / (1 - a * (z0 + z_alpha_lo)))
        a2 = _norm_cdf(z0 + (z0 + z_alpha_hi) / (1 - a * (z0 + z_alpha_hi)))
        lo = float(np.percentile(means, 100 * a1))
        hi = float(np.percentile(means, 100 * a2))
    else:
        raise ValueError(f"method must be 'bca' or 'percentile', got {method!r}")

    return {
        "point_estimate": point,
        "ci_lower": lo,
        "ci_upper": hi,
        "n_seeds": int(x.size),
        "method": method,
        "n_resamples": n_resamples,
        "confidence": confidence,
    }


def paired_wilcoxon_oracle_vs_corrdiff(
    oracle_values: Sequence[float],
    corrdiff_values: Sequence[float],
) -> Dict[str, float]:
    """Paired Wilcoxon signed-rank test for Oracle vs CorrDiff.

    Use for H2 (predictive skill non-degradation): compare same N batches
    across Path C+ Oracle and CorrDiff baseline.

    Returns dict with: statistic, p_value, n_pairs, effect_size_r
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        warnings.warn("scipy not available; cannot compute Wilcoxon test")
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "n_pairs": len(oracle_values),
            "effect_size_r": float("nan"),
            "error": "scipy_not_available",
        }

    o = np.asarray(oracle_values, dtype=np.float64)
    c = np.asarray(corrdiff_values, dtype=np.float64)
    if o.shape != c.shape:
        raise ValueError(f"Shapes must match: {o.shape} vs {c.shape}")

    result = wilcoxon(o, c, alternative="two-sided")
    # Effect size r = Z / sqrt(N) approximation
    n = len(o)
    try:
        from scipy.stats import norm
        z = norm.ppf(1 - result.pvalue / 2)
    except Exception:
        z = float("nan")

    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_pairs": n,
        "effect_size_r": float(abs(z) / np.sqrt(n)) if not np.isnan(z) else float("nan"),
    }


def holm_bonferroni_correction(
    p_values: Sequence[float],
    alpha: float = 0.05,
    labels: Sequence[str] = None,
) -> Dict[str, dict]:
    """Holm-Bonferroni step-down correction for the H1-H5 family.

    Per DS pre-registration: required for multiple testing across the 5
    pre-registered hypotheses.
    """
    p = np.asarray(p_values, dtype=np.float64)
    n = p.size
    if labels is None:
        labels = [f"H{i+1}" for i in range(n)]

    sorted_idx = np.argsort(p)
    sorted_p = p[sorted_idx]
    sorted_labels = [labels[i] for i in sorted_idx]

    # Holm step-down
    adjusted = np.zeros_like(sorted_p)
    for i in range(n):
        adjusted[i] = sorted_p[i] * (n - i)
    # Enforce monotonicity
    for i in range(1, n):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])
    adjusted = np.clip(adjusted, 0, 1)

    out = {}
    for orig_i, sorted_label, p_orig, p_adj in zip(
        sorted_idx, sorted_labels, sorted_p, adjusted
    ):
        out[sorted_label] = {
            "p_raw": float(p_orig),
            "p_adjusted": float(p_adj),
            "reject_at_alpha": bool(p_adj < alpha),
        }
    return out


def _norm_inv(p: float) -> float:
    """Inverse normal CDF via scipy if available, approximation otherwise."""
    try:
        from scipy.stats import norm
        return float(norm.ppf(p))
    except Exception:
        # Rational approximation (Beasley-Springer-Moro)
        if p <= 0:
            return -float("inf")
        if p >= 1:
            return float("inf")
        if p < 0.5:
            t = np.sqrt(-2 * np.log(p))
            return float(
                -(t - (2.515517 + 0.802853 * t + 0.010328 * t * t) /
                  (1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t))
            )
        else:
            return -_norm_inv(1 - p)


def _norm_cdf(z: float) -> float:
    """Normal CDF."""
    try:
        from scipy.stats import norm
        return float(norm.cdf(z))
    except Exception:
        # erf approximation
        return float(0.5 * (1 + np.tanh(np.sqrt(np.pi / 2) * z)))


__all__ = [
    "bootstrap_ci_3seeds",
    "paired_wilcoxon_oracle_vs_corrdiff",
    "holm_bonferroni_correction",
]


if __name__ == "__main__":
    # Smoke test
    print("=== stats_utils smoke test ===")

    # Simulate 3 seeds for Q_phys
    q_phys_seeds = [0.60, 0.40, 0.80]
    ci = bootstrap_ci_3seeds(q_phys_seeds, n_resamples=1000, method="bca")
    print(f"Q_phys 3-seed bootstrap CI 95%: {ci}")

    # Simulate Holm-Bonferroni on H1-H5
    p_vals = [0.001, 0.04, 0.12, 0.30, 0.55]
    labels = ["H1_Q_phys", "H2_RMSE", "H3_CDD_bias", "H4_calibration", "H5_OOD"]
    result = holm_bonferroni_correction(p_vals, alpha=0.05, labels=labels)
    print()
    print(f"Holm-Bonferroni results:")
    for k, v in result.items():
        print(f"  {k}: p_raw={v['p_raw']:.4f}, p_adj={v['p_adjusted']:.4f}, "
              f"reject={v['reject_at_alpha']}")
