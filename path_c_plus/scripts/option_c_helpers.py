"""path_c_plus/scripts/option_c_helpers.py

Helpers for Option C : 3-seeds Path C+ Oracle retrain (Stage 1 + Stage 2)
matching CorrDiff Normal V2 config, with A1 acquis ported.

Lineage : ports the A1 notebook (commit 78a3783) statistical and audit logic
into a reusable module that the Option C notebook can import. Adds H2-H5
cross-seed analysis vs the existing ckpt_noncausal/ baseline.

References :
- PRE_REGISTRATION.md PC4 / PC5 / PC6 / PC8 / PC10 / PC12 / PC13
- A1 notebook cells 4-7 (helpers + verdict tree)
- NONCAUSAL_EVAL_CONTRACT.md (eval pipeline contract)
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Constants
# =============================================================================

# PC13 allowlist : NEW Path C+ params that V5_DIR-style ckpts (pre-fix) cannot
# have. From-scratch training does NOT use PC13 -- it is here only for resume
# scenarios where a partial Path C+ ckpt is loaded mid-run.
PC13_NEW_PARAMS_ALLOWLIST: List[str] = [
    "metapath_convs",   # §1.6 per-metapath SAGEConvs (after legacy migration)
    "layer_norms",      # J3   per-metapath LayerNorms (after legacy migration)
    "_state_adapter",   # J1   eager Linear in regression_head
]

# Path C+ hyperparameter overrides applied ON TOP of training_config_corrdiff_normal.yaml.
# These are the values that A1 used and produced Q_phys_cont = 0.521.
PATHCPLUS_HYPERPARAM_OVERRIDES: Dict[str, Any] = {
    "lambda_dag_prior": 0.40,    # was 0.05 (V5-mini) -- §1.8/I1 fix
    "lambda_l1_start": 0.04,     # was 0.001 (V5-mini) -- schedule_lambdas start
    "lambda_l1_end": 0.005,      # was 0.001 (V5-mini) -- schedule_lambdas end
    "g_phys_alpha": 0.25,        # was 0.20 (V5-mini)
    # dag_grad_gate : None means auto-scale ramp (per Batch F-1 fix)
    "dag_gate_warmup_start_epoch": None,
    "dag_gate_warmup_end_epoch": None,
}


# =============================================================================
# Q_phys metrics (4 variants -- A1 acquis)
# =============================================================================

def compute_q_phys_binary(
    A_dag_np: np.ndarray,
    G_phys_np: np.ndarray,
    threshold: float = 0.01,
) -> Tuple[float, int, int, int]:
    """BINARY Q_phys (= V5-mini original metric).

    Returns (Q_phys, matches, n_phys, n_extra).
    """
    A = np.array(A_dag_np)
    np.fill_diagonal(A, 0.0)
    G = np.array(G_phys_np)
    mask = G != 0
    if not mask.any():
        return 0.0, 0, 0, 0
    A_thresh = A.copy()
    A_thresh[np.abs(A_thresh) <= threshold] = 0.0
    matches = int(((np.sign(A_thresh) == np.sign(G)) & mask).sum())
    n_phys = int(mask.sum())
    n_extra = int(
        ((G == 0) & (np.abs(A) > threshold) & (~np.eye(A.shape[0], dtype=bool))).sum()
    )
    return matches / n_phys, matches, n_phys, n_extra


def compute_q_phys_adaptive(
    A_dag_np: np.ndarray,
    G_phys_np: np.ndarray,
    frac: float = 0.3,
) -> Tuple[float, int, int, float]:
    """Math Prof Q1 : adaptive-threshold binary Q_phys.

    threshold = max(0.01, frac * max(|A_dag|))
    DAGMA/NOTEARS convention : a "recovered" edge must have magnitude
    proportional to the largest non-zero entry.

    Returns (Q_phys, matches, n_phys, threshold).
    """
    A = np.array(A_dag_np)
    np.fill_diagonal(A, 0.0)
    G = np.array(G_phys_np)
    threshold = max(0.01, frac * float(np.abs(A).max()))
    mask = G != 0
    if not mask.any():
        return 0.0, 0, 0, threshold
    A_thresh = A.copy()
    A_thresh[np.abs(A_thresh) <= threshold] = 0.0
    matches = int(((np.sign(A_thresh) == np.sign(G)) & mask).sum())
    return matches / int(mask.sum()), matches, int(mask.sum()), threshold


def compute_q_phys_continuous(
    A_dag_np: np.ndarray,
    G_phys_np: np.ndarray,
    *,
    collapse_eps: float = 1e-12,
) -> Tuple[float, bool]:
    """Math Prof Q4 : magnitude ratio Q_phys (PRIMARY ENDPOINT for H1, PC5).

    Q_phys_cont = sum(|A[phys positions, sign-correct]|) / sum(|A[off-diag]|)
    Range [0, 1]. Gaming-resistant.

    Returns (Q_phys_cont, collapsed_flag). collapsed=True when |A|.sum < eps.
    """
    A = np.array(A_dag_np)
    np.fill_diagonal(A, 0.0)
    G = np.array(G_phys_np)
    mask_phys = G != 0
    sign_correct = (np.sign(A) == np.sign(G)) & mask_phys
    num = float(np.abs(A[sign_correct]).sum())
    den = float(np.abs(A).sum())
    if den < collapse_eps:
        return 0.0, True
    return num / den, False


def compute_phys_mag_gained(
    A_dag_now: np.ndarray,
    A_dag_init: np.ndarray,
    G_phys_np: np.ndarray,
) -> float:
    """Blindspot #2+#3 : phys-edge mass gained (signed).

    Returns sum(|A_now[phys, sign-correct]|) - sum(|A_init[phys, sign-correct]|).
    Positive when learning has REINFORCED sign-correct physical edges.
    """
    A_now = np.array(A_dag_now)
    A_init = np.array(A_dag_init)
    G = np.array(G_phys_np)
    np.fill_diagonal(A_now, 0.0)
    np.fill_diagonal(A_init, 0.0)
    mask_phys = G != 0
    sc_now = (np.sign(A_now) == np.sign(G)) & mask_phys
    sc_init = (np.sign(A_init) == np.sign(G)) & mask_phys
    return float(np.abs(A_now[sc_now]).sum() - np.abs(A_init[sc_init]).sum())


def compute_skeleton_f1(
    A_dag_np: np.ndarray,
    G_phys_np: np.ndarray,
    *,
    frac: float = 0.3,
) -> Tuple[float, float]:
    """Adaptive-threshold skeleton F1 (sign-agnostic undirected match).

    Returns (skeleton_f1, threshold_used).
    """
    A = np.array(A_dag_np)
    np.fill_diagonal(A, 0.0)
    G = np.array(G_phys_np)
    threshold = max(0.01, frac * float(np.abs(A).max()))
    A_skel = ((np.abs(A) > threshold) | (np.abs(A.T) > threshold)).astype(int)
    np.fill_diagonal(A_skel, 0)
    G_skel = ((G != 0) | (G.T != 0)).astype(int)
    np.fill_diagonal(G_skel, 0)
    iu = np.triu_indices_from(A_skel, k=1)
    A_e = A_skel[iu]
    G_e = G_skel[iu]
    tp = int(((A_e == 1) & (G_e == 1)).sum())
    fp = int(((A_e == 1) & (G_e == 0)).sum())
    fn = int(((A_e == 0) & (G_e == 1)).sum())
    if tp + fp == 0 or tp + fn == 0:
        return 0.0, threshold
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    if p + r == 0:
        return 0.0, threshold
    return 2 * p * r / (p + r), threshold


# =============================================================================
# Projection hook (3-point : pre_spectral / post_spectral / post_floor)
# =============================================================================

def install_projection_hook(rcn_cell) -> Dict[str, list]:
    """Monkey-patch rcn_cell.project_dag_spectral + project_dag_floor.

    Returns the projection_log dict that will be populated per batch.
    The hook records A_dag at 3 points :
      pre_spectral  : optimizer.step() output (what the optimizer produced)
      post_spectral : after project_dag_spectral (top-of-norm cap)
      post_floor    : after project_dag_floor (anti-collapse floor)

    The 3-point structure lets us distinguish "what learning did" from
    "what the projector enforced".
    """
    log: Dict[str, list] = {
        "pre_spectral_A_dag": [],
        "post_spectral_A_dag": [],
        "post_floor_A_dag": [],
        "spectral_rescale": [],
        "floor_rescale": [],
    }
    _orig_spectral = rcn_cell.project_dag_spectral
    _orig_floor = rcn_cell.project_dag_floor

    def _patched_spectral(max_radius=0.95):
        pre = rcn_cell.A_dag.data.detach().cpu().clone().numpy()
        rescale = _orig_spectral(max_radius=max_radius)
        post = rcn_cell.A_dag.data.detach().cpu().clone().numpy()
        log["pre_spectral_A_dag"].append(pre)
        log["post_spectral_A_dag"].append(post)
        log["spectral_rescale"].append(float(rescale))
        return rescale

    def _patched_floor(min_norm=0.10, prior=None):
        rescale = _orig_floor(min_norm=min_norm, prior=prior)
        post = rcn_cell.A_dag.data.detach().cpu().clone().numpy()
        log["post_floor_A_dag"].append(post)
        log["floor_rescale"].append(float(rescale))
        return rescale

    rcn_cell.project_dag_spectral = _patched_spectral
    rcn_cell.project_dag_floor = _patched_floor
    return log


# =============================================================================
# PC4 + PC13 audit gate
# =============================================================================

def _key_is_pc13_eligible(key: str) -> bool:
    """Returns True iff the key starts with any PC13-allowlisted prefix."""
    return any(
        key.startswith(p) or f".{p}" in key
        for p in PC13_NEW_PARAMS_ALLOWLIST
    )


def check_pc4_pc13_gate(
    load_audit: Dict[str, Any],
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """PC4 + PC13 combined gate.

    PC4 BLOCKING :
      - encoder_state_dict_n_keys == 0           -> EXCLUDE
      - 'diffusion' in modules_loaded_fallback   -> EXCLUDE

    PC13 SOFT-PASS :
      For modules in fallback (excluding 'diffusion') :
        - unexpected_keys == []                                 -> required
        - ALL missing_keys in PC13_NEW_PARAMS_ALLOWLIST          -> required
      Otherwise -> HARD BLOCK (smoke #2 pattern).

    Returns (pc4_ok, exclusion_reason, pc13_softpass).
    For from-scratch training (no warm-start), load_audit may be empty/None
    and this function returns (True, None, None).
    """
    if not load_audit:
        # From-scratch training : no warm-start = no PC4/PC13 to check.
        return True, None, None

    fallback = list(load_audit.get("modules_loaded_fallback", []))
    n_keys = load_audit.get("encoder_state_dict_n_keys", 0)

    if n_keys == 0:
        return False, {
            "reason": "encoder_state_dict empty (PC4 §123)",
            "modules_loaded_fallback": fallback,
            "encoder_n_keys": n_keys,
        }, None

    if "diffusion" in fallback:
        return False, {
            "reason": ("diffusion loaded with strict=False -- J29 is runtime, "
                       "not a weight (PC4 §128 + PC13 ineligibility)."),
            "modules_loaded_fallback": fallback,
            "encoder_n_keys": n_keys,
        }, None

    pc13_evidence: Dict[str, Any] = {}
    for mod_name in fallback:
        if mod_name == "diffusion":
            continue
        missing = load_audit.get("missing_keys_per_module", {}).get(mod_name, [])
        unexpected = load_audit.get("unexpected_keys_per_module", {}).get(mod_name, [])

        if unexpected:
            return False, {
                "reason": (f"PC13 hard block : module '{mod_name}' has "
                           f"{len(unexpected)} unexpected_keys -- smoke #2 pattern."),
                "module": mod_name,
                "unexpected_keys_sample": list(unexpected)[:5],
                "missing_keys_sample": list(missing)[:5],
                "modules_loaded_fallback": fallback,
                "encoder_n_keys": n_keys,
            }, None

        non_allowed = [k for k in missing if not _key_is_pc13_eligible(k)]
        if non_allowed:
            return False, {
                "reason": (f"PC13 hard block : module '{mod_name}' has "
                           f"{len(non_allowed)} missing_keys NOT in PC13 allowlist."),
                "module": mod_name,
                "non_allowed_missing_sample": non_allowed[:5],
                "missing_keys_total": len(missing),
                "modules_loaded_fallback": fallback,
                "encoder_n_keys": n_keys,
            }, None

        pc13_evidence[mod_name] = {
            "missing_keys_count": len(missing),
            "missing_keys_sample": list(missing)[:5],
            "all_matched_allowlist": True,
        }

    return True, None, (pc13_evidence if pc13_evidence else None)


# =============================================================================
# H1 statistical analysis (PC5 / PC6 / PC8 / PC12 verdict tree)
# =============================================================================

def _bca_ci_n3_safe(
    values: List[float],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    rng_seed: int = 42,
) -> Tuple[float, float]:
    """BCa CI with n=3 guards (Math Prof PASS-WITH-NITS fix).

    Clips z0 to [-3, 3] and a1/a2 to [1e-3, 1-1e-3] to prevent NaN from
    degenerate n=3 bootstrap distribution. The AND-gate with Student-t
    in compute_h1_verdict keeps the test conservative.
    """
    from scipy import stats as _stats

    rng = np.random.default_rng(rng_seed)
    x = np.array(values, dtype=float)
    alpha = 1.0 - confidence

    # Bootstrap resample means
    means = np.array([
        rng.choice(x, size=x.size, replace=True).mean()
        for _ in range(n_resamples)
    ])
    point = float(x.mean())

    # Bias correction with clip
    z0 = float(_stats.norm.ppf(np.mean(means < point)))
    z0 = float(np.clip(z0, -3.0, 3.0))

    # Acceleration via jackknife
    jackknife = np.array([np.mean(np.delete(x, i)) for i in range(x.size)])
    jack_mean = jackknife.mean()
    num = np.sum((jack_mean - jackknife) ** 3)
    denom = 6 * (np.sum((jack_mean - jackknife) ** 2)) ** 1.5
    a = float(num / denom) if denom > 1e-12 else 0.0

    z_lo = _stats.norm.ppf(alpha / 2)
    z_hi = _stats.norm.ppf(1 - alpha / 2)
    a1 = _stats.norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
    a2 = _stats.norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))
    a1 = float(np.clip(a1, 1e-3, 1 - 1e-3))
    a2 = float(np.clip(a2, 1e-3, 1 - 1e-3))

    lo = float(np.percentile(means, 100 * a1))
    hi = float(np.percentile(means, 100 * a2))
    return lo, hi


def compute_h1_verdict(
    q_cont_per_seed: List[float],
    n_extra_per_seed: List[int],
    collapsed_per_seed: List[bool],
    *,
    baseline_cont: float = 0.04,
    random_null: float = 0.083,
    h1_threshold: float = 0.50,
    reporting_floor: float = 0.30,
    sd_floor: float = 0.05,
) -> Dict[str, Any]:
    """Compute H1 verdict per PC5/PC6/PC8/PC12 amendments.

    Returns dict with verdict + all gate diagnostics. Verdict values :
      - "PROTOCOL_FAILURE_COLLAPSE" (PC5-bis : >=2/3 seeds collapsed)
      - "H1_PASS_SPARSE"             (PC5 strict + n_extra < 3)
      - "H1_PASS_INTERVENTIONAL_ONLY" (PC5 strict + n_extra >= 3, PC6 row 2)
      - "BELOW_H1_ABOVE_FLOOR"       (PC12 reporting category, NOT acceptance)
      - "FAIL"                       (below floor)
    """
    from scipy import stats as _stats

    n = len(q_cont_per_seed)
    if n < 3:
        raise ValueError(f"compute_h1_verdict requires n=3 seeds, got {n}")

    mean_cont = float(np.mean(q_cont_per_seed))
    sd_cont = float(np.std(q_cont_per_seed, ddof=1))
    sd_floored = max(sd_cont, sd_floor)
    cohens_d = (mean_cont - baseline_cont) / sd_floored
    t_crit = _stats.t.ppf(0.95, df=n - 1)
    student_t_lower = mean_cont - t_crit * sd_floored / (n ** 0.5)
    bca_lower, bca_upper = _bca_ci_n3_safe(q_cont_per_seed)

    collapse_failure = sum(collapsed_per_seed) >= 2  # PC5-bis
    h1_strict_pass = (
        mean_cont >= h1_threshold
        and bca_lower > baseline_cont
        and student_t_lower > baseline_cont
        and bca_lower > random_null
        and cohens_d >= 0.8
    )
    sparse_recovery = max(n_extra_per_seed) < 3  # PC6

    if collapse_failure:
        verdict = "PROTOCOL_FAILURE_COLLAPSE"
        msg = (f"PC5-bis : {sum(collapsed_per_seed)}/{n} seeds COLLAPSED. "
               f"H1 test REPLACED by protocol-failure report. "
               f"Re-tune lambda_l1 to 0.02 (PC9 allowed once).")
    elif h1_strict_pass and sparse_recovery:
        verdict = "H1_PASS_SPARSE"
        msg = ("H1 ACCEPTED with sparse structural recovery. "
               "Thesis claim : interventional + sparse structural recovery.")
    elif h1_strict_pass:
        verdict = "H1_PASS_INTERVENTIONAL_ONLY"
        msg = (f"H1 met under primary endpoint, structural sparsity NOT achieved "
               f"(n_extra={max(n_extra_per_seed)} >= 3). "
               f"PC6 row 2 : 'interventional sign-consistency achieved; "
               f"sparse structural recovery not achieved'.")
    elif mean_cont >= reporting_floor and bca_lower > baseline_cont:
        verdict = "BELOW_H1_ABOVE_FLOOR"
        msg = (f"PC12 reporting category. Q_phys_cont={mean_cont:.4f} "
               f"({mean_cont/baseline_cont:.1f}x baseline) but H1 NOT met. "
               f"Thesis : 'preliminary evidence, H1 not statistically accepted'.")
    else:
        verdict = "FAIL"
        msg = (f"Q_phys_cont={mean_cont:.4f} below PC12 floor ({reporting_floor}). "
               f"No evidential claim possible. Investigate per PC3.")

    return {
        "verdict": verdict,
        "verdict_message": msg,
        "mean_q_phys_continuous": mean_cont,
        "sd_q_phys_continuous": sd_cont,
        "sd_q_phys_continuous_floored": sd_floored,
        "cohens_d_vs_baseline": cohens_d,
        "bca_ci_95": [bca_lower, bca_upper],
        "student_t_lower_95": float(student_t_lower),
        "baseline_cont": baseline_cont,
        "random_null": random_null,
        "h1_threshold": h1_threshold,
        "reporting_floor_pc12": reporting_floor,
        "h1_strict_pass": h1_strict_pass,
        "sparse_recovery_pc6": sparse_recovery,
        "collapse_failure_pc5bis": collapse_failure,
        "n_seeds": n,
    }


# =============================================================================
# H2-H5 paired comparison vs noncausal baseline
# =============================================================================

def load_noncausal_baseline_metrics(ckpt_noncausal_dir: Path) -> Dict[str, Any]:
    """Load all baseline JSONs from ckpt_noncausal directory.

    Returns nested dict :
      {
        "final_validation_metrics": {...},
        "domain_metrics": {...},
        "aligned_metrics_per_gcm": {"ACCESS-CM2": {...}, "EC-Earth3": {...}, "NorESM2-MM": {...}},
        "probabilistic_metrics_per_gcm": {...},
      }
    """
    d = Path(ckpt_noncausal_dir)
    result: Dict[str, Any] = {}

    fv = d / "final_validation_metrics.json"
    if fv.exists():
        result["final_validation_metrics"] = json.loads(fv.read_text())

    dm = d / "domain_metrics.json"
    if dm.exists():
        result["domain_metrics"] = json.loads(dm.read_text())

    aligned: Dict[str, Any] = {}
    for gcm in ["ACCESS-CM2", "EC-Earth3", "NorESM2-MM"]:
        p = d / f"aligned_metrics_{gcm}_noncausal.json"
        if p.exists():
            aligned[gcm] = json.loads(p.read_text())
    if aligned:
        result["aligned_metrics_per_gcm"] = aligned

    proba: Dict[str, Any] = {}
    for gcm in ["ACCESS-CM2", "EC-Earth3", "NorESM2-MM"]:
        p = d / f"probabilistic_metrics_{gcm}_noncausal.json"
        if p.exists():
            proba[gcm] = json.loads(p.read_text())
    if proba:
        result["probabilistic_metrics_per_gcm"] = proba

    return result


def paired_t_h2_h5(
    oracle_per_seed: List[float],
    noncausal_value: float,
    *,
    label: str,
    delta_threshold: Optional[float] = None,
    direction: str = "lower_is_better",
) -> Dict[str, Any]:
    """Single-sample paired comparison : Path C+ Oracle (3 seeds) vs noncausal scalar.

    With n=3 we use Student-t one-sided (df=2) since paired Wilcoxon needs
    n_pairs>=6 for any non-trivial p-value.

    Parameters
    ----------
    direction : "lower_is_better" (MAE, CRPS, distance) or "higher_is_better"
                (Pearson, F1).
    delta_threshold : if provided, also report whether the difference is
                      smaller than this delta (non-degradation test).
    """
    from scipy import stats as _stats

    n = len(oracle_per_seed)
    if n < 2:
        raise ValueError(f"paired_t_h2_h5 needs n>=2, got {n}")

    diffs = [v - noncausal_value for v in oracle_per_seed]
    mean_diff = float(np.mean(diffs))
    sd_diff = max(float(np.std(diffs, ddof=1)), 1e-9)
    t_stat = mean_diff / (sd_diff / (n ** 0.5))

    if direction == "lower_is_better":
        # H_a : Oracle < noncausal -> diff < 0 -> t < 0
        p_value = float(_stats.t.cdf(t_stat, df=n - 1))
    else:
        # H_a : Oracle > noncausal -> diff > 0 -> t > 0
        p_value = float(1.0 - _stats.t.cdf(t_stat, df=n - 1))

    non_degradation = None
    if delta_threshold is not None:
        # Non-degradation : |oracle - noncausal| <= delta_threshold
        non_degradation = float(abs(mean_diff)) <= delta_threshold

    return {
        "label": label,
        "n_seeds": n,
        "oracle_per_seed": list(oracle_per_seed),
        "oracle_mean": float(np.mean(oracle_per_seed)),
        "oracle_sd": float(np.std(oracle_per_seed, ddof=1)),
        "noncausal_value": noncausal_value,
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "t_stat": float(t_stat),
        "p_value_one_sided": p_value,
        "direction": direction,
        "delta_threshold": delta_threshold,
        "non_degradation_within_delta": non_degradation,
    }


def holm_bonferroni_h1_h5(
    p_values: Dict[str, float],
    *,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Holm-Bonferroni correction across H1-H5 family (k=5).

    Returns dict mapping each hypothesis -> {p_raw, p_adjusted, reject}.
    Sorted by p_raw ascending. Reject if p_adjusted < alpha.
    """
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    k = len(items)
    out: Dict[str, Any] = {}
    for i, (name, p) in enumerate(items):
        adj_factor = k - i
        p_adj = min(1.0, p * adj_factor)
        out[name] = {
            "p_raw": float(p),
            "p_adjusted": float(p_adj),
            "reject_at_alpha": bool(p_adj < alpha),
            "rank": i + 1,
            "adjustment_factor": adj_factor,
        }
    return out


# =============================================================================
# Stamp helper (Option C variant)
# =============================================================================

def stamp_option_c_json(
    results_dict: Dict[str, Any],
    *,
    seed: Optional[int] = None,
    seeds_lineage: Optional[List[int]] = None,
    k9_dates: Optional[Dict[str, List[str]]] = None,
    pathcplus_hyperparams: Optional[Dict[str, Any]] = None,
    pre_registration_commit: Optional[str] = None,
    is_aggregate: bool = False,
) -> Dict[str, Any]:
    """Stamp result JSON for Option C (Path C+ Oracle retrain from-scratch).

    schema_version = "path-c-plus-option-c-v1"
    valid_for_analysis = True (PC4 gate must have been verified upstream)
    """
    results_dict["schema_version"] = "path-c-plus-option-c-v1"
    results_dict["path_c_plus_batch"] = "option-c"
    results_dict["option_c_run_type"] = (
        "aggregate" if is_aggregate else f"seed_{seed}"
    )
    results_dict["valid_for_analysis"] = True

    if seeds_lineage is not None:
        results_dict["seeds_lineage"] = list(seeds_lineage)
    if k9_dates is not None:
        results_dict["k9_temporal_split"] = copy.deepcopy(k9_dates)
    if pathcplus_hyperparams is not None:
        results_dict["pathcplus_hyperparams"] = copy.deepcopy(pathcplus_hyperparams)
    if pre_registration_commit is not None:
        results_dict["pre_registration_commit"] = pre_registration_commit

    # Lineage : same as Batch D stamp helper for auditability
    results_dict["path_c_plus_full_fix_lineage"] = {
        "batch_A": "consensus 10 fixes (section 1.6, J3, I11, J8, ...)",
        "batch_B": "math prof 16 fixes (I1-I16)",
        "batch_C": "AI eng 42 fixes (J1-J42, subset of P0 picked)",
        "batch_D": ["J29", "K2", "K3", "K9", "K5"],
        "batch_E": ["J29 default", "K2 plumbing", "K5 raise", "K9 assert", "tombstone schema"],
        "batch_F": ["DEFAULT_HYPERPARAMS gate", "Q_phys variants", "projection hook",
                    "sigma_data calib", "PC5-PC8"],
        "batch_F_bis": ["3-point projection hook", "PC5/PC8 amendments",
                        "smoke JSON HARKing-resistant tag"],
        "batch_F_blindspots": ["#1 path_b doc", "#2-3 phys_mag_gained criterion",
                                "#6 collapse explicit", "#7 rename",
                                "#8 fetch depth 200", "#9 adaptive skeleton F1"],
        "option_c": ["full Stage 1+2 from-scratch", "CorrDiff Normal V2 config",
                     "Path C+ hyperparam override", "K9 temporal split",
                     "3 seeds", "noncausal baseline paired comparison"],
    }
    return results_dict


__all__ = [
    "PC13_NEW_PARAMS_ALLOWLIST",
    "PATHCPLUS_HYPERPARAM_OVERRIDES",
    "compute_q_phys_binary",
    "compute_q_phys_adaptive",
    "compute_q_phys_continuous",
    "compute_phys_mag_gained",
    "compute_skeleton_f1",
    "install_projection_hook",
    "check_pc4_pc13_gate",
    "compute_h1_verdict",
    "load_noncausal_baseline_metrics",
    "paired_t_h2_h5",
    "holm_bonferroni_h1_h5",
    "stamp_option_c_json",
]
