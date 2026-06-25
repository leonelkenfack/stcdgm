"""
V6 MVP canonical constants (hardcoded, not derived from CONFIG).

Following the memory rule ``feedback_hardcode_canonical_lists_phase8`` :
NEVER capture lr_variables from a mutable CONFIG state. Always hardcode the
canonical list, fail loudly if YAML diverges.

Source : `path_c_plus/audit/PLAN_V6_BOOST_UNET.md` §1.2 + S2.1 preproc script.
"""

from __future__ import annotations

# --- 15 NONCAUSAL_15_VARS (baseline 9-node seed 42) ----------------------- #
NONCAUSAL_15_VARS = [
    "GP850",
    "GP500",
    "GP250",
    "Q850",
    "Q500",
    "T850",
    "T500",
    "U850",
    "U500",
    "U250",
    "V850",
    "V500",
    "V250",
    "W850",
    "W500",
]

# --- 6 V6 features Climat (obligatoires) --------------------------------- #
V6_CLIMAT_OBLIGATORY = [
    "w_700",
    "theta_e_850",
    "theta_e_500",
    "mucape_proxy",
    "T_850_minus_T_500",
    "u850_grad_orog_HR_LR_repr",
]

# --- 1 V6 feature optionnelle (§4.4) ------------------------------------- #
V6_CLIMAT_OPTIONAL = [
    "theta_w_850",
]

# --- V6 LR vars total (obligatoires) ------------------------------------- #
V6_LR_VARS_OBLIGATORY = NONCAUSAL_15_VARS + V6_CLIMAT_OBLIGATORY  # 15 + 6 = 21

# --- V6 LR vars complet (avec optionnel) --------------------------------- #
V6_LR_VARS_FULL = V6_LR_VARS_OBLIGATORY + V6_CLIMAT_OPTIONAL  # 22

# --- V6 graph node types ------------------------------------------------- #
# Existing 9-node : GP850, GP500, GP250, Q850, W500, IVT (6 dynamic) + SP_HR
# V6 adds : U850, V850 (2 new dynamic) → 8 dynamic + SP_HR
V6_DYNAMIC_NODE_TYPES = [
    "GP850", "GP500", "GP250",          # always-on (mid_layer)
    "Q850", "W500", "IVT",              # extended_9node
    "U850", "V850",                     # extended_v6_wind (V6 MVP)
]
V6_NUM_DYNAMIC_NODES = len(V6_DYNAMIC_NODE_TYPES)  # 8 → A_dag ∈ R^{8×8} (not 11×11)

# --- Hyperparam adaptation 11-node → V6 wind (Climat ronde 3 : λ_l1 -30%) - #
# Original 9-node seed 42 : λ_l1 cosine 0.04 → 0.005
# V6 11-node              : λ_l1 cosine 0.028 → 0.0035 (×0.7)
V6_LAMBDA_L1_START = 0.028   # was 0.04
V6_LAMBDA_L1_END = 0.0035    # was 0.005

# All other hyperparams unchanged from 9-node seed 42 :
V6_LAMBDA_DAG_PRIOR = 0.40           # unchanged (MVP simplicity)
V6_GAMMA_DAG = 0.10                  # unchanged
V6_DAG_GRAD_GATE_MAX = 1.0           # unchanged
V6_DAG_FLOOR_MIN_NORM = 0.10         # unchanged
V6_ABORT_ON_COLLAPSE = True          # unchanged
V6_COLLAPSE_THRESHOLD = 0.05         # unchanged
V6_DAG_SPECTRAL_PROJECTION = True    # unchanged


def assert_lr_vars_match(yaml_lr_vars: list[str], require_optional: bool = False) -> None:
    """Raise if the YAML lr_variables list diverges from V6 canonical.

    Parameters
    ----------
    yaml_lr_vars : list[str]
        List captured from a config YAML.
    require_optional : bool
        If True, also expects ``V6_CLIMAT_OPTIONAL`` features. Default False
        (= V6 MVP obligatoire only).
    """
    expected = V6_LR_VARS_FULL if require_optional else V6_LR_VARS_OBLIGATORY
    if set(yaml_lr_vars) != set(expected):
        missing = set(expected) - set(yaml_lr_vars)
        extra = set(yaml_lr_vars) - set(expected)
        raise RuntimeError(
            f"YAML lr_variables diverges from V6 canonical.\n"
            f"  Expected ({len(expected)}): {sorted(expected)}\n"
            f"  YAML     ({len(yaml_lr_vars)}): {sorted(yaml_lr_vars)}\n"
            f"  Missing : {sorted(missing)}\n"
            f"  Extra   : {sorted(extra)}"
        )


__all__ = [
    "NONCAUSAL_15_VARS",
    "V6_CLIMAT_OBLIGATORY",
    "V6_CLIMAT_OPTIONAL",
    "V6_LR_VARS_OBLIGATORY",
    "V6_LR_VARS_FULL",
    "V6_DYNAMIC_NODE_TYPES",
    "V6_NUM_DYNAMIC_NODES",
    "V6_LAMBDA_L1_START",
    "V6_LAMBDA_L1_END",
    "V6_LAMBDA_DAG_PRIOR",
    "V6_GAMMA_DAG",
    "V6_DAG_GRAD_GATE_MAX",
    "V6_DAG_FLOOR_MIN_NORM",
    "V6_ABORT_ON_COLLAPSE",
    "V6_COLLAPSE_THRESHOLD",
    "V6_DAG_SPECTRAL_PROJECTION",
    "assert_lr_vars_match",
]
