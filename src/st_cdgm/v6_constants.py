"""
V6 MVP canonical constants (hardcoded, not derived from CONFIG).

Following the memory rule ``feedback_hardcode_canonical_lists_phase8`` :
NEVER capture lr_variables from a mutable CONFIG state. Always hardcode the
canonical list, fail loudly if YAML diverges.

Source : `path_c_plus/audit/PLAN_V6_BOOST_UNET.md` §1.2 + S2.1 preproc script.
"""

from __future__ import annotations

# --- 15 NONCAUSAL_15_VARS (baseline 9-node seed 42) ----------------------- #
# CORRIGÉ (audit V6' full, 2026-06-30) : ce sont les noms RÉELS des canaux LR
# atmosphériques (u/v/w/q/t à 850/500/250 hPa), PAS les node-types du graphe.
# Vérifié contre config/training_config.yaml + corrdiff_normal.
# L'ancienne liste (GP850, Q850, ...) était fabriquée et aurait planté au
# runtime (lr_grid_to_nodes indexe par ces noms de canaux).
NONCAUSAL_15_VARS = [
    "u_850", "u_500", "u_250",
    "v_850", "v_500", "v_250",
    "w_850", "w_500", "w_250",
    "q_850", "q_500", "q_250",
    "t_850", "t_500", "t_250",
]

# --- 6 V6' features Climat (obligatoires — noms POST-fixes P1/P3/P4) ------ #
# P3 : conditional_instability = θe850 − θe*_sat500 (remplace mucape_proxy)
# P1 : w_orog_signed_LR_repr = (u,v)·∇h signé (remplace u850·|∇h|)
# P4 : theta_w_850 DROPPÉ (approximation fausse de +14 K)
V6_CLIMAT_OBLIGATORY = [
    "w_700",
    "theta_e_850",
    "theta_e_500",
    "conditional_instability",
    "T_850_minus_T_500",
    "w_orog_signed_LR_repr",
]

# --- Bonus V6' (audit Climat validation finale) — persistance AR ---------- #
V6_CLIMAT_BONUS = [
    "ivt_persistence_72h",
]

# --- V6 LR vars total (obligatoires) ------------------------------------- #
V6_LR_VARS_OBLIGATORY = NONCAUSAL_15_VARS + V6_CLIMAT_OBLIGATORY  # 15 + 6 = 21

# --- V6' LR vars complet (avec bonus IVT-72h) ----------------------------- #
V6_LR_VARS_FULL = V6_LR_VARS_OBLIGATORY + V6_CLIMAT_BONUS  # 22

# --- V6' Stage 2 conditioning ---------------------------------------------- #
# Nombre de canaux LR concaténés dans l'UNet de diffusion (pivot V6').
# = len(V6_LR_VARS_FULL) — UNet_in = [c_in·y_noisy, mu_HR, baseline] + ces canaux.
V6_PRIME_LR_CONDITIONING_CHANNELS = len(V6_LR_VARS_FULL)  # 22

# --- V6 graph node types (builder.dynamic_node_types) -------------------- #
# extended_9node : GP850, GP500, GP250, Q850, W500, IVT (6 dynamic)
# extended_v6_wind : + U850, V850 → 8 dynamic node types (+ SP_HR static)
V6_DYNAMIC_NODE_TYPES = [
    "GP850", "GP500", "GP250",          # always-on (mid_layer)
    "Q850", "W500", "IVT",              # extended_9node
    "U850", "V850",                     # extended_v6_wind
]
V6_NUM_DYNAMIC_NODES = len(V6_DYNAMIC_NODE_TYPES)  # 8 builder node types
# NOTE : num_vars (encoder intelligible variables = metapaths + static SP_HR)
# = 5 base GP metapaths + 3 humid (Q850/W500/IVT) + 2 wind (U850/V850) + 1
# static SP_HR = 11 → A_dag ∈ R^{11×11}, aligned with physics_prior.VAR_LABELS_V6.
V6_NUM_ENCODER_VARS = 11

# Per-node channel routing (V6) : which raw LR channels feed each humid/wind node
V6_NODE_CHANNEL_ROUTING = {
    "Q850": ["q_850", "q_500", "q_250"],   # humidity at 3 levels
    "W500": ["w_850", "w_500", "w_250"],   # vertical velocity at 3 levels
    "U850": ["u_850", "u_500", "u_250"],   # zonal wind at 3 levels
    "V850": ["v_850", "v_500", "v_250"],   # meridional wind at 3 levels
    # IVT : derived proxy ; GP850/GP500/GP250 : full LR
}

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
        If True, also expects ``V6_CLIMAT_BONUS`` features (ivt_persistence_72h).
        Default False (= 21 vars obligatoires only).
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
    "V6_CLIMAT_BONUS",
    "V6_PRIME_LR_CONDITIONING_CHANNELS",
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
