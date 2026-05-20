"""Patch st_cdgm_training_evaluation.ipynb for V5 default runbook (BS40).

Applies the V5 "Pearson 0.90+" preparation patches identified in
``C:/Users/reall/.claude/plans/faire-un-extraplan-complet-transient-lollipop.md``:

* default override pointer        -> training_config_v5_pearson_090.yaml
* EDMConfig manual builder        -> EDMConfig.from_yaml_dict(CONFIG.diffusion.edm)
* FINAL_VALIDATION K_SAMPLES      -> 64 (ensemble averaging boost)
* train_epoch_stage2_cached call  -> + conditioning_dropout_prob + ema_warmup_steps

Run from project root::

    python scripts/_patch_notebook_v5_runbook.py

The patch is JSON-aware (operates on cell source line lists) and idempotent.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "st_cdgm_training_evaluation.ipynb"


# ---------------------------------------------------------------------------
# Patch 1 — _default_override : V4 -> V5. Single-line, simple replace.
# ---------------------------------------------------------------------------
DEFAULT_OVERRIDE_OLD = "_default_override = \"training_config_v4_corrdiff_large.yaml\""
DEFAULT_OVERRIDE_NEW = "_default_override = \"training_config_v5_pearson_090.yaml\""


# ---------------------------------------------------------------------------
# Patch 2 — EDMConfig manual builder -> EDMConfig.from_yaml_dict.
# Multi-line block — replaced as a contiguous slice in the source list.
# ---------------------------------------------------------------------------
EDM_BUILDER_OLD_LINES = [
    "    from st_cdgm.models.edm_preconditioner import EDMConfig\n",
    "    _edm_cfg_raw = CONFIG.diffusion.get(\"edm\", {})\n",
    "    # >>> BS34_TAIL_LOSS — read tail_weight block (None if absent).\n",
    "    from st_cdgm.models.edm_preconditioner import TailWeightConfig as _TWConfig\n",
    "    _tw_raw = _edm_cfg_raw.get(\"tail_weight\", None)\n",
    "    if _tw_raw is not None and bool(_tw_raw.get(\"enabled\", False)):\n",
    "        _tail_w = _TWConfig(\n",
    "            enabled=True,\n",
    "            tau95_mmday=float(_tw_raw.get(\"tau95_mmday\", 15.0)),\n",
    "            tau99_mmday=float(_tw_raw.get(\"tau99_mmday\", 35.0)),\n",
    "            weight_p95=float(_tw_raw.get(\"weight_p95\", 5.0)),\n",
    "            weight_p99=float(_tw_raw.get(\"weight_p99\", 10.0)),\n",
    "        )\n",
    "        print(f\"   📈 BS34 tail-weight ON : τ95={_tail_w.tau95_mmday}mm \"\n",
    "              f\"(w={_tail_w.weight_p95}), τ99={_tail_w.tau99_mmday}mm \"\n",
    "              f\"(w={_tail_w.weight_p99})\")\n",
    "    else:\n",
    "        _tail_w = None\n",
    "    _edm_config = EDMConfig(\n",
    "        sigma_data=float(_edm_cfg_raw.get(\"sigma_data\", 0.1)),\n",
    "        sigma_min=float(_edm_cfg_raw.get(\"sigma_min\", 0.002)),\n",
    "        sigma_max=float(_edm_cfg_raw.get(\"sigma_max\", 80.0)),\n",
    "        rho=float(_edm_cfg_raw.get(\"rho\", 7.0)),\n",
    "        P_mean=float(_edm_cfg_raw.get(\"P_mean\", -1.2)),\n",
    "        P_std=float(_edm_cfg_raw.get(\"P_std\", 1.2)),\n",
    "        S_churn=float(_edm_cfg_raw.get(\"S_churn\", 0.0)),\n",
    "        S_tmin=float(_edm_cfg_raw.get(\"S_tmin\", 0.0)),\n",
    "        S_tmax=float(_edm_cfg_raw.get(\"S_tmax\", float(\"inf\"))),\n",
    "        S_noise=float(_edm_cfg_raw.get(\"S_noise\", 1.0)),\n",
    "        tail_weight=_tail_w,\n",
    "    )\n",
]
EDM_BUILDER_NEW_LINES = [
    "    # >>> V5 (BS40) — single-entry builder reads tail_weight + spectral_loss +\n",
    "    # wasserstein_reg from the YAML and produces a fully populated EDMConfig.\n",
    "    from st_cdgm.models.edm_preconditioner import EDMConfig\n",
    "    _edm_cfg_raw = CONFIG.diffusion.get(\"edm\", {})\n",
    "    _edm_config = EDMConfig.from_yaml_dict(_edm_cfg_raw)\n",
    "    if _edm_config.tail_weight is not None and _edm_config.tail_weight.enabled:\n",
    "        _tw = _edm_config.tail_weight\n",
    "        print(f\"   📈 tail-weight ON : τ95={_tw.tau95_mmday}mm \"\n",
    "              f\"(w={_tw.weight_p95}), τ99={_tw.tau99_mmday}mm \"\n",
    "              f\"(w={_tw.weight_p99})\")\n",
    "    if _edm_config.spectral_loss is not None and _edm_config.spectral_loss.enabled:\n",
    "        _sl = _edm_config.spectral_loss\n",
    "        print(f\"   🌀 V5 FACL spectral ON : α={_sl.alpha_amplitude}, \"\n",
    "              f\"β={_sl.beta_correlation}, λ={_sl.lambda_weight}\")\n",
    "    if _edm_config.wasserstein_reg is not None and _edm_config.wasserstein_reg.enabled:\n",
    "        _wr = _edm_config.wasserstein_reg\n",
    "        print(f\"   📏 V5 Sliced-W1 ON : n_slices={_wr.n_slices}, \"\n",
    "              f\"λ={_wr.lambda_weight}\")\n",
]
EDM_BUILDER_MARKER_NEW = "    _edm_config = EDMConfig.from_yaml_dict(_edm_cfg_raw)\n"


# ---------------------------------------------------------------------------
# Patch 3 — K_SAMPLES bump 16 -> 64 (already applied via string substitute).
# We do it on the source list too for idempotence.
# ---------------------------------------------------------------------------
K_SAMPLES_OLD_LINE = (
    "K_SAMPLES = 16             # ensemble pour CRPS/spread (was 4 → 16)\n"
)
K_SAMPLES_NEW_LINE = (
    "K_SAMPLES = int(globals().get(\"K_SAMPLES_OVERRIDE\", 64))  "
    "# V5 ensemble averaging boost (16 -> 64); override via globals()\n"
)


# ---------------------------------------------------------------------------
# Patch 4 — train_epoch_stage2_cached call: inject V5 kwargs after ema_decay.
# ---------------------------------------------------------------------------
S2_CALL_OLD_LINES = [
    "            ema_model=_ema_diffusion,\n",
    "            ema_decay=_ema_decay,\n",
    "        )\n",
]
S2_CALL_NEW_LINES = [
    "            ema_model=_ema_diffusion,\n",
    "            ema_decay=_ema_decay,\n",
    "            # >>> V5 (BS40) — Track B1 EMA warmup + B2 conditioning dropout +\n",
    "            # Track C logging (FACL/SW components). Safe defaults (0 / 0.0)\n",
    "            # keep V4 behaviour when the YAML omits these keys.\n",
    "            ema_warmup_steps=int(\n",
    "                (CONFIG.get('two_stage', {}).get('stage2', {}) or {})\n",
    "                .get('ema', {}).get('warmup_steps', 0)\n",
    "            ),\n",
    "            conditioning_dropout_prob=float(\n",
    "                CONFIG.diffusion.get('conditioning_dropout_prob', 0.0)\n",
    "            ),\n",
    "            log_loss_components=True,\n",
    "        )\n",
]
S2_CALL_MARKER_NEW = "            log_loss_components=True,\n"


# ---------------------------------------------------------------------------
# Patch 5 (BS41) — FINAL_VALIDATION : add bypass-EMA flag when EMA is suspected
# stale (V5 EMA bug F1). Default behaviour unchanged ; user opts in by setting
# ``BS41_FORCE_LIVE_INFERENCE = True`` in globals() before running the cell.
# ---------------------------------------------------------------------------
EMA_LOAD_OLD_LINES = [
    "_ema_sd_eval = _ckpt.get(\"diffusion_ema_state_dict\")\n",
    "if _ema_sd_eval is not None:\n",
    "    print(\"  🌗 BS37 EMA detected in checkpoint — loading EMA weights for FINAL_VALIDATION\")\n",
    "    _persist_load_state_dict(diffusion, _ema_sd_eval)\n",
    "else:\n",
    "    _persist_load_state_dict(diffusion, _ckpt.get(\"diffusion_state_dict\"))\n",
]
EMA_LOAD_NEW_LINES = [
    "_ema_sd_eval = _ckpt.get(\"diffusion_ema_state_dict\")\n",
    "# >>> BS41 FIX F4 — bypass-EMA escape hatch. Set this global to True\n",
    "# *before* running the FINAL_VALIDATION cell to load the LIVE diffusion\n",
    "# weights instead of the EMA shadow (useful when the EMA is suspected to\n",
    "# be stale due to the V5 ema_steps reset bug).\n",
    "_bs41_force_live = bool(globals().get(\"BS41_FORCE_LIVE_INFERENCE\", False))\n",
    "if _ema_sd_eval is not None and not _bs41_force_live:\n",
    "    print(\"  🌗 BS37 EMA detected in checkpoint — loading EMA weights for FINAL_VALIDATION\")\n",
    "    _persist_load_state_dict(diffusion, _ema_sd_eval)\n",
    "elif _bs41_force_live:\n",
    "    print(\"  ⚠️  BS41 FORCE_LIVE_INFERENCE=True — bypass EMA, loading live diffusion weights\")\n",
    "    _persist_load_state_dict(diffusion, _ckpt.get(\"diffusion_state_dict\"))\n",
    "else:\n",
    "    _persist_load_state_dict(diffusion, _ckpt.get(\"diffusion_state_dict\"))\n",
]
EMA_LOAD_MARKER_NEW = "_bs41_force_live = bool(globals().get(\"BS41_FORCE_LIVE_INFERENCE\", False))\n"


def _find_line_in_source(source: List[str], line: str) -> int:
    for idx, src_line in enumerate(source):
        if src_line == line:
            return idx
    return -1


def _find_block_in_source(source: List[str], block: List[str]) -> int:
    """Return start index of ``block`` in ``source`` (sliding match)."""
    n = len(block)
    for idx in range(len(source) - n + 1):
        if source[idx:idx + n] == block:
            return idx
    return -1


def _replace_single_line(source: List[str], old: str, new: str) -> bool:
    idx = _find_line_in_source(source, old)
    if idx < 0:
        return False
    source[idx] = new
    return True


def _replace_block(
    source: List[str],
    old_block: List[str],
    new_block: List[str],
    marker_new: str,
) -> Tuple[bool, str]:
    """Return (changed, status). Idempotent: if marker_new already present, skip."""
    if any(marker_new == ln for ln in source):
        return False, "already applied"
    start = _find_block_in_source(source, old_block)
    if start < 0:
        return False, "marker not found - manual review"
    end = start + len(old_block)
    source[start:end] = new_block
    return True, "applied"


def _replace_line_in_string(
    source_str: str, old: str, new: str
) -> Tuple[str, bool, str]:
    """For single-line replacement where the source is a single big string."""
    if new in source_str:
        return source_str, False, "already applied"
    if old not in source_str:
        return source_str, False, "marker not found - manual review"
    return source_str.replace(old, new, 1), True, "applied"


def main() -> None:
    if not NB_PATH.exists():
        raise FileNotFoundError(f"Notebook not found: {NB_PATH}")
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    applied: List[str] = []
    skipped: List[str] = []

    # Iterate cells once; each patch tracks whether it has been applied.
    p1_done = False
    p2_done = False
    p3_done = False
    p4_done = False
    p5_done = False
    p1_status = "marker not found - manual review"
    p2_status = "marker not found - manual review"
    p3_status = "marker not found - manual review"
    p4_status = "marker not found - manual review"
    p5_status = "marker not found - manual review"

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source")
        if not isinstance(source, list):
            continue

        # Patch 1 — single line within ONE cell source line.
        if not p1_done:
            for i, ln in enumerate(source):
                if DEFAULT_OVERRIDE_NEW in ln:
                    p1_done = True
                    p1_status = "already applied"
                    break
                if DEFAULT_OVERRIDE_OLD in ln:
                    source[i] = ln.replace(
                        DEFAULT_OVERRIDE_OLD, DEFAULT_OVERRIDE_NEW
                    )
                    p1_done = True
                    p1_status = "applied"
                    break

        # Patch 2 — multi-line EDMConfig builder block.
        if not p2_done:
            changed, status = _replace_block(
                source, EDM_BUILDER_OLD_LINES, EDM_BUILDER_NEW_LINES,
                EDM_BUILDER_MARKER_NEW,
            )
            if status == "already applied":
                p2_done = True
                p2_status = status
            elif changed:
                p2_done = True
                p2_status = status

        # Patch 3 — K_SAMPLES single line.
        if not p3_done:
            for i, ln in enumerate(source):
                if K_SAMPLES_NEW_LINE.strip() in ln.strip():
                    p3_done = True
                    p3_status = "already applied"
                    break
                if ln == K_SAMPLES_OLD_LINE:
                    source[i] = K_SAMPLES_NEW_LINE
                    p3_done = True
                    p3_status = "applied"
                    break

        # Patch 4 — Stage 2 call block.
        if not p4_done:
            changed, status = _replace_block(
                source, S2_CALL_OLD_LINES, S2_CALL_NEW_LINES,
                S2_CALL_MARKER_NEW,
            )
            if status == "already applied":
                p4_done = True
                p4_status = status
            elif changed:
                p4_done = True
                p4_status = status

        # Patch 5 (BS41) — FINAL_VALIDATION EMA bypass flag.
        if not p5_done:
            changed, status = _replace_block(
                source, EMA_LOAD_OLD_LINES, EMA_LOAD_NEW_LINES,
                EMA_LOAD_MARKER_NEW,
            )
            if status == "already applied":
                p5_done = True
                p5_status = status
            elif changed:
                p5_done = True
                p5_status = status

    for label, done, status in [
        ("default override -> V5", p1_done, p1_status),
        ("EDMConfig builder -> from_yaml", p2_done, p2_status),
        ("K_SAMPLES 16 -> 64", p3_done, p3_status),
        ("Stage 2 call -> V5 kwargs", p4_done, p4_status),
        ("BS41 FINAL_VAL EMA bypass", p5_done, p5_status),
    ]:
        if status == "applied":
            applied.append(label)
        else:
            skipped.append(f"{label} ({status})")

    # Validate by re-serializing then re-parsing.
    text = json.dumps(nb, ensure_ascii=False, indent=1)
    json.loads(text)
    NB_PATH.write_text(text, encoding="utf-8")

    print(f"[v5 patch] {NB_PATH}")
    if applied:
        print("  [+] applied:")
        for label in applied:
            print(f"     - {label}")
    if skipped:
        print("  [-] skipped:")
        for label in skipped:
            print(f"     - {label}")
    if not applied:
        print("  (no changes - patch is idempotent)")


if __name__ == "__main__":
    main()
