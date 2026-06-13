# >>> BS43 — EXPORT EVAL SAMPLES (causal vs non-causal qualitative comparison)
# Dumps a small subset of the FINAL_VALIDATION tensors to <ckpt>/eval_samples.npz
# so the comparison notebook can plot maps / spectra / PDFs WITHOUT rebuilding or
# re-sampling either model. Runs after FINAL_VALIDATION + BS42 (reuses their globals).
import numpy as _np43
from pathlib import Path as _P43

try:
    _ck43 = _P43(str(globals().get("_ckpt_dir", globals().get("CKPT_SAVE_DIR", "results"))))
    _ck43.mkdir(parents=True, exist_ok=True)
    _N43 = min(8, int(_targets.shape[0]))

    def _np_sub(t):
        return t[:_N43].detach().float().cpu().numpy() if hasattr(t, "detach") else _np43.asarray(t)[:_N43]

    _runv = str(CONFIG.get("two_stage", {}).get("run_variant", "causal"))
    _payload43 = dict(
        target=_np_sub(_targets),
        pred_full=_np_sub(_pred_full),
        pred_std=_np_sub(_pred_std),
        valid_mask=_np_sub(_valid).astype("float32"),
        run_variant=_np_sub(_targets)[:0].astype("float32"),  # placeholder, replaced below
    )
    _payload43.pop("run_variant")
    _payload43["run_variant"] = _np43.array(_runv)  # fixed-length unicode, no pickle needed

    _mu_src43 = globals().get("_mu_concat", None)
    if _mu_src43 is not None:
        _payload43["mu_HR"] = _np_sub(_mu_src43)

    _np43.savez_compressed(_ck43 / "eval_samples.npz", **_payload43)
    print(f"💾 eval_samples.npz saved: {_ck43 / 'eval_samples.npz'}  "
          f"(N={_N43}, run_variant={_runv})")
except Exception as _e43:
    print(f"[warn] eval_samples export failed: {_e43}")
