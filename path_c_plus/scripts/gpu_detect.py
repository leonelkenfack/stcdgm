"""GPU profile detection for adaptive config (T4 vs A100 40/80 vs V100/H100).

Used at notebook bootstrap to set batch_size, amp_dtype, num_workers
automatically based on detected hardware. Keeps code T4-backward-compatible.

Usage:
    from path_c_plus.scripts.gpu_detect import detect_gpu_profile, apply_profile_to_config

    profile = detect_gpu_profile()
    CONFIG = apply_profile_to_config(CONFIG, profile)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


PROFILES: Dict[str, Dict[str, Any]] = {
    "cpu": {
        "name": "CPU only",
        "batch_size": 4,
        "amp_dtype": "float32",
        "use_amp": False,
        "num_workers": 2,
        "pin_memory": False,
        "use_torch_compile": False,
    },
    "t4": {
        "name": "Tesla T4 (16GB, sm_75)",
        "batch_size": 8,
        "amp_dtype": "float16",
        "use_amp": True,
        "num_workers": 2,
        "pin_memory": True,
        "use_torch_compile": False,
    },
    "v100": {
        "name": "V100 (16GB or 32GB)",
        "batch_size": 16,
        "amp_dtype": "float16",
        "use_amp": True,
        "num_workers": 4,
        "pin_memory": True,
        "use_torch_compile": True,
    },
    "a100_40": {
        "name": "A100 40GB (sm_80)",
        "batch_size": 32,
        "amp_dtype": "bfloat16",
        "use_amp": True,
        "num_workers": 8,
        "pin_memory": True,
        "use_torch_compile": True,
    },
    "a100_80": {
        "name": "A100 80GB (sm_80)",
        "batch_size": 64,
        "amp_dtype": "bfloat16",
        "use_amp": True,
        "num_workers": 8,
        "pin_memory": True,
        "use_torch_compile": True,
    },
    "h100": {
        "name": "H100 (sm_90)",
        "batch_size": 64,
        "amp_dtype": "bfloat16",
        "use_amp": True,
        "num_workers": 12,
        "pin_memory": True,
        "use_torch_compile": True,
    },
}


def detect_gpu_profile() -> Dict[str, Any]:
    """Detect GPU and return optimal training profile.

    Returns a dict with keys:
        - profile_id: short string identifier (e.g. "a100_80")
        - name: human-readable GPU name
        - vram_gb: total VRAM in GB
        - compute_capability: (major, minor) tuple
        - batch_size: recommended batch size
        - amp_dtype: "float16", "bfloat16", or "float32"
        - use_amp: whether to enable mixed precision
        - num_workers: DataLoader num_workers
        - pin_memory: whether to pin memory in DataLoader
        - use_torch_compile: whether torch.compile is recommended
    """
    if not _HAS_TORCH:
        return {**PROFILES["cpu"], "profile_id": "cpu", "vram_gb": 0, "compute_capability": (0, 0)}

    if not torch.cuda.is_available():
        return {**PROFILES["cpu"], "profile_id": "cpu", "vram_gb": 0, "compute_capability": (0, 0)}

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    sm = (props.major, props.minor)
    name = props.name

    # Decision tree based on compute capability + VRAM
    if sm[0] >= 9:
        profile_id = "h100"
    elif sm[0] >= 8:
        # Ampere or newer (A100, A40, A6000)
        if vram_gb >= 70:
            profile_id = "a100_80"
        elif vram_gb >= 35:
            profile_id = "a100_40"
        else:
            # A40, A6000 etc — use a100_40 profile as proxy
            profile_id = "a100_40"
    elif sm == (7, 0):
        # V100 (Volta)
        profile_id = "v100"
    elif sm == (7, 5):
        # Turing (T4, RTX 20-series)
        profile_id = "t4"
    elif sm[0] >= 7:
        # Other Volta/Turing fallback
        profile_id = "t4"
    else:
        # Pascal or older
        profile_id = "t4"

    profile = dict(PROFILES[profile_id])
    profile["profile_id"] = profile_id
    profile["vram_gb"] = round(vram_gb, 1)
    profile["compute_capability"] = sm
    profile["detected_name"] = name

    return profile


def apply_profile_to_config(config: Any, profile: Dict[str, Any]) -> Any:
    """Apply detected profile to OmegaConf-like config in place.

    Sets:
        - config.training.batch_size
        - config.training.amp_dtype
        - config.training.use_amp
        - config.training.num_workers
        - config.training.pin_memory
        - config.training.compile.enabled

    Preserves any explicit user overrides (only sets if not present or matches default).
    """
    if hasattr(config, "training"):
        t = config.training
    else:
        return config

    # Always apply (overrides any default but not user-specified)
    t.batch_size = profile["batch_size"]
    t.amp_dtype = profile["amp_dtype"]
    t.use_amp = profile["use_amp"]
    t.num_workers = profile["num_workers"]
    t.pin_memory = profile["pin_memory"]

    # torch.compile section (create if missing)
    if not hasattr(t, "compile") or t.compile is None:
        # OmegaConf — assume dict-like
        try:
            from omegaconf import OmegaConf
            t.compile = OmegaConf.create({
                "enabled": profile["use_torch_compile"],
                "diffusion_mode": "default",
                "encoder_mode": "default",
                "rcn_mode": "default",
            })
        except ImportError:
            pass
    else:
        t.compile.enabled = profile["use_torch_compile"]

    return config


def print_profile_banner(profile: Dict[str, Any]) -> None:
    """Print a visible startup banner with the detected profile."""
    print("=" * 70)
    print(f"GPU PROFILE DETECTED : {profile['profile_id'].upper()}")
    print("=" * 70)
    print(f"  Device              : {profile.get('detected_name', profile['name'])}")
    print(f"  Compute capability  : sm_{profile['compute_capability'][0]}{profile['compute_capability'][1]}")
    print(f"  VRAM                : {profile['vram_gb']} GB")
    print(f"  Recommended config  :")
    print(f"    batch_size        : {profile['batch_size']}")
    print(f"    amp_dtype         : {profile['amp_dtype']}")
    print(f"    use_amp           : {profile['use_amp']}")
    print(f"    num_workers       : {profile['num_workers']}")
    print(f"    pin_memory        : {profile['pin_memory']}")
    print(f"    use_torch_compile : {profile['use_torch_compile']}")
    print("=" * 70)


if __name__ == "__main__":
    profile = detect_gpu_profile()
    print_profile_banner(profile)
