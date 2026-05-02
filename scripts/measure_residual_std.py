"""
Measure the empirical sigma_data of the log1p precipitation residual.

The EDM preconditioning (Karras 2022 Eq. 7) requires sigma_data to be
calibrated to the empirical std of the *target* the diffusion model is
asked to predict. In ST-CDGM the target is the log1p residual

    r = log1p(HR + delta) - log1p(baseline + delta)

where ``baseline`` is the bicubic upsampling of the LR field (or the
hr_smoothing variant, depending on config) and ``delta`` is the
``precipitation_delta`` regulariser from the data pipeline.

Image-domain DDPM defaults assume sigma_data ~= 0.5; for log1p
precipitation residuals over a coarse-to-fine downscaling task we
typically see sigma_data ~= 0.05 to 0.2. Mismatching this parameter by
~5x is the most likely root cause of the sigma_r ~= 9 ensemble
over-dispersion observed on the Sprint 4 checkpoint (cf. Gemini Deep
Research report, Axis 2 / Axis 7 diagnosis #2).

Usage
-----
Run from the repo root::

    .venv/Scripts/python scripts/measure_residual_std.py

Outputs the (mean, std, min, max, p1, p99) of the residual. Copy the
``std`` into ``config/training_config.yaml`` under
``diffusion.edm.sigma_data``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

# ensure src is importable when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from st_cdgm.data.pipeline import NetCDFDataPipeline  # noqa: E402


def measure_residual_std(
    *,
    lr_path: str,
    hr_path: str,
    static_path: str | None,
    means_path: str | None,
    stds_path: str | None,
    seq_len: int,
    baseline_strategy: str,
    baseline_factor: int,
    normalize: bool,
    target_transform: str,
    nan_fill_strategy: str,
    precipitation_delta: float,
    lr_variables: list[str],
    hr_variables: list[str],
    static_variables: list[str],
    max_samples: int | None = None,
) -> dict:
    """Build the pipeline, iterate the dataset, accumulate residual stats."""
    pipeline = NetCDFDataPipeline(
        lr_path=lr_path,
        hr_path=hr_path,
        static_path=static_path,
        seq_len=seq_len,
        baseline_strategy=baseline_strategy,
        baseline_factor=baseline_factor,
        normalize=normalize,
        nan_fill_strategy=nan_fill_strategy,
        precipitation_delta=precipitation_delta,
        target_transform=target_transform,
        lr_variables=lr_variables,
        hr_variables=hr_variables,
        static_variables=static_variables,
        means_path=means_path if means_path and Path(means_path).exists() else None,
        stds_path=stds_path if stds_path and Path(stds_path).exists() else None,
    )

    dataset = pipeline.build_sequence_dataset(seq_len=seq_len, as_torch=False)

    # Accumulate Welford-style for memory safety on large datasets.
    n = 0
    mean = 0.0
    M2 = 0.0
    p_buffer: list[float] = []  # for percentiles, sample-level
    minimum = np.inf
    maximum = -np.inf

    for idx, sample in enumerate(dataset):
        if max_samples is not None and idx >= max_samples:
            break
        # The pipeline returns ``residual`` which is exactly
        # log1p(HR + delta) - log1p(baseline + delta) when target_transform=log1p.
        residual = np.asarray(sample["residual"][-1])  # last step of seq, [C, H, W]
        flat = residual[np.isfinite(residual)].astype(np.float64).ravel()
        if flat.size == 0:
            continue
        # Welford
        for x in flat[::1024]:  # subsample the running mean update for speed
            n += 1
            delta = x - mean
            mean += delta / n
            M2 += delta * (x - mean)
        # Track extrema and a rolling sample for percentiles
        minimum = float(min(minimum, flat.min()))
        maximum = float(max(maximum, flat.max()))
        if len(p_buffer) < 2_000_000:
            # Cap the percentile buffer at ~16 MB
            p_buffer.extend(flat[:: max(1, flat.size // 1000)].tolist())
        if (idx + 1) % 50 == 0:
            print(
                f"  processed {idx + 1} samples, running mean={mean:+.4f}, "
                f"running var={(M2 / max(n - 1, 1)):.4f}",
                flush=True,
            )

    if n < 2:
        raise RuntimeError("Empty dataset — check pipeline paths.")

    var = M2 / (n - 1)
    std = float(np.sqrt(var))
    p_arr = np.asarray(p_buffer)

    return {
        "n_residual_pixels": int(n),
        "mean": float(mean),
        "std": std,
        "min": minimum,
        "max": maximum,
        "p1": float(np.percentile(p_arr, 1)),
        "p99": float(np.percentile(p_arr, 99)),
        "p50": float(np.percentile(p_arr, 50)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=str, default="config/training_config.yaml",
        help="Path to YAML config (default: config/training_config.yaml).",
    )
    parser.add_argument(
        "--max-samples", type=int, default=200,
        help="Cap on number of sequences to scan (default 200 ≈ 10 min on CPU).",
    )
    parser.add_argument(
        "--lr-path", type=str, default=None,
        help="Override CONFIG.data.lr_path.",
    )
    parser.add_argument(
        "--hr-path", type=str, default=None,
        help="Override CONFIG.data.hr_path.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    lr_path = args.lr_path or cfg.data.lr_path
    hr_path = args.hr_path or cfg.data.hr_path

    if not Path(lr_path).exists() or not Path(hr_path).exists():
        print(f"[ERROR] LR or HR path missing: lr={lr_path} hr={hr_path}", file=sys.stderr)
        return 1

    means_path = (
        cfg.data.get("means_path")
        or "data/raw/normalization_coefs/mean_1974_2011.nc"
    )
    stds_path = (
        cfg.data.get("stds_path")
        or "data/raw/normalization_coefs/std_1974_2011.nc"
    )

    print(f"Reading LR: {lr_path}")
    print(f"Reading HR: {hr_path}")
    print(f"max_samples = {args.max_samples}")

    stats = measure_residual_std(
        lr_path=lr_path,
        hr_path=hr_path,
        static_path=cfg.data.get("static_path"),
        means_path=means_path,
        stds_path=stds_path,
        seq_len=int(cfg.data.seq_len),
        baseline_strategy=str(cfg.data.baseline_strategy),
        baseline_factor=int(cfg.data.get("baseline_factor", 4)),
        normalize=bool(cfg.data.normalize),
        target_transform=str(cfg.data.get("target_transform", "log1p")),
        nan_fill_strategy=str(cfg.data.get("nan_fill_strategy", "mean")),
        precipitation_delta=float(cfg.data.get("precipitation_delta", 0.01)),
        lr_variables=list(cfg.data.lr_variables),
        hr_variables=list(cfg.data.hr_variables),
        static_variables=list(cfg.data.static_variables),
        max_samples=args.max_samples,
    )

    print("\n" + "=" * 60)
    print("EMPIRICAL RESIDUAL STATISTICS (log1p domain)")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:>20s} = {v}")
    print("=" * 60)
    print(
        "\nRECOMMENDED CONFIG UPDATE:\n"
        f"  config/training_config.yaml -> diffusion.edm.sigma_data: {stats['std']:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
