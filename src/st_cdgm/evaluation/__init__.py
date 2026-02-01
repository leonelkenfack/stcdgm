"""
Modules d'évaluation et XAI pour ST-CDGM.
"""

from .evaluation_xai import (
    autoregressive_inference,
    evaluate_metrics,
    compute_crps,
    compute_fss,
    compute_f1_extremes,
    compute_spectrum_distance,
    compute_temporal_variance_metrics,
    compute_wasserstein_distance,
    compute_energy_score,
    compute_structural_hamming_distance,
    plot_dag_heatmap,
    export_dag_to_csv,
    export_dag_to_json,
    MetricReport,
    InferenceResult,
)

__all__ = [
    "autoregressive_inference",
    "evaluate_metrics",
    "compute_crps",
    "compute_fss",
    "compute_f1_extremes",
    "compute_spectrum_distance",
    "compute_temporal_variance_metrics",
    "compute_wasserstein_distance",
    "compute_energy_score",
    "compute_structural_hamming_distance",
    "plot_dag_heatmap",
    "export_dag_to_csv",
    "export_dag_to_json",
    "MetricReport",
    "InferenceResult",
]

