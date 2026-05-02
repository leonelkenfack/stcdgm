"""
Adapte les notebooks training + validation pour le mode EDM (Karras 2022).

Fait trois choses :

1. Training notebook (st_cdgm_training_evaluation.ipynb):
   * Ajoute ``edm_config`` au constructeur ``CausalDiffusionDecoder`` quand
     ``scheduler_type='edm_karras'`` (lit ``CONFIG.diffusion.edm.*``).
   * Insère une cellule pre-flight qui mesure ``sigma_data`` empirique sur
     un sous-échantillon avant le grand training.
   * Met à jour le COLAB_BOOTSTRAP pour basculer sur la branche
     ``edm-rewrite`` au démarrage.

2. Validation notebook (st_cdgm_validation_inference.ipynb):
   * Ajoute le même branchement EDM dans la cellule diffusion rebuild.
   * Insère un test SHD (Structural Hamming Distance) sur le DAG appris
     vs prior config.

Toutes les éditions sont sentinellées → idempotent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_NB = ROOT / "st_cdgm_training_evaluation.ipynb"
VAL_NB = ROOT / "st_cdgm_validation_inference.ipynb"


# ---------------------------------------------------------------------
# Patch building blocks
# ---------------------------------------------------------------------

EDM_CONFIG_BLOCK = '''# >>> EDM_BUILD_CONFIG
# Bascule EDM (Karras 2022) si scheduler_type=='edm_karras'.
# Lit la sous-section CONFIG.diffusion.edm pour instancier EDMConfig.
_scheduler_type = CONFIG.diffusion.get("scheduler_type", "ddpm")
_edm_config = None
if _scheduler_type == "edm_karras":
    from st_cdgm.models.edm_preconditioner import EDMConfig
    _edm_cfg_raw = CONFIG.diffusion.get("edm", {})
    _edm_config = EDMConfig(
        sigma_data=float(_edm_cfg_raw.get("sigma_data", 0.1)),
        sigma_min=float(_edm_cfg_raw.get("sigma_min", 0.002)),
        sigma_max=float(_edm_cfg_raw.get("sigma_max", 80.0)),
        rho=float(_edm_cfg_raw.get("rho", 7.0)),
        P_mean=float(_edm_cfg_raw.get("P_mean", -1.2)),
        P_std=float(_edm_cfg_raw.get("P_std", 1.2)),
        S_churn=float(_edm_cfg_raw.get("S_churn", 0.0)),
        S_tmin=float(_edm_cfg_raw.get("S_tmin", 0.0)),
        S_tmax=float(_edm_cfg_raw.get("S_tmax", float("inf"))),
        S_noise=float(_edm_cfg_raw.get("S_noise", 1.0)),
    )
    print(f"🌊 EDM config : sigma_data={_edm_config.sigma_data}, "
          f"sigma_range=[{_edm_config.sigma_min}, {_edm_config.sigma_max}], "
          f"rho={_edm_config.rho}")
'''


PREFLIGHT_SIGMA_DATA = '''# >>> EDM_PREFLIGHT_SIGMA_DATA
# Pre-flight : mesure empirique de sigma_data sur 50 séquences.
# CRITIQUE pour la calibration EDM (Karras §5) — un sigma_data mal
# calibré cause sigma_r >> 1 (ensemble overdispersé).
# Si sigma_data mesuré != CONFIG.diffusion.edm.sigma_data, override
# in-place et log un avertissement.
import numpy as np

if CONFIG.diffusion.get("scheduler_type") == "edm_karras":
    print("🔬 Pre-flight EDM : mesure sigma_data empirique...")

    # Sous-échantillonnage pour rester rapide (<5 min sur GPU/CPU)
    _PREFLIGHT_N_SEQUENCES = 50
    _residuals = []
    _ds_iter = iter(dataset)
    for _ in range(_PREFLIGHT_N_SEQUENCES):
        try:
            _sample = next(_ds_iter)
        except StopIteration:
            break
        _r = np.asarray(_sample["residual"][-1])
        _r = _r[np.isfinite(_r)]
        if _r.size > 0:
            _residuals.append(_r.ravel())

    if not _residuals:
        print("  ⚠️  Aucun résidu valide trouvé — skip calibration.")
    else:
        _all = np.concatenate(_residuals).astype(np.float64)
        _measured_sigma = float(_all.std())
        _config_sigma = float(CONFIG.diffusion.edm.sigma_data)

        print(f"  mesuré : sigma_data = {_measured_sigma:.4f}")
        print(f"  config : sigma_data = {_config_sigma:.4f}")
        print(f"  ratio  : {_measured_sigma / max(_config_sigma, 1e-8):.2f}x")

        # Si écart > 50%, override (plus prudent que d'utiliser une valeur fausse).
        if abs(_measured_sigma - _config_sigma) / max(_config_sigma, 1e-8) > 0.5:
            print(f"  🔧 Écart >50% détecté — override CONFIG.diffusion.edm.sigma_data "
                  f"-> {_measured_sigma:.4f}")
            CONFIG.diffusion.edm.sigma_data = _measured_sigma
            # Si _edm_config existe déjà (cellule build_diffusion exécutée),
            # le réinstancier avec la nouvelle valeur.
            if "_edm_config" in dir() and _edm_config is not None:
                from st_cdgm.models.edm_preconditioner import EDMConfig as _EDMConfig
                _edm_config = _EDMConfig(
                    sigma_data=_measured_sigma,
                    sigma_min=_edm_config.sigma_min,
                    sigma_max=_edm_config.sigma_max,
                    rho=_edm_config.rho,
                    P_mean=_edm_config.P_mean,
                    P_std=_edm_config.P_std,
                    S_churn=_edm_config.S_churn,
                    S_tmin=_edm_config.S_tmin,
                    S_tmax=_edm_config.S_tmax,
                    S_noise=_edm_config.S_noise,
                )
                if "diffusion" in dir() and hasattr(diffusion, "edm_config"):
                    diffusion.edm_config = _edm_config
                    print("  ✓ diffusion.edm_config mis à jour in-place.")
        else:
            print("  ✓ écart < 50%, calibration acceptable.")
'''


SHD_TEST_BLOCK = '''# >>> EDM_SHD_TEST
# Structural Hamming Distance (Tsamardinos 2006) entre le DAG appris
# et le prior théorique. Mesure quantitative XAI réclamée par le papier
# §sec:mat:eval ("Structural recovery").
#
# SHD(A_learned, A_prior) = nb_edges_added + nb_edges_removed + nb_edges_reversed
# où l'on seuille les matrices à un cutoff (par défaut 1e-2).
import numpy as np
import torch

def structural_hamming_distance(A_learned, A_prior, threshold=1e-2):
    """SHD non-pondéré entre 2 matrices q×q.

    Convention : entry (i,j) > threshold = "il y a une arête de i vers j".
    Retourne (added, removed, reversed, total) en nombre d'arêtes.
    """
    if isinstance(A_learned, torch.Tensor):
        A_learned = A_learned.detach().cpu().numpy()
    if isinstance(A_prior, torch.Tensor):
        A_prior = A_prior.detach().cpu().numpy()
    A_learned = np.asarray(A_learned)
    A_prior = np.asarray(A_prior)
    # Binariser
    L = (np.abs(A_learned) > threshold).astype(int)
    P = (np.abs(A_prior) > threshold).astype(int)
    # No diagonal (no self-loops by construction)
    np.fill_diagonal(L, 0)
    np.fill_diagonal(P, 0)
    added = int(((L == 1) & (P == 0)).sum())
    removed = int(((L == 0) & (P == 1)).sum())
    # Reversed: (i,j) in L but (j,i) in P (and not (i,j) in P)
    reversed_edges = 0
    q = L.shape[0]
    for i in range(q):
        for j in range(i + 1, q):
            l_ij, l_ji = L[i, j], L[j, i]
            p_ij, p_ji = P[i, j], P[j, i]
            # If exactly one direction in L and the OTHER direction in P
            if (l_ij and not l_ji and not p_ij and p_ji) or (
                l_ji and not l_ij and not p_ji and p_ij
            ):
                reversed_edges += 1
    total = added + removed + reversed_edges
    return {
        "added": added,
        "removed": removed,
        "reversed": reversed_edges,
        "total": total,
        "n_edges_learned": int(L.sum()),
        "n_edges_prior": int(P.sum()),
        "q": int(q),
    }


# Extraire A_dag du modèle reconstruit (rcn_cell)
_rcn_base = rcn_cell.module if hasattr(rcn_cell, "module") else rcn_cell
_rcn_base = getattr(_rcn_base, "_orig_mod", _rcn_base)
A_learned = _rcn_base.A_dag.detach().cpu().numpy()
# Mask the diagonal (consistent with project_dag_spectral logic)
np.fill_diagonal(A_learned, 0)

# Récupérer le prior depuis CONFIG.loss.dag_prior si défini
_prior_cfg = CONFIG.loss.get("dag_prior", None)
if _prior_cfg is not None:
    A_prior = np.array(_prior_cfg, dtype=np.float64)
    if A_prior.shape != A_learned.shape:
        print(f"⚠️  prior shape {A_prior.shape} != learned shape {A_learned.shape} - skip SHD")
        shd_report = None
    else:
        shd_report = structural_hamming_distance(A_learned, A_prior, threshold=0.05)
        print("📊 SHD vs prior (threshold=0.05):")
        for k, v in shd_report.items():
            print(f"     {k}: {v}")

        # Plot la matrice apprise pour inspection visuelle
        import matplotlib.pyplot as _plt
        _fig, _axes = _plt.subplots(1, 2, figsize=(10, 4))
        _axes[0].imshow(A_learned, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
        _axes[0].set_title(f"A_learned (q={A_learned.shape[0]})")
        _axes[1].imshow(A_prior, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
        _axes[1].set_title("A_prior (config)")
        _plt.tight_layout()
        _plt.show()
else:
    print("ℹ️  CONFIG.loss.dag_prior non défini — SHD non calculé.")
    shd_report = None
'''


def _make_code_cell(src, cell_id):
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def _find_cell(cells, predicate):
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        if predicate("".join(c.get("source", []))):
            return i
    return None


# ---------------------------------------------------------------------
# Patcher : training notebook
# ---------------------------------------------------------------------

def patch_training_notebook() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    n_changed = 0

    # 1) build_diffusion : injection de _edm_config + arg edm_config=
    diff_idx = _find_cell(
        cells,
        lambda s: "diffusion = CausalDiffusionDecoder(" in s
        and "UNET_KWARGS" in s
        and "EDM_BUILD_CONFIG" not in s,
    )
    if diff_idx is not None:
        src = "".join(cells[diff_idx]["source"])
        # Insérer le bloc EDM_CONFIG juste avant ``diffusion = CausalDiffusionDecoder(``
        marker = "diffusion = CausalDiffusionDecoder("
        if marker in src:
            new_src = src.replace(marker, EDM_CONFIG_BLOCK + "\n" + marker, 1)
            # Ajouter edm_config=_edm_config dans les kwargs
            new_src = new_src.replace(
                'anti_checkerboard=bool(CONFIG.diffusion.get("anti_checkerboard", False)),\n).to(DEVICE)',
                'anti_checkerboard=bool(CONFIG.diffusion.get("anti_checkerboard", False)),\n'
                '    edm_config=_edm_config,  # None pour DDPM, EDMConfig pour edm_karras\n).to(DEVICE)',
                1,
            )
            cells[diff_idx]["source"] = new_src.splitlines(keepends=True)
            cells[diff_idx]["outputs"] = []
            cells[diff_idx]["execution_count"] = None
            n_changed += 1
            print(f"  ~ training cell {diff_idx} : EDM_BUILD_CONFIG injecté")
    else:
        print("  = training EDM_BUILD_CONFIG déjà appliqué (ou cellule introuvable)")

    # 2) Pre-flight sigma_data : insérer après la cellule build_diffusion (qui contient le decoder)
    if _find_cell(cells, lambda s: "EDM_PREFLIGHT_SIGMA_DATA" in s) is None:
        # Trouver la cellule où ``dataset`` existe (avant le for epoch). On cherche
        # la cellule qui contient ``DataLoader`` ou ``train_dataloader`` setup.
        loader_idx = _find_cell(
            cells, lambda s: "train_dataloader" in s and "build_loader" in s
        )
        if loader_idx is not None:
            cells.insert(
                loader_idx, _make_code_cell(PREFLIGHT_SIGMA_DATA, "edm_preflight_sigma")
            )
            n_changed += 1
            print(f"  + training pre-flight sigma_data inséré en cell {loader_idx}")
        else:
            print("  ! impossible d'insérer pre-flight (cellule loader introuvable)")
    else:
        print("  = training pre-flight sigma_data déjà inséré")

    # 3) COLAB_BOOTSTRAP : basculer sur edm-rewrite branch
    boot_idx = _find_cell(cells, lambda s: "COLAB_BOOTSTRAP" in s)
    if boot_idx is not None:
        src = "".join(cells[boot_idx]["source"])
        OLD_BRANCH = 'GIT_BRANCH: str = "main"'
        NEW_BRANCH = 'GIT_BRANCH: str = "edm-rewrite"  # Phase 1 EDM rewrite (Karras 2022)'
        if OLD_BRANCH in src:
            src = src.replace(OLD_BRANCH, NEW_BRANCH, 1)
            cells[boot_idx]["source"] = src.splitlines(keepends=True)
            cells[boot_idx]["outputs"] = []
            cells[boot_idx]["execution_count"] = None
            n_changed += 1
            print(f"  ~ training COLAB_BOOTSTRAP : branche -> edm-rewrite")
        elif "edm-rewrite" in src:
            print("  = training COLAB_BOOTSTRAP : branche déjà edm-rewrite")
        else:
            print("  ! GIT_BRANCH pattern non trouvé dans COLAB_BOOTSTRAP")

    if n_changed > 0:
        TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return n_changed


# ---------------------------------------------------------------------
# Patcher : validation notebook
# ---------------------------------------------------------------------

def patch_validation_notebook() -> int:
    nb = json.loads(VAL_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    n_changed = 0

    # 1) Diffusion rebuild : injection EDM_BUILD_CONFIG + edm_config arg
    diff_idx = _find_cell(
        cells,
        lambda s: "diffusion = CausalDiffusionDecoder(" in s
        and "VALIDATION_DIFF_REBUILD" in s
        and "EDM_BUILD_CONFIG" not in s,
    )
    if diff_idx is not None:
        src = "".join(cells[diff_idx]["source"])
        # Insérer EDM_CONFIG_BLOCK juste avant ``diffusion = CausalDiffusionDecoder(``
        marker = "diffusion = CausalDiffusionDecoder("
        new_src = src.replace(marker, EDM_CONFIG_BLOCK + "\n" + marker, 1)
        # Ajouter edm_config dans les kwargs (avant ).to(DEVICE))
        new_src = new_src.replace(
            "anti_checkerboard=bool(CONFIG.diffusion.get(\"anti_checkerboard\", False)),\n).to(DEVICE)",
            "anti_checkerboard=bool(CONFIG.diffusion.get(\"anti_checkerboard\", False)),\n"
            "    edm_config=_edm_config,\n).to(DEVICE)",
            1,
        )
        cells[diff_idx]["source"] = new_src.splitlines(keepends=True)
        cells[diff_idx]["outputs"] = []
        cells[diff_idx]["execution_count"] = None
        n_changed += 1
        print(f"  ~ validation cell {diff_idx} : EDM_BUILD_CONFIG injecté")
    elif _find_cell(cells, lambda s: "EDM_BUILD_CONFIG" in s) is not None:
        print("  = validation EDM_BUILD_CONFIG déjà injecté")
    else:
        print("  ! validation diffusion rebuild non trouvée")

    # 2) SHD test : insérer APRÈS la cellule de helpers (qui définit
    # convert_sample_to_batch, extract_target_and_baseline_t_mean, etc.).
    # Ces helpers sont dépendances de SHD pour reconstruire les états
    # ; placer SHD avant échouerait au runtime sur ``rcn_cell.A_dag``.
    if _find_cell(cells, lambda s: "EDM_SHD_TEST" in s) is None:
        helpers_idx = _find_cell(
            cells, lambda s: "def convert_sample_to_batch" in s
        )
        if helpers_idx is not None:
            cells.insert(helpers_idx + 1, _make_code_cell(SHD_TEST_BLOCK, "edm_shd_test"))
            n_changed += 1
            print(f"  + validation SHD test inséré en cell {helpers_idx + 1} (après helpers)")
        else:
            print("  ! validation SHD anchor introuvable (helpers)")
    else:
        print("  = validation SHD test déjà inséré")

    # 3) Bootstrap : edm-rewrite branch
    boot_idx = _find_cell(cells, lambda s: "VALIDATION_COLAB_BOOTSTRAP" in s)
    if boot_idx is not None:
        src = "".join(cells[boot_idx]["source"])
        OLD = 'GIT_BRANCH = "main"'
        NEW = 'GIT_BRANCH = "edm-rewrite"  # Phase 1 EDM rewrite'
        if OLD in src:
            src = src.replace(OLD, NEW, 1)
            cells[boot_idx]["source"] = src.splitlines(keepends=True)
            cells[boot_idx]["outputs"] = []
            cells[boot_idx]["execution_count"] = None
            n_changed += 1
            print(f"  ~ validation COLAB_BOOTSTRAP : branche -> edm-rewrite")
        elif "edm-rewrite" in src:
            print("  = validation COLAB_BOOTSTRAP : branche déjà edm-rewrite")

    if n_changed > 0:
        VAL_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return n_changed


def main() -> int:
    n = 0
    print("=== Training notebook ===")
    n += patch_training_notebook()
    print("\n=== Validation notebook ===")
    n += patch_validation_notebook()
    print(f"\n{n} modifications totales appliquées")
    return 0


if __name__ == "__main__":
    sys.exit(main())
