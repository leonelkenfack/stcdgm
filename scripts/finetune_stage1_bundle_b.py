"""Fine-tune Stage 1 with Bundle B + CASTLE + G_phys mask (Phase F, 2026-06-09).

Script orchestrateur autonome qui charge le checkpoint V5-mini et le ré-entraîne
25-30 epochs avec toutes les modifications Phases A à E intégrées :

- CASTLE-style joint prediction anchoring  (Phase A)
- Weighted MSE power-law + L1 annealing    (Phase B)
- Pinball, CC regularizer, spectral high-k (Phase C)
- Physical-prior mask G_phys                (Phase D)
- TailStratifiedSampler                    (Phase E)

Le script implémente sa propre boucle de training (slim) au lieu d'utiliser
``train_epoch_stage1`` du module legacy, pour avoir le contrôle fin nécessaire
aux nouvelles loss (notamment CC reg qui nécessite ``lr_input.requires_grad``).

Usage depuis Colab
------------------
.. code-block:: python

    from scripts.finetune_stage1_bundle_b import finetune_bundle_b

    finetune_bundle_b(
        stack=stack_v5,             # dict avec encoder, rcn_runner, regression_head, skip_block
        builder=builder,            # HeteroGraphBuilder
        train_dataset=train_ds,
        val_dataset=val_ds,         # pour la calibration sigma_data
        CONFIG=CONFIG,
        DEVICE=DEVICE,
        epochs=25,
        ckpt_save_dir=ORACLE_FINETUNED_DIR,
        batch_size=8,
        sanity_eval_every=5,
    )

References
----------
Voir architecture_journey.md §12 pour le plan détaillé et les cibles
d'amélioration métriques.
"""
from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.st_cdgm.models.causal_rcn import CASTLEAnchor
from src.st_cdgm.training.two_stage import (
    clausius_clapeyron_reg,
    gamma_dag_warmup,
    high_k_rapsd_loss,
    lambda_l1_cosine_anneal,
    pinball_loss,
    stage1_compute_loss,
)
from src.st_cdgm.training.physics_prior import (
    build_physical_mask,
    physical_prior_loss,
)
from src.st_cdgm.training.stage1_paths import (
    TailStratifiedSampler,
    compute_sample_max_values,
)


# ---------------------------------------------------------------------
# Hyperparamètres par défaut (recommandés par §12.7 de architecture_journey.md)
# ---------------------------------------------------------------------

DEFAULT_HYPERPARAMS: Dict[str, Any] = {
    # Optimizer (10× plus bas que from-scratch original)
    "lr_encoder": 5e-5,
    "lr_rcn": 3e-5,
    "lr_regression": 1e-4,
    "lr_skip": 5e-5,
    "lr_castle": 1e-4,
    "weight_decay": 1e-4,
    "gradient_clipping": 1.0,

    # Phase B — weighted MSE power-law
    "tail_weight_alpha": 0.5,
    "tail_weight_beta": 1.0,

    # Phase B — L1 cosine annealing
    "lambda_l1_start": 0.10,
    "lambda_l1_end": 0.01,

    # Phase A — CASTLE
    "lambda_castle_epoch_lt_10": 0.05,
    "lambda_castle_epoch_ge_10": 0.10,
    "castle_expansion": 2,

    # Phase C — Pinball
    "lambda_pinball": 0.20,
    "pinball_taus": (0.95, 0.99),

    # Phase C — CC regularizer (avec warmup epoch 0-10)
    "lambda_cc_reg_warmup_epochs": 10,
    "lambda_cc_reg_full": 0.05,
    "cc_rate": 0.07,

    # Phase C — Spectral high-k
    "lambda_spectral_highk": 0.05,
    "k_highk_min": 30,

    # Phase D — Physical prior mask
    # I1 fix companion: physical_prior_loss now defaults to normalize=False
    # (sum-of-squared-errors, not /N(N-1)). The old V5-mini default of 0.05
    # was calibrated to the divided form -> effectively lambda_eff = 0.05/30 = 0.00167
    # for 6-node, which was 60x weaker than L1 (lambda_l1 ~ 0.10). After I1 fix,
    # the unnormalized loss makes 0.05 too weak again. New value 0.40 gives
    # ~14x larger force than L1=0.01 per entry — meets KKT bound for both
    # 4-node and 6-node graphs.
    "lambda_dag_prior": 0.40,
    "g_phys_alpha": 0.20,

    # Phase E — Sampling
    "tail_fraction": 0.30,
    "tail_percentile": 95.0,

    # DAGMA (inchangé du V5-mini)
    "gamma_dag_max": 0.10,
    "gamma_dag_warmup_epochs": 5,

    # §1.1 fix: dag_grad_gate warmup schedule
    # Gate ramps 0 -> 1 over [warmup_start_epoch, warmup_end_epoch], so A_dag
    # receives prediction-loss gradient progressively (cold-start safety).
    # Before this fix, set_dag_grad_gate was never called and stayed at 0.0.
    "dag_gate_warmup_start_epoch": 5,
    "dag_gate_warmup_end_epoch": 20,
}


# ---------------------------------------------------------------------
# Schedule des lambda_* — §12.7 architecture_journey.md
# ---------------------------------------------------------------------


def schedule_lambdas(epoch: int, total_epochs: int, hp: Dict[str, Any]) -> Dict[str, float]:
    """Retourne tous les lambda_* à utiliser pour cette epoch.

    Consensus §1.1 fix: also schedules dag_grad_gate so A_dag receives
    gradient from the prediction loss (L_data) starting from epoch
    dag_gate_warmup_start_epoch, ramping linearly to 1.0 by epoch
    dag_gate_warmup_end_epoch. Before this fix, the gate was never set
    (remained at construction default 0.0), meaning A_dag was completely
    detached from the prediction objective and only shaped by L1+DAGMA+L_phys
    — the root cause of the observed Q_phys=0.40 band-diagonal collapse.
    """
    # §1.1: dag_grad_gate ramp 0 -> 1 over [warmup_start, warmup_end]
    # AI eng revise: scale schedule relative to total_epochs so short fine-tunes
    # (25 epochs) still get a meaningful full-gate phase. For total_epochs=25:
    #   start = min(5, 25//8) = min(5, 3) = 3
    #   end   = min(20, 25//4) = min(20, 6) = 6
    # For total_epochs=200 (full retrain):
    #   start = min(5, 25) = 5
    #   end   = min(20, 50) = 20
    gate_warmup_start_default = min(5, max(2, total_epochs // 8))
    gate_warmup_end_default = min(20, max(gate_warmup_start_default + 2, total_epochs // 4))
    gate_warmup_start = hp.get("dag_gate_warmup_start_epoch", gate_warmup_start_default)
    gate_warmup_end = hp.get("dag_gate_warmup_end_epoch", gate_warmup_end_default)
    if epoch < gate_warmup_start:
        dag_grad_gate = 0.0
    elif epoch >= gate_warmup_end:
        dag_grad_gate = 1.0
    else:
        dag_grad_gate = (epoch - gate_warmup_start) / max(
            gate_warmup_end - gate_warmup_start, 1
        )

    return {
        "lambda_l1": lambda_l1_cosine_anneal(
            epoch, total_epochs, hp["lambda_l1_start"], hp["lambda_l1_end"]
        ),
        "lambda_castle": (
            hp["lambda_castle_epoch_lt_10"] if epoch < 10
            else hp["lambda_castle_epoch_ge_10"]
        ),
        "lambda_pinball": hp["lambda_pinball"],
        "lambda_cc_reg": (
            0.0 if epoch < hp["lambda_cc_reg_warmup_epochs"]
            else hp["lambda_cc_reg_full"]
        ),
        "lambda_spectral_highk": hp["lambda_spectral_highk"],
        "lambda_dag_prior": hp["lambda_dag_prior"],
        "gamma_dag": gamma_dag_warmup(
            epoch, hp["gamma_dag_max"], hp["gamma_dag_warmup_epochs"]
        ),
        "dag_grad_gate": dag_grad_gate,  # §1.1 fix: wired to RCNCell.set_dag_grad_gate()
    }


# ---------------------------------------------------------------------
# Boucle de training pour une epoch
# ---------------------------------------------------------------------


def train_one_epoch_bundle_b(
    *,
    stack: Dict[str, Any],
    builder,
    data_loader,
    castle_anchor: CASTLEAnchor,
    G_phys: Tensor,
    optimizer: torch.optim.Optimizer,
    epoch_idx: int,
    total_epochs: int,
    hp: Dict[str, Any],
    device: torch.device,
    t850_channel_idx: int,
    convert_sample_to_batch_fn,
    verbose: bool = True,
) -> Dict[str, float]:
    """Une epoch de fine-tune avec toutes les Phase A-E losses actives.

    Returns
    -------
    dict de moyennes des composantes loss sur l'epoch.
    """
    lambdas = schedule_lambdas(epoch_idx, total_epochs, hp)

    encoder = stack["encoder"]
    rcn_runner = stack["rcn_runner"]
    regression_head = stack["regression_head"]
    skip_block = stack.get("skip_block")
    rcn_cell = rcn_runner.cell

    # §1.1 fix: wire dag_grad_gate so A_dag receives prediction-loss gradient.
    # Before this fix, set_dag_grad_gate was NEVER called in the entire
    # finetune script (grep confirms). The gate remained at construction
    # default 0.0, meaning L_data was detached from A_dag and only L1+DAGMA+
    # L_phys+L_castle could shape it. With L1 dominating (per KKT analysis,
    # consensus §1.8), A_dag collapsed to band-diagonal Q_phys=0.40.
    if hasattr(rcn_cell, "set_dag_grad_gate"):
        rcn_cell.set_dag_grad_gate(float(lambdas["dag_grad_gate"]))
    elif verbose and epoch_idx == 0:
        warnings.warn(
            "§1.1 fix: rcn_cell.set_dag_grad_gate not available — "
            "A_dag will NOT receive prediction-loss gradient. "
            "Q_phys is expected to converge to band-diagonal pattern."
        )

    # Modules en mode train
    encoder.train()
    rcn_cell.train()
    regression_head.train()
    if skip_block is not None:
        skip_block.train()
    castle_anchor.train()

    epoch_losses: Dict[str, List[float]] = {}
    n_batches = 0
    t0 = time.time()

    for batch_idx, raw_sample in enumerate(data_loader):
        batch = convert_sample_to_batch_fn(raw_sample, builder, device)
        target_residual = batch["residual"][-1].to(device)
        if target_residual.dim() == 3:
            target_residual = target_residual.unsqueeze(0)

        # === Forward Stage 1 avec lr_input grad-traçable pour CC reg ===
        lr_data = batch["lr"].to(device).detach().requires_grad_(lambdas["lambda_cc_reg"] > 0.0)

        H_init = encoder.init_state(batch["hetero"]).to(device)
        drivers = [lr_data[t] for t in range(lr_data.shape[0])]
        seq_out = rcn_runner.run(H_init, drivers, reconstruction_sources=None)
        H_last = seq_out.states[-1]
        A_dag_attached = seq_out.dag_matrices[-1]

        mu_c = regression_head(H_last)
        if mu_c.shape[-2:] != target_residual.shape[-2:]:
            mu_c = F.interpolate(
                mu_c, size=target_residual.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        # Skip block (V5-mini, A1)
        if skip_block is not None:
            lr_last = drivers[-1] if drivers[-1].dim() == 4 else drivers[-1].unsqueeze(0)
            try:
                mu_HR, _ = skip_block(lr_last, mu_c)
            except Exception:
                mu_HR = mu_c
        else:
            mu_HR = mu_c

        # === Compute CC regularizer (autograd grad on lr_data) ===
        cc_reg_tensor: Optional[Tensor] = None
        if lambdas["lambda_cc_reg"] > 0.0 and lr_data.requires_grad:
            try:
                cc_reg_tensor = clausius_clapeyron_reg(
                    mu_HR=mu_HR, lr_input=lr_data,
                    t850_channel_idx=t850_channel_idx,
                    cc_rate=hp["cc_rate"], weight=1.0,
                    create_graph=False,
                )
            except Exception as e:
                warnings.warn(f"CC reg failed (epoch {epoch_idx}): {type(e).__name__}: {e}")
                cc_reg_tensor = None

        # === Compute DAGMA + L1 + physical prior ===
        # Le DAGMA est calculé via la pénalité h(W) = tr(e^{W ◦ W}) - d
        # (DAGMA Bello 2022). On utilise A_dag absolute squared.
        A_dag_param = rcn_cell.A_dag  # the leaf Parameter (post-projection if any)
        A_dag_masked_post = A_dag_param - torch.diag(torch.diagonal(A_dag_param))
        A_dag_sq = A_dag_masked_post.pow(2)

        # DAGMA (log-det formulation, Bello 2022)
        d = A_dag_sq.size(0)
        # §1.2 fix: s must strictly exceed the spectral radius of A_dag_sq for
        # log-det(sI - A^2) to be defined. Bello 2022 §3.2 prescribes s > rho(A^2)
        # with margin. The previous hardcoded s=1.0 silently fails when PCMCI
        # init (or any non-trivial A_dag) has row-sum(A^2) > 1.
        # Use Gershgorin upper bound as a conservative spectral radius estimate.
        # AI eng revise: explicit .float() cast for autocast safety on A100 BF16,
        # and use math.log(s) instead of torch tensor roundtrip for efficiency.
        with torch.no_grad():
            gershgorin_bound = float(A_dag_sq.sum(dim=1).max().item())
        s = max(1.05 * gershgorin_bound + 1e-3, 1.0)
        try:
            # Force FP32 for slogdet stability under autocast (BF16/FP16 paths
            # silently fall back to FP32 internally but inductor + torch.compile
            # can break this fall-through). Explicit cast is invariant.
            A_dag_sq_f32 = A_dag_sq.float()
            M = s * torch.eye(d, device=A_dag_sq_f32.device, dtype=torch.float32) - A_dag_sq_f32
            # log-det positive seulement si M definite positive
            sign, logabsdet = torch.linalg.slogdet(M)
            if sign.item() > 0:
                # h(W) = -log det(sI - W^2) + d*log(s) per Bello 2022 Eq. 5
                # math.log(s) avoids creating a 0-d tensor + CPU sync per step
                import math as _math
                L_dag = -logabsdet + d * _math.log(s)
            else:
                # Fallback: h(W) = tr(exp(A * A)) - d (NOTEARS form)
                L_dag = torch.trace(torch.matrix_exp(A_dag_sq)) - d
        except Exception:
            L_dag = torch.trace(torch.matrix_exp(A_dag_sq)) - d

        L_l1 = A_dag_masked_post.abs().sum()

        # Phase D — physical prior mask
        L_phys = physical_prior_loss(
            A_dag_masked_post, G_phys.to(A_dag_masked_post.device),
            alpha=hp["g_phys_alpha"],
        )

        # === Assemble loss totale via stage1_compute_loss ===
        loss_total, components = stage1_compute_loss(
            mu_HR=mu_HR,
            target_residual=target_residual,
            rcn_reconstruction_loss=None,  # peut être ajouté si rec_loss dispo
            dagma_loss=L_dag,
            dag_l1_loss=L_l1,
            dag_prior_loss=L_phys,
            lambda_reg=1.0,
            beta_rec=0.05,
            gamma_dag=lambdas["gamma_dag"],
            lambda_l1=lambdas["lambda_l1"],
            lambda_dag_prior=lambdas["lambda_dag_prior"],
            # Phase B
            tail_weight_alpha=hp["tail_weight_alpha"],
            tail_weight_beta=hp["tail_weight_beta"],
            # Phase A — CASTLE
            castle_anchor=castle_anchor,
            castle_H_t=H_last,
            castle_A_dag=A_dag_attached,
            lambda_castle=lambdas["lambda_castle"],
            # Phase C
            lambda_pinball=lambdas["lambda_pinball"],
            pinball_taus=hp["pinball_taus"],
            cc_reg_loss=cc_reg_tensor,
            lambda_cc_reg=lambdas["lambda_cc_reg"],
            lambda_spectral_highk=lambdas["lambda_spectral_highk"],
            k_highk_min=hp["k_highk_min"],
        )

        # === Backward + grad clip + step ===
        # P0 fix : NaN/Inf guard. Si la loss est non-finite (overflow CC reg,
        # slogdet fail, spectral log explosion), skip le step pour ne pas
        # corrompre les poids ni le checkpoint inprogress.
        if not torch.isfinite(loss_total):
            warnings.warn(
                f"[NaN guard] epoch {epoch_idx}, batch {batch_idx} : "
                f"loss_total non-finite ({loss_total.item()}). Skip step."
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        if hp.get("gradient_clipping", None):
            all_params = []
            for group in optimizer.param_groups:
                all_params.extend(group["params"])
            torch.nn.utils.clip_grad_norm_(all_params, hp["gradient_clipping"])
        optimizer.step()

        # P1 fix : DAG anti-collapse projections apres step.
        # Avec lambda_l1_start=0.10 (10x baseline), A_dag peut collapser
        # vers 0 sur premieres epochs. project_dag_spectral garde le rayon
        # spectral < 0.95 (acyclicite), project_dag_floor preserve le prior.
        try:
            if hasattr(rcn_cell, "project_dag_spectral"):
                rcn_cell.project_dag_spectral(max_radius=0.95)
            if hasattr(rcn_cell, "project_dag_floor"):
                rcn_cell.project_dag_floor(min_norm=0.10, prior=G_phys)
        except Exception as e:
            if epoch_idx == 0 and batch_idx == 0:
                warnings.warn(f"DAG projection skipped: {type(e).__name__}: {e}")

        # === Logging ===
        for k, v in components.items():
            epoch_losses.setdefault(k, []).append(v)
        n_batches += 1

        if verbose and (batch_idx + 1) % 10 == 0:
            print(
                f"  [epoch {epoch_idx:02d}/{total_epochs}] batch {batch_idx+1} : "
                f"loss={components.get('loss_total', 0):.4f}  "
                f"reg={components.get('loss_reg', 0):.4f}  "
                f"castle={components.get('loss_castle', 0):.4f}  "
                f"pinball={components.get('loss_pinball', 0):.4f}  "
                f"cc={components.get('loss_cc_reg', 0):.4f}  "
                f"spec={components.get('loss_spec_highk', 0):.4f}"
            )

    elapsed = time.time() - t0
    avg = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
    avg["n_batches"] = n_batches
    avg["time_sec"] = elapsed
    avg.update({f"lambda_{k}": v for k, v in lambdas.items()})

    # A_dag stats pour sanity
    with torch.no_grad():
        A_now = (rcn_cell.A_dag - torch.diag(torch.diagonal(rcn_cell.A_dag))).detach().cpu()
        avg["A_dag_norm"] = float(A_now.norm().item())
        avg["A_dag_max"] = float(A_now.abs().max().item())
        avg["A_dag_var"] = float(A_now[A_now.abs() > 0.01].var().item()) if (A_now.abs() > 0.01).any() else 0.0

    return avg


# ---------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------


def finetune_bundle_b(
    *,
    stack: Dict[str, Any],
    builder,
    train_dataset,
    val_dataset,
    CONFIG,
    DEVICE: torch.device,
    epochs: int = 25,
    batch_size: int = 8,
    ckpt_save_dir: Path,
    convert_sample_to_batch_fn,
    sanity_eval_every: int = 5,
    hp_override: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    skip_sigma_data_recalib: bool = False,
) -> Dict[str, Any]:
    """Pipeline de fine-tune complet Bundle B + CASTLE + G_phys.

    Parameters
    ----------
    stack : dict
        Dict avec clés ``encoder``, ``rcn_runner``, ``regression_head``,
        ``skip_block`` (optionnel), ``diffusion`` (sera frozen).
    builder : HeteroGraphBuilder
    train_dataset : Iterable
        Dataset Stage 1 (avec clé ``residual``).
    val_dataset : Iterable
        Dataset pour la calibration ``sigma_data`` post-train.
    CONFIG : OmegaConf-like
        Pour récupérer lr_variables, encoder.hidden_dim, etc.
    DEVICE : torch.device
    epochs : int
        Nombre d'epochs de fine-tune. Défaut 25.
    batch_size : int
    ckpt_save_dir : Path
        Dossier où sauvegarder le checkpoint post-train.
    convert_sample_to_batch_fn : callable
        Fonction qui transforme un raw sample en batch (existe déjà dans
        le notebook).
    sanity_eval_every : int
        Imprime sanity check tous les N epochs.
    hp_override : dict, optional
        Override les hyperparamètres par défaut.
    seed : int
    skip_sigma_data_recalib : bool
        Si True, skip la recalibration sigma_data en fin de run.

    Returns
    -------
    dict avec ``history`` (liste de dicts par epoch), ``final_ckpt``,
    ``sigma_data_new``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    hp = dict(DEFAULT_HYPERPARAMS)
    if hp_override is not None:
        hp.update(hp_override)

    print("=" * 70)
    print(f"Fine-tune Bundle B + CASTLE + G_phys ({epochs} epochs)")
    print("=" * 70)
    print(f"  Device : {DEVICE}")
    print(f"  Batch size : {batch_size}")
    print(f"  Train samples : {len(train_dataset)}")
    print(f"  Hyperparams : {hp}")
    print()

    # === Setup TailStratifiedSampler ===
    print("[Setup] Computing sample max values for tail stratification...")
    sample_maxes = compute_sample_max_values(train_dataset, field_key="residual")
    p_thr = float(np.percentile(sample_maxes, hp["tail_percentile"]))
    print(f"  P{hp['tail_percentile']:.0f} threshold : {p_thr:.4f} (log1p mm/day)")

    sampler = TailStratifiedSampler(
        sample_maxes, p_thr,
        tail_fraction=hp["tail_fraction"],
        batch_size=batch_size,
        num_samples=len(train_dataset),
        seed=seed,
    )
    print(f"  {sampler}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=torch.utils.data.BatchSampler(sampler, batch_size, drop_last=True),
        collate_fn=lambda x: x[0],  # convert_sample_to_batch s'occupe du collate
    )

    # === Setup CASTLE anchor ===
    print("[Setup] Initializing CASTLE anchor...")
    num_vars = int(CONFIG.encoder.get("num_vars", 6))
    hidden_dim = int(CONFIG.encoder.hidden_dim)
    castle_anchor = CASTLEAnchor(
        num_vars=num_vars, hidden_dim=hidden_dim,
        expansion=hp["castle_expansion"],
    ).to(DEVICE)
    print(f"  CASTLE : {sum(p.numel() for p in castle_anchor.parameters())} params")

    # === Setup G_phys ===
    print("[Setup] Building physical mask G_phys...")
    G_phys = build_physical_mask(num_vars=num_vars).to(DEVICE)
    print(f"  G_phys non-zero : {int((G_phys != 0).sum().item())} edges")

    # === Find t850_channel_idx ===
    lr_vars = list(CONFIG.data.lr_variables)
    t850_channel_idx = lr_vars.index("t_850") if "t_850" in lr_vars else 2
    print(f"  t_850 channel index : {t850_channel_idx} (of {len(lr_vars)} LR vars)")

    # === Freeze diffusion (Stage 2 stays static) ===
    if "diffusion" in stack and stack["diffusion"] is not None:
        for p in stack["diffusion"].parameters():
            p.requires_grad_(False)
        stack["diffusion"].eval()
        print("  Diffusion (Stage 2) frozen")

    # === Setup optimizer with per-group LR ===
    encoder = stack["encoder"]
    rcn_cell = stack["rcn_runner"].cell
    regression_head = stack["regression_head"]
    skip_block = stack.get("skip_block")

    param_groups = [
        {"params": list(encoder.parameters()), "lr": hp["lr_encoder"]},
        {"params": list(rcn_cell.parameters()), "lr": hp["lr_rcn"]},
        {"params": list(regression_head.parameters()), "lr": hp["lr_regression"]},
        {"params": list(castle_anchor.parameters()), "lr": hp["lr_castle"]},
    ]
    if skip_block is not None:
        param_groups.append({"params": list(skip_block.parameters()), "lr": hp["lr_skip"]})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=hp["weight_decay"])
    print(f"[Setup] Optimizer : AdamW with {len(param_groups)} param groups")

    # === Resume logic (post-V5-mini robustness for Colab disconnect) ===
    ckpt_save_dir = Path(ckpt_save_dir)
    ckpt_save_dir.mkdir(parents=True, exist_ok=True)
    inprogress_path = ckpt_save_dir / "epoch_finetuned_inprogress.pth"

    history: List[Dict[str, float]] = []
    start_epoch = 0

    # === Diagnostic visible : afficher l'etat des checkpoints sur disque ===
    print()
    print("=" * 70)
    print(f"[Resume diagnostic] Verification des checkpoints dans {ckpt_save_dir}/")
    for fname in ["epoch_finetuned_inprogress.pth", "epoch_finetuned.pth", "epoch_last.pth"]:
        fp = ckpt_save_dir / fname
        if fp.exists():
            size_mb = fp.stat().st_size / 1024**2
            print(f"  [TROUVE] {fname} ({size_mb:.0f} MB)")
        else:
            print(f"  [ABSENT] {fname}")
    print("=" * 70)

    if inprogress_path.exists():
        try:
            print(f"\n[Resume] Reprise depuis {inprogress_path.name}")
            ckpt = torch.load(inprogress_path, map_location=DEVICE, weights_only=False)
            saved_epoch = int(ckpt.get("epoch", 0))
            if saved_epoch >= epochs:
                print(f"  [Resume] Saved epoch ({saved_epoch}) >= target ({epochs}). Skip training.")
                start_epoch = epochs
            else:
                # Load all state
                encoder.load_state_dict(ckpt["encoder_state_dict"], strict=False)
                rcn_cell.load_state_dict(ckpt["rcn_cell_state_dict"], strict=False)
                regression_head.load_state_dict(ckpt["regression_head_state_dict"], strict=False)
                castle_anchor.load_state_dict(ckpt["castle_anchor_state_dict"], strict=False)
                if skip_block is not None and "skip_block_state_dict" in ckpt:
                    try:
                        skip_block.load_state_dict(ckpt["skip_block_state_dict"], strict=False)
                    except Exception as e:
                        print(f"  [WARN] skip_block load failed: {e}")
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception as e:
                    print(f"  [WARN] optimizer state load failed: {e} (will use fresh state)")
                history = ckpt.get("history", [])
                start_epoch = saved_epoch
                print(f"  [Resume] Reprise a epoch {start_epoch + 1}/{epochs}")
                print(f"  [Resume] Historique : {len(history)} epochs sauvegardes")
        except Exception as e:
            print(f"[WARN] Resume failed ({type(e).__name__}: {e}). Start from scratch.")
            history = []
            start_epoch = 0

    # === Training loop ===
    t_global = time.time()

    for epoch_idx in range(start_epoch, epochs):
        print(f"\n--- Epoch {epoch_idx + 1}/{epochs} ---")
        avg = train_one_epoch_bundle_b(
            stack=stack, builder=builder,
            data_loader=train_loader,
            castle_anchor=castle_anchor,
            G_phys=G_phys, optimizer=optimizer,
            epoch_idx=epoch_idx, total_epochs=epochs,
            hp=hp, device=DEVICE,
            t850_channel_idx=t850_channel_idx,
            convert_sample_to_batch_fn=convert_sample_to_batch_fn,
            verbose=True,
        )
        history.append(avg)

        if (epoch_idx + 1) % sanity_eval_every == 0 or epoch_idx == epochs - 1:
            print(f"  [Sanity] A_dag norm={avg['A_dag_norm']:.4f}  max={avg['A_dag_max']:.4f}  var={avg['A_dag_var']:.6f}")
            print(f"  [Sanity] avg loss={avg.get('loss_total', 0):.4f}  ({avg.get('time_sec', 0):.1f}s)")

        # === Persist intermediate checkpoint a chaque epoch (resume-safe) ===
        try:
            intermediate_state = {
                "encoder_state_dict": encoder.state_dict(),
                "rcn_cell_state_dict": rcn_cell.state_dict(),
                "regression_head_state_dict": regression_head.state_dict(),
                "castle_anchor_state_dict": castle_anchor.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "hyperparameters": hp,
                "epoch": epoch_idx + 1,   # epochs completes
                "epochs_target": epochs,
            }
            if skip_block is not None:
                intermediate_state["skip_block_state_dict"] = skip_block.state_dict()
            # Save to temp file then atomic rename (resilient to Drive sync interruption)
            tmp_path = inprogress_path.with_suffix(".pth.tmp")
            torch.save(intermediate_state, tmp_path)
            try:
                import os as _os
                _os.replace(tmp_path, inprogress_path)
            except Exception:
                # Fallback : direct save si replace fail
                torch.save(intermediate_state, inprogress_path)
            # Force Drive sync flush (Colab specifique : fsync ou flush_and_unmount)
            try:
                import os as _os
                with open(inprogress_path, "rb") as _f:
                    _os.fsync(_f.fileno())
            except Exception:
                pass
            # Print [Persist] CHAQUE epoch (pas seulement sanity) pour que
            # l'utilisateur voit clairement que la persistance fonctionne.
            size_mb = inprogress_path.stat().st_size / 1024**2
            print(f"  [Persist] {inprogress_path.name} sauve (epoch {epoch_idx + 1}/{epochs}, {size_mb:.0f} MB)")
        except Exception as e:
            warnings.warn(f"Per-epoch checkpoint save failed: {type(e).__name__}: {e}")

    print(f"\n[OK] Training terminé en {(time.time() - t_global)/60:.1f} min")

    # === Save final checkpoint (renomme l'inprogress) ===
    ckpt_path = ckpt_save_dir / "epoch_finetuned.pth"

    state = {
        "encoder_state_dict": encoder.state_dict(),
        "rcn_cell_state_dict": rcn_cell.state_dict(),
        "regression_head_state_dict": regression_head.state_dict(),
        "castle_anchor_state_dict": castle_anchor.state_dict(),
        "history": history,
        "hyperparameters": hp,
        "epoch": epochs,
    }
    if skip_block is not None:
        state["skip_block_state_dict"] = skip_block.state_dict()
    torch.save(state, ckpt_path)
    print(f"[OK] Checkpoint final sauvegarde : {ckpt_path}")

    # BUG fix : sauve aussi epoch_last.pth pour que CHECKPOINT_NAME='epoch_last'
    # (defaut) recharge bien les poids fine-tunes quand EVAL_VERSION='finetuned'.
    # Sans ce save, Cell 4 lirait l'ancien epoch_last.pth (copie pre-training
    # du baseline) et l'utilisateur penserait que Phase F n'a rien change.
    ckpt_alias = ckpt_save_dir / "epoch_last.pth"
    torch.save(state, ckpt_alias)
    print(f"[OK] Alias sauvegarde : {ckpt_alias.name} (pour EVAL_VERSION='finetuned')")

    # Cleanup : supprime l'inprogress puisque le final est en place
    if inprogress_path.exists():
        try:
            inprogress_path.unlink()
            print(f"[OK] Cleanup : {inprogress_path.name} supprime (run termine)")
        except Exception as e:
            warnings.warn(f"Cleanup inprogress failed: {e}")

    # === Recalibrate sigma_data ===
    sigma_data_new = None
    if not skip_sigma_data_recalib and "diffusion" in stack:
        print("\n[Sigma_data] Recalibrating sigma_data on val_dataset (inline)...")
        try:
            # Inline recalibration : empirical std of delta_target =
            # log1p(HR) - log1p(baseline) - mu_HR over the val set.
            # Plus simple et moins fragile que calibrate_sigma_data_variant.
            deltas = []
            n_done = 0
            for i in range(min(len(val_dataset), 200)):
                try:
                    sample = val_dataset[i]
                    batch = convert_sample_to_batch_fn(sample, builder, DEVICE)
                    target_res = batch["residual"][-1].to(DEVICE)
                    if target_res.dim() == 3:
                        target_res = target_res.unsqueeze(0)
                    with torch.no_grad():
                        H_init = encoder.init_state(batch["hetero"]).to(DEVICE)
                        drivers = [batch["lr"].to(DEVICE)[t] for t in range(batch["lr"].shape[0])]
                        seq = stack["rcn_runner"].run(H_init, drivers, reconstruction_sources=None)
                        mu_c = regression_head(seq.states[-1])
                        if mu_c.shape[-2:] != target_res.shape[-2:]:
                            mu_c = torch.nn.functional.interpolate(
                                mu_c, size=target_res.shape[-2:],
                                mode="bilinear", align_corners=False,
                            )
                        if skip_block is not None:
                            lr_last = drivers[-1] if drivers[-1].dim() == 4 else drivers[-1].unsqueeze(0)
                            try:
                                mu_HR_pred, _ = skip_block(lr_last, mu_c)
                            except Exception:
                                mu_HR_pred = mu_c
                        else:
                            mu_HR_pred = mu_c
                        delta = target_res - mu_HR_pred
                        valid = torch.isfinite(delta)
                        if valid.any():
                            deltas.append(delta[valid].std().item())
                            n_done += 1
                except Exception as ex_inner:
                    warnings.warn(f"sigma_data sample {i}: {ex_inner}")
            if deltas:
                import numpy as _np
                sigma_data_new = float(_np.mean(deltas))
                print(f"  [OK] sigma_data_new = {sigma_data_new:.6f} (sur {n_done} samples)")
                # BUG fix : propage sigma_data_new au stack["diffusion"] in-memory
                # pour que Phase 7 dans la meme session kernel l'utilise.
                try:
                    if hasattr(stack["diffusion"], "edm_config"):
                        old_sigma = stack["diffusion"].edm_config.sigma_data
                        stack["diffusion"].edm_config.sigma_data = sigma_data_new
                        print(f"  [OK] stack[\"diffusion\"].edm_config.sigma_data : {old_sigma:.6f} -> {sigma_data_new:.6f}")
                except Exception as ex_prop:
                    warnings.warn(f"sigma_data propagation skipped: {ex_prop}")
            else:
                print(f"  [WARN] Aucun sample valide pour sigma_data, on garde l'ancien")
                # Preserve OLD sigma_data so JSON consumers can do float() safely
                try:
                    if hasattr(stack.get("diffusion"), "edm_config"):
                        sigma_data_new = float(stack["diffusion"].edm_config.sigma_data)
                except Exception:
                    sigma_data_new = 0.5  # EDM default fallback

            # Save into checkpoint dict for downstream consumption
            state["sigma_data_new"] = sigma_data_new
            torch.save(state, ckpt_path)
            torch.save(state, ckpt_save_dir / "epoch_last.pth")
        except Exception as e:
            warnings.warn(f"sigma_data recalibration failed: {type(e).__name__}: {e}")

    # === Save history JSON ===
    history_path = ckpt_save_dir / "finetune_history.json"
    history_path.write_text(
        json.dumps({"history": history, "hyperparameters": hp,
                     "sigma_data_new": sigma_data_new}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[OK] History sauvegardé : {history_path}")

    return {
        "history": history,
        "final_ckpt": str(ckpt_path),
        "sigma_data_new": sigma_data_new,
        "hyperparameters": hp,
    }


__all__ = [
    "DEFAULT_HYPERPARAMS",
    "schedule_lambdas",
    "train_one_epoch_bundle_b",
    "finetune_bundle_b",
]
