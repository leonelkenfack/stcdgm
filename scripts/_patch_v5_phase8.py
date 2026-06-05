"""Patch st_cdgm_v5_evaluation.ipynb : remplace Cell 9 par Phase 8 complete."""
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

NB_PATH = Path("st_cdgm_v5_evaluation.ipynb")
with NB_PATH.open(encoding="utf-8") as f:
    nb = json.load(f)

BACKUP = Path("st_cdgm_v5_evaluation.ipynb.bak")
if not BACKUP.exists():
    BACKUP.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] Backup cree : {BACKUP}")

PHASE8 = r'''# ============================================================
# Phase 8 - EXECUTION COMPLETE
# Charge les 2 modeles, evalue 3 interventions, sauvegarde Q_int.
#
# Pre-requis :
#   - CONFIG charge (depuis training_evaluation cell 14-20)
#   - builder, test_dataset prepares
#   - DEVICE (typiquement torch.device('cuda'))
#   - convert_sample_to_batch defini
#
# Si pas encore charge : execute d'abord les cellules 14 a 30 de
#   st_cdgm_training_evaluation.ipynb puis reviens ici.
# ============================================================

import json
import time
import torch
import numpy as np

# 0. Verification des pre-requis
required_globals = ['CONFIG', 'builder', 'DEVICE', 'convert_sample_to_batch']
missing = [g for g in required_globals if g not in globals()]
if missing:
    raise RuntimeError(
        f"Globals manquants : {missing}. "
        "Execute d'abord les cellules d'init de st_cdgm_training_evaluation.ipynb."
    )

print(f"[OK] Pre-requis valides : {required_globals}")
print(f"     DEVICE = {DEVICE}")
print(f"     lr_variables = {list(CONFIG.data.lr_variables)}")
print()

# ── 1. Helpers de chargement d'un stack complet depuis un checkpoint ─────
from st_cdgm.models import (
    IntelligibleVariableEncoder, IntelligibleVariableConfig,
    GraphToGridDecoder, RCNCell, RCNSequenceRunner,
    CausalDiffusionDecoder,
)
try:
    from st_cdgm.models import ConditionalSkipBlock
    SKIP_AVAILABLE = True
except ImportError:
    SKIP_AVAILABLE = False
    print("[INFO] ConditionalSkipBlock non disponible (V5 pas encore deploye)")

def build_stack_from_ckpt(ckpt_path, variant_name):
    """Construit encoder + RCN + regression_head + (skip si V5) + diffusion
    et charge les poids depuis ckpt_path."""
    print(f"  [{variant_name}] Loading {ckpt_path}")
    t0 = time.time()
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    print(f"  [{variant_name}] Checkpoint loaded in {time.time()-t0:.1f}s "
          f"(epoch={ckpt.get('epoch')}, causal_concat={ckpt.get('causal_concat')})")

    enc_cfg = IntelligibleVariableConfig(
        num_variables=len(CONFIG.data.lr_variables),
        hidden_dim=CONFIG.encoder.hidden_dim,
        conditioning_dim=CONFIG.encoder.conditioning_dim,
        num_dag_tokens=int(CONFIG.encoder.get("num_dag_tokens", 2)),
        causal_conditioning=bool(CONFIG.encoder.get("causal_conditioning", True)),
    )
    encoder = IntelligibleVariableEncoder(enc_cfg).to(DEVICE)

    rcn_cell = RCNCell(
        num_variables=enc_cfg.num_variables,
        hidden_dim=CONFIG.rcn.hidden_dim,
        driver_dim=CONFIG.rcn.driver_dim,
        reconstruction_dim=CONFIG.rcn.reconstruction_dim,
        dropout=CONFIG.rcn.dropout,
    ).to(DEVICE)
    rcn_runner = RCNSequenceRunner(rcn_cell, detach_interval=CONFIG.rcn.detach_interval)

    regression_head = GraphToGridDecoder(
        d_model=CONFIG.encoder.hidden_dim,
        hr_h=CONFIG.graph.hr_shape[0],
        hr_w=CONFIG.graph.hr_shape[1],
    ).to(DEVICE)

    from st_cdgm.models.edm_preconditioner import EDMConfig
    edm_cfg = EDMConfig.from_yaml_dict(CONFIG.diffusion.get("edm", {}))
    diffusion = CausalDiffusionDecoder(
        in_channels=CONFIG.diffusion.in_channels,
        conditioning_dim=CONFIG.diffusion.conditioning_dim,
        height=CONFIG.diffusion.height,
        width=CONFIG.diffusion.width,
        scheduler_type=CONFIG.diffusion.scheduler_type,
        causal_concat=True,
        edm_config=edm_cfg,
        unet_kwargs=dict(CONFIG.diffusion.unet_kwargs),
    ).to(DEVICE)

    def _load(name, module):
        key = f"{name}_state_dict"
        if key not in ckpt:
            print(f"  [{variant_name}] [WARN] {key} absent")
            return False
        module.load_state_dict(ckpt[key])
        return True
    _load("encoder", encoder)
    _load("rcn_cell", rcn_cell)
    _load("regression_head", regression_head)
    _load("diffusion", diffusion)

    skip_block = None
    if SKIP_AVAILABLE and "skip_block_state_dict" in ckpt:
        skip_block = ConditionalSkipBlock(
            lr_channels=len(CONFIG.data.lr_variables),
            hr_shape=tuple(CONFIG.graph.hr_shape),
        ).to(DEVICE)
        skip_block.load_state_dict(ckpt["skip_block_state_dict"])
        print(f"  [{variant_name}] [+] ConditionalSkipBlock loaded ({skip_block.num_params()} params)")

    encoder.eval(); rcn_cell.eval(); regression_head.eval(); diffusion.eval()
    if skip_block is not None:
        skip_block.eval()

    return {
        "encoder": encoder, "rcn_runner": rcn_runner,
        "regression_head": regression_head, "diffusion": diffusion,
        "skip_block": skip_block,
        "variant": variant_name,
    }

# ── 2. Chargement des deux stacks ──────────────────────────────────
print("Chargement des 2 stacks (V5 + Noncausal)...")
print()
t_load = time.time()
stack_v5 = build_stack_from_ckpt(V5_DIR / "epoch_last.pth", "V5")
print()
stack_nc = build_stack_from_ckpt(NONCAUSAL_DIR / "epoch_last.pth", "Noncausal")
print()
print(f"[OK] 2 stacks charges en {time.time()-t_load:.1f}s")
print()

# ── 3. Fonction predict generique ──────────────────────────────────
@torch.no_grad()
def predict_with_stack(stack, batch, K=4, n_steps=32):
    """Genere un ensemble [K, B, 1, H, W] depuis un stack + batch."""
    encoder = stack["encoder"]
    rcn_runner = stack["rcn_runner"]
    regression_head = stack["regression_head"]
    diffusion = stack["diffusion"]
    skip_block = stack["skip_block"]

    lr_data = batch["lr"].to(DEVICE)
    H_init = encoder.init_state(batch["hetero"]).to(DEVICE)
    drivers = [lr_data[t] for t in range(lr_data.shape[0])]
    seq_out = rcn_runner.run(H_init, drivers, reconstruction_sources=None)
    H_T = seq_out.states[-1]
    mu_HR_causal = regression_head(H_T)

    target_shape = batch["residual"][-1].to(DEVICE).shape
    if target_shape[-2:] != mu_HR_causal.shape[-2:]:
        mu_HR_causal = torch.nn.functional.interpolate(
            mu_HR_causal, size=target_shape[-2:],
            mode="bilinear", align_corners=False,
        )
    if skip_block is not None:
        lr_last = drivers[-1] if drivers[-1].dim() == 4 else drivers[-1].unsqueeze(0)
        mu_HR, alpha = skip_block(lr_last, mu_HR_causal)
    else:
        mu_HR = mu_HR_causal

    mu_HR = torch.nan_to_num(mu_HR, nan=0.0, posinf=0.0, neginf=0.0)

    baseline_t = batch["baseline"][-1].to(DEVICE)
    if baseline_t.dim() == mu_HR.dim() - 1:
        baseline_t = baseline_t.unsqueeze(0)
    baseline_log = torch.nan_to_num(baseline_t, nan=0.0, posinf=0.0, neginf=0.0)

    ensemble = []
    for k in range(K):
        out = diffusion.sample(
            conditioning=None,
            num_steps=n_steps,
            scheduler_type="edm_karras",
            apply_constraints=False,
            mu_HR=mu_HR,
            baseline_log=baseline_log,
        )
        residual = out.residual if hasattr(out, 'residual') else out
        full_pred = baseline_log + mu_HR + residual
        ensemble.append(full_pred.cpu())

    return torch.stack(ensemble, dim=0)

# ── 4. Resolution des interventions ──────────────────────────────
from scripts.intervention_test import (
    INTERVENTIONS, resolve_variable_indices,
    apply_intervention, evaluate_intervention,
    compute_q_int, save_intervention_results,
)
lr_vars = list(CONFIG.data.lr_variables)
resolved = resolve_variable_indices(lr_vars)
print("Interventions resolved :")
for spec in resolved:
    status = "OK" if spec['variable_idx'] is not None else "SKIP"
    print(f"  [{status}] {spec['name']:<32s} {spec['variable_name']:<10s} idx={spec['variable_idx']}")
print()

# ── 5. Boucle d'evaluation ──────────────────────────────────────
N_BATCHES_INT = 2
K_SAMPLES_INT = 4

print(f"Configuration : {N_BATCHES_INT} batches x {K_SAMPLES_INT} K-samples x 3 interventions x 2 variants")
est_seconds = N_BATCHES_INT * K_SAMPLES_INT * 3 * 2 * 2 * 5
print(f"Estimation cout : ~{est_seconds}s ({est_seconds/60:.0f}min)")
print()

t_inf = time.time()
results_v5 = []
results_nc = []

for spec_idx, spec in enumerate(resolved):
    if spec['variable_idx'] is None:
        results_v5.append({"intervention": spec['name'], "skipped": True,
                          "reason": f"variable {spec['variable_name']} absent du LR"})
        results_nc.append({"intervention": spec['name'], "skipped": True,
                          "reason": f"variable {spec['variable_name']} absent du LR"})
        continue

    print(f"[{spec_idx+1}/{len(resolved)}] {spec['name']}")
    deltas_v5 = []
    deltas_nc = []

    sample_iter = iter(test_dataset)
    for batch_idx in range(N_BATCHES_INT):
        try:
            sample = next(sample_iter)
        except StopIteration:
            break
        batch_normal = convert_sample_to_batch(sample, builder, DEVICE)

        pred_normal_v5 = predict_with_stack(stack_v5, batch_normal, K=K_SAMPLES_INT)
        batch_int = dict(batch_normal)
        batch_int['lr'] = apply_intervention(batch_normal['lr'], spec, standardization=None)
        pred_int_v5 = predict_with_stack(stack_v5, batch_int, K=K_SAMPLES_INT)
        deltas_v5.append((pred_int_v5.nanmean(0) - pred_normal_v5.nanmean(0)).mean().item())

        pred_normal_nc = predict_with_stack(stack_nc, batch_normal, K=K_SAMPLES_INT)
        pred_int_nc = predict_with_stack(stack_nc, batch_int, K=K_SAMPLES_INT)
        deltas_nc.append((pred_int_nc.nanmean(0) - pred_normal_nc.nanmean(0)).mean().item())

    delta_v5 = float(np.mean(deltas_v5))
    delta_nc = float(np.mean(deltas_nc))
    sign_pred_v5 = 1 if delta_v5 > 0 else (-1 if delta_v5 < 0 else 0)
    sign_pred_nc = 1 if delta_nc > 0 else (-1 if delta_nc < 0 else 0)
    expected = int(spec['expected_sign'])

    results_v5.append({
        "intervention": spec['name'], "variable": spec['variable_name'],
        "delta_pred_mean": delta_v5,
        "sign_predicted": sign_pred_v5, "sign_expected": expected,
        "match": sign_pred_v5 == expected, "skipped": False,
        "physical_justification": spec['physical_justification'],
    })
    results_nc.append({
        "intervention": spec['name'], "variable": spec['variable_name'],
        "delta_pred_mean": delta_nc,
        "sign_predicted": sign_pred_nc, "sign_expected": expected,
        "match": sign_pred_nc == expected, "skipped": False,
        "physical_justification": spec['physical_justification'],
    })
    print(f"  V5: delta={delta_v5:+.5f} sign={sign_pred_v5:+d} expected={expected:+d} match={'YES' if sign_pred_v5==expected else 'NO'}")
    print(f"  NC: delta={delta_nc:+.5f} sign={sign_pred_nc:+d} expected={expected:+d} match={'YES' if sign_pred_nc==expected else 'NO'}")

print()
print(f"[OK] Phase 8 terminee en {(time.time()-t_inf)/60:.1f} min")
print()

save_intervention_results(results_v5, results_nc, RESULTS_DIR / "phase8_intervention.json")
q_v5 = compute_q_int(results_v5)
q_nc = compute_q_int(results_nc)
print()
print(f"Q_int V5         : {q_v5:.3f}")
print(f"Q_int Noncausal  : {q_nc:.3f}")
print(f"V5 wins intervention : {'YES' if q_v5 > q_nc else 'NO (egal ou perdant)'}")
'''

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [ln + "\n" for ln in PHASE8.split("\n")[:-1]] + [PHASE8.split("\n")[-1]],
}
nb["cells"][9] = new_cell

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[OK] Cell 9 remplacee. Notebook sauvegarde : {NB_PATH}")
print(f"[OK] {len(nb['cells'])} cellules au total")
