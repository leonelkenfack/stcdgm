# === Cell 8 : Stage 1 training loop from-scratch ===
# 15 epochs (or 2 in SMOKE_MODE), Adam optimizer.
# DAG prior + L1 sparsity cosine decay + physical losses warmup at epoch 10+.
import time
from torch.optim import Adam

# Optimizer (all Stage 1 modules trainable)
stage1_params = (
    list(encoder.parameters()) + list(rcn_cell.parameters())
    + list(regression_head.parameters()) + list(dual_path.parameters())
)
optimizer_s1 = Adam(stage1_params, lr=STAGE1_LR)

# L1 sparsity schedule (cosine decay)
def _l1_schedule(epoch, total):
    if total <= 1: return LAMBDA_L1_END
    cos_factor = 0.5 * (1 + np.cos(np.pi * epoch / total))
    return LAMBDA_L1_END + (LAMBDA_L1_START - LAMBDA_L1_END) * cos_factor

# Helper : Stage 1 forward through dual-path
def _stage1_forward(batch):
    """Returns (mu_A, mu_total, mu_B, gate, baseline_log, target_residual)."""
    lr_data = batch['lr'].to(DEVICE)
    h_init  = encoder.init_state(batch['hetero']).to(DEVICE)
    drivers = [lr_data[t] for t in range(lr_data.shape[0])]
    seq_out = rcn_runner.run(h_init, drivers, reconstruction_sources=None)
    mu_A = regression_head(seq_out.states[-1])
    if mu_A.dim() == 3: mu_A = mu_A.unsqueeze(0)
    lr_grid = batch_lr_grid_last(batch, builder=builder, device=DEVICE)
    lr_safe = torch.nan_to_num(lr_grid, nan=0.0)
    mu_total, mu_B, gate = dual_path(lr_safe, mu_A)
    mu_total = torch.nan_to_num(mu_total, nan=0.0)
    bl = batch['baseline'][-1].to(DEVICE)
    if bl.dim() == mu_total.dim() - 1: bl = bl.unsqueeze(0)
    bl = torch.nan_to_num(bl, nan=0.0)
    tgt = batch['residual'][-1].to(DEVICE)
    if tgt.dim() == 3: tgt = tgt.unsqueeze(0)
    return mu_A, mu_total, mu_B, gate, bl, tgt

# Training history
stage1_history = []

print(f'[Cell 8] Stage 1 training : {STAGE1_EPOCHS} epochs (SMOKE_MODE={SMOKE_MODE})')
for ep in range(1, STAGE1_EPOCHS + 1):
    _t0 = time.time()
    # Mode train
    for m in [encoder, rcn_cell, regression_head, dual_path]:
        m.train()

    lam_l1 = _l1_schedule(ep - 1, STAGE1_EPOCHS)
    phys_active = (ep >= PHYS_LOSS_WARMUP_EPOCH)

    epoch_losses = {'mse': [], 'dag': [], 'l1': [], 'r10mm': [], 'rx1day': [], 'cdd': [], 'cc': []}
    _n_batches_done = 0
    for sample in train_dataset:
        batch = convert_sample_to_batch(sample, builder, DEVICE)
        optimizer_s1.zero_grad(set_to_none=True)

        mu_A, mu_total, mu_B, gate, baseline_log, target_residual = _stage1_forward(batch)

        # Core MSE on residual (HR - baseline)
        valid = torch.isfinite(target_residual)
        diff = (mu_total - target_residual) ** 2
        diff_masked = torch.where(valid, diff, torch.zeros_like(diff))
        loss_mse = diff_masked.sum() / valid.sum().clamp_min(1)

        # DAG prior + L1 sparsity (existing pattern in repo)
        _rcn_core = rcn_cell._orig_mod if hasattr(rcn_cell, '_orig_mod') else rcn_cell
        if hasattr(_rcn_core, 'A_dag'):
            A_dag = _rcn_core.A_dag
            loss_dag = LAMBDA_DAG_PRIOR * ((A_dag * A_dag).sum())  # placeholder prior penalty
            loss_l1  = lam_l1 * A_dag.abs().sum()
        else:
            loss_dag = torch.tensor(0.0, device=DEVICE)
            loss_l1  = torch.tensor(0.0, device=DEVICE)

        loss_total = loss_mse + loss_dag + loss_l1

        # Physical losses (after warmup)
        loss_r10  = torch.tensor(0.0, device=DEVICE)
        loss_rx1d = torch.tensor(0.0, device=DEVICE)
        loss_cdd  = torch.tensor(0.0, device=DEVICE)
        loss_cc   = torch.tensor(0.0, device=DEVICE)
        if phys_active:
            # Full HR pred in log1p space
            full_pred = baseline_log + mu_total   # residual reconstruction (no delta yet at Stage 1)
            full_tgt  = baseline_log + target_residual
            loss_r10  = LAMBDA_R10MM  * loss_R10mm(full_pred, full_tgt, valid_mask=valid.float())
            loss_rx1d = LAMBDA_RX1DAY * loss_Rx1day(full_pred, full_tgt)
            loss_cdd  = LAMBDA_CDD    * loss_CDD(full_pred, full_tgt)
            loss_total = loss_total + loss_r10 + loss_rx1d + loss_cdd
            # CC loss : skip if T_850 not in batch (would need to thread it through)
            # For now, document as TODO -- requires lr_grid to carry T_850 explicitly

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(stage1_params, 1.0)
        optimizer_s1.step()

        epoch_losses['mse'].append(float(loss_mse.detach()))
        epoch_losses['dag'].append(float(loss_dag.detach()))
        epoch_losses['l1'].append(float(loss_l1.detach()))
        epoch_losses['r10mm'].append(float(loss_r10.detach()))
        epoch_losses['rx1day'].append(float(loss_rx1d.detach()))
        epoch_losses['cdd'].append(float(loss_cdd.detach()))
        epoch_losses['cc'].append(float(loss_cc.detach()))
        _n_batches_done += 1
        if SMOKE_MODE and _n_batches_done >= 30:
            break   # SMOKE : process only 30 train batches per epoch

    ep_time = time.time() - _t0
    avg = {k: float(np.mean(v)) if v else 0.0 for k, v in epoch_losses.items()}
    print(f'[ep{ep}/{STAGE1_EPOCHS}] mse={avg["mse"]:.4f} dag={avg["dag"]:.4f} l1={avg["l1"]:.4f} '
          f'r10={avg["r10mm"]:.4f} rx1d={avg["rx1day"]:.4f} cdd={avg["cdd"]:.4f} '
          f'(phys={phys_active}) time={ep_time:.0f}s n_batches={_n_batches_done}')

    stage1_history.append({
        'epoch': ep, 'epoch_time_s': ep_time, 'losses': avg,
        'lambda_l1': lam_l1, 'phys_active': phys_active,
        'n_batches': _n_batches_done,
    })

    # Save Stage 1 checkpoint each epoch
    payload_s1 = {
        'epoch': ep,
        'encoder_state_dict': encoder.state_dict(),
        'rcn_cell_state_dict': rcn_cell.state_dict(),
        'regression_head_state_dict': regression_head.state_dict(),
        'dual_path_state_dict': dual_path.state_dict(),
        'alpha_logit': alpha_logit.detach().cpu(),
        'optimizer_state_dict': optimizer_s1.state_dict(),
        'history': stage1_history,
        'pre_reg': PRE_REG_RECORD,
    }
    torch.save(payload_s1, CKPT_STAGE1_LAST)

    if SMOKE_MODE and ep >= STAGE1_EPOCHS:
        break

# Freeze Stage 1 after training
for m in [encoder, rcn_cell, regression_head, dual_path]:
    for p in m.parameters(): p.requires_grad_(False)
    m.eval()

# Print final A_dag stats (Q_phys interpretability)
_rcn_core = rcn_cell._orig_mod if hasattr(rcn_cell, '_orig_mod') else rcn_cell
if hasattr(_rcn_core, 'A_dag'):
    _A = _rcn_core.A_dag.detach()
    print(f'[Cell 8] Final A_dag : shape={tuple(_A.shape)} norm={_A.norm():.4f} '
          f'asymmetry={(_A - _A.T).abs().mean():.4f}')

print(f'[Cell 8] Stage 1 training complete. Checkpoint : {CKPT_STAGE1_LAST}')
