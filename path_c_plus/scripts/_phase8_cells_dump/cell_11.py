# === Cell 11 : Stage 2 training (Min-SNR + tail + Dispersive + cond_drop + multi-EMA) ===
# Inline implementation -- adapts train_epoch_stage2_cached pattern with all the fixes.
from st_cdgm.models.edm_preconditioner import sample_training_sigma
import time

# Optimizer (Stage 2 live + alpha_logit)
stage2_params = list(diff_decoder.parameters()) + [alpha_logit]
optimizer_s2 = torch.optim.AdamW(
    stage2_params,
    lr=STAGE2_LR, betas=(0.9, 0.999), weight_decay=STAGE2_WEIGHT_DECAY,
)

stage2_history = []
start_epoch_s2 = 1

# Resume support
if CKPT_STAGE2_LAST.exists() and not SMOKE_MODE:
    print(f'[Cell 11] RESUME from {CKPT_STAGE2_LAST}')
    _ck = torch.load(CKPT_STAGE2_LAST, map_location=DEVICE, weights_only=False)
    diff_decoder.load_state_dict(_ck['diffusion_state_dict'])
    if 'ema_state_dicts' in _ck:
        for i, ema in enumerate(ema_decoders):
            if i < len(_ck['ema_state_dicts']):
                ema.load_state_dict(_ck['ema_state_dicts'][i])
    if _ck.get('alpha_logit') is not None:
        alpha_logit.data = _ck['alpha_logit'].to(DEVICE)
    if _ck.get('optimizer_state_dict') is not None:
        try: optimizer_s2.load_state_dict(_ck['optimizer_state_dict'])
        except Exception as e: print(f'  optimizer resume failed : {e}')
    start_epoch_s2 = int(_ck.get('epoch', 0)) + 1
    stage2_history = list(_ck.get('history', []))
    print(f'  Resumed at epoch {start_epoch_s2}/{STAGE2_EPOCHS}')

print(f'[Cell 11] Stage 2 training : epochs {start_epoch_s2}..{STAGE2_EPOCHS} (SMOKE={SMOKE_MODE})')

def _ema_update_all(decoders, decays, live_model):
    """Multi-EMA update : in-place mul + add on params, copy on buffers."""
    with torch.no_grad():
        for ema, decay in zip(decoders, decays):
            for p_e, p_l in zip(ema.parameters(), live_model.parameters()):
                p_e.data.mul_(decay).add_(p_l.data, alpha=1.0 - decay)
            for b_e, b_l in zip(ema.buffers(), live_model.buffers()):
                b_e.data.copy_(b_l.data)
            ema._ema_step_counter += 1

for ep in range(start_epoch_s2, STAGE2_EPOCHS + 1):
    _t0 = time.time()
    diff_decoder.train()

    losses_log = {'edm': [], 'disp': [], 'alpha_reg': [], 'total': []}
    n_batches_seen = 0

    for batch in train_cached_dataloader:
        mu_HR = batch['mu_HR'].to(DEVICE, non_blocking=True)
        baseline_log = batch['baseline_log'].to(DEVICE, non_blocking=True)
        delta_target = batch['delta_target'].to(DEVICE, non_blocking=True)

        # Conditioning dropout (CFG compatibility, CorrDiff Nature CEE 2025)
        # When mu_HR is dropped, target also adjusted (Ho & Salimans branch correctness)
        mu_HR_used = mu_HR
        delta_target_used = delta_target
        if COND_DROPOUT_P > 0:
            B = mu_HR.shape[0]
            drop_mask = (torch.rand(B, device=DEVICE) < COND_DROPOUT_P).view(B, 1, 1, 1)
            if drop_mask.any():
                mu_HR_used = torch.where(drop_mask, torch.zeros_like(mu_HR), mu_HR)
                delta_target_used = torch.where(drop_mask, delta_target + mu_HR, delta_target)

        optimizer_s2.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            B = delta_target_used.shape[0]
            sigma = sample_training_sigma(B, P_mean=edm_cfg.P_mean, P_std=edm_cfg.P_std,
                                            device=DEVICE, dtype=delta_target_used.dtype)
            noise = torch.randn_like(delta_target_used)
            y_noisy = delta_target_used + sigma.view(-1, 1, 1, 1) * noise

            # Forward EDM (this triggers the dispersive hook on mid_block)
            D_y = diff_decoder.forward_edm(
                y_noisy, sigma, conditioning=None, conditioning_spatial=None,
                mu_HR=mu_HR_used, baseline_log=baseline_log,
            )

            # ----- Combined loss weight : Karras lambda * Min-SNR gamma -----
            lambda_karras = (sigma**2 + edm_cfg.sigma_data**2) / (sigma * edm_cfg.sigma_data)**2
            min_snr_w = _min_snr_weight(sigma, edm_cfg.sigma_data, MIN_SNR_GAMMA)
            w_total = lambda_karras.view(-1, 1, 1, 1) * min_snr_w

            # ----- Tail weight on extreme pixels -----
            hr_log_recon = delta_target_used + mu_HR_used + baseline_log   # full HR in log1p
            tail_w = _tail_weight(hr_log_recon).detach()
            # Clip combined weight to avoid extreme values (Math : W_max=50)
            w_combined = (w_total * tail_w).clamp(max=50.0)

            sq_err = (D_y - delta_target_used) ** 2
            loss_edm = (w_combined * sq_err).mean()

            # ----- Dispersive Loss on mid-block features -----
            loss_disp = loss_dispersive(dispersive_features.get('mid', None))

            # ----- Alpha regularization -----
            alpha = torch.sigmoid(alpha_logit)
            loss_alpha = loss_alpha_reg(alpha)

            loss_total = loss_edm + DISPERSIVE_LAMBDA * loss_disp + loss_alpha

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(stage2_params, STAGE2_GRADIENT_CLIP)
        optimizer_s2.step()

        # Multi-EMA update
        _ema_update_all(ema_decoders, EMA_DECAYS, diff_decoder)

        losses_log['edm'].append(float(loss_edm.detach()))
        losses_log['disp'].append(float(loss_disp.detach()))
        losses_log['alpha_reg'].append(float(loss_alpha.detach()))
        losses_log['total'].append(float(loss_total.detach()))
        n_batches_seen += 1

        if SMOKE_MODE and n_batches_seen >= 10:
            break

    ep_time = time.time() - _t0
    avg = {k: float(np.mean(v)) if v else 0.0 for k, v in losses_log.items()}
    cur_alpha = float(torch.sigmoid(alpha_logit).detach())
    print(f'[ep{ep}/{STAGE2_EPOCHS}] edm={avg["edm"]:.4f} disp={avg["disp"]:.4f} '
          f'alpha_reg={avg["alpha_reg"]:.4f} total={avg["total"]:.4f} '
          f'alpha={cur_alpha:.4f} time={ep_time:.0f}s n_batches={n_batches_seen}')

    stage2_history.append({
        'epoch': ep, 'epoch_time_s': ep_time, 'losses': avg,
        'alpha': cur_alpha, 'n_batches': n_batches_seen,
    })

    # Save Stage 2 checkpoint each epoch
    payload_s2 = {
        'epoch': ep,
        'diffusion_state_dict': diff_decoder.state_dict(),
        'ema_state_dicts': [ema.state_dict() for ema in ema_decoders],
        'ema_decays': EMA_DECAYS,
        'ema_step_counters': [ema._ema_step_counter for ema in ema_decoders],
        'alpha_logit': alpha_logit.detach().cpu(),
        'optimizer_state_dict': optimizer_s2.state_dict(),
        'history': stage2_history,
        'sigma_data': float(SIGMA_DATA_NEW),
        'pre_reg': PRE_REG_RECORD,
    }
    torch.save(payload_s2, CKPT_STAGE2_LAST)

    if SMOKE_MODE and ep >= STAGE2_EPOCHS:
        break

# Detach hook
if _disp_handle is not None:
    _disp_handle.remove()

# Set all to eval at the end
diff_decoder.eval()
for ema in ema_decoders:
    ema.eval()

print(f'[Cell 11] Stage 2 training complete')
print(f'[Cell 11] Final alpha = {float(torch.sigmoid(alpha_logit)):.4f}')
print(f'[Cell 11] Final EMA step counters : {[ema._ema_step_counter for ema in ema_decoders]}')
print(f'[Cell 11] Checkpoint : {CKPT_STAGE2_LAST}')
