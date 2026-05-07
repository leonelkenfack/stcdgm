"""
Diagnostic cellule : breakdown du temps par batch Stage 2.

À coller dans une cellule isolée du notebook Colab après cell 47
(PERSIST_HELPERS) et après que le training loop soit interrompu.
Mesure le temps de chaque phase sur 4 batches successifs.

Output → on saura si le bottleneck est I/O, Stage 1 forward,
diffusion forward, ou backward.
"""

DIAG_CELL = '''
# >>> BS32_DIAG_TIMING — à coller dans une cellule isolée
# Profile le breakdown temps par batch Stage 2 sur 4 batches.
import time
import torch

# Vérifier que tout est prêt (encoder, rcn_runner, regression_head, diffusion, val_dataloader, builder, DEVICE)
assert all(n in dir() for n in ("encoder", "rcn_runner", "regression_head", "diffusion", "val_dataloader", "builder", "DEVICE"))

encoder.eval(); rcn_runner.cell.eval(); regression_head.eval()
diffusion.train()  # pour activer le full forward+backward path

print(f"DEVICE = {DEVICE}")
print(f"DATA_ROOT current = {globals().get('DATA_ROOT', 'undefined')}")
print(f"data.lr_path = {CONFIG.data.lr_path}")
print(f"data.hr_path = {CONFIG.data.hr_path}")
print(f"BATCH_SIZE = {int(CONFIG.training.get('batch_size', 1))}")
print()

t_io = []
t_stage1 = []
t_diff_fwd = []
t_diff_bwd = []
n_micros_per_batch = []

# Une seule passe sur 4 logical batches
n_batches = 0
t_outer_start = time.time()

for raw_batch in val_dataloader:
    if n_batches >= 4:
        break

    # --- I/O + iterate_batches CPU work ---
    t0 = time.time()
    if not isinstance(raw_batch, list):
        raw_batch = [raw_batch]
    converted = [convert_sample_to_batch(s, builder, DEVICE) for s in raw_batch]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_io.append(time.time() - t0)
    n_micros_per_batch.append(len(converted))

    # --- Per-micro loop : Stage 1 + Diffusion ---
    s1_total = 0.0
    fwd_total = 0.0
    bwd_total = 0.0

    for micro in converted:
        lr_data = micro["lr"].to(DEVICE)
        target = micro["residual"][-1].to(DEVICE)
        if target.dim() == 3:
            target = target.unsqueeze(0)
        baseline_t = micro["baseline"][-1].to(DEVICE) if micro.get("baseline") is not None else None
        if baseline_t is not None and baseline_t.dim() == 3:
            baseline_t = baseline_t.unsqueeze(0)

        # Stage 1 fwd
        t0 = time.time()
        with torch.no_grad():
            H_init = encoder.init_state(micro["hetero"]).to(DEVICE)
            drivers = [lr_data[t] for t in range(lr_data.shape[0])]
            seq_out = rcn_runner.run(H_init, drivers, reconstruction_sources=None)
            H_T = seq_out.states[-1]
            mu_HR = regression_head(H_T)
            if mu_HR.shape != target.shape:
                mu_HR = torch.nn.functional.interpolate(mu_HR, size=target.shape[-2:], mode="bilinear", align_corners=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        s1_total += time.time() - t0

        # Sanitize
        baseline_log = baseline_t if baseline_t is not None else torch.zeros_like(target)
        mu_HR = torch.nan_to_num(mu_HR, nan=0.0, posinf=0.0, neginf=0.0)
        baseline_log = torch.nan_to_num(baseline_log, nan=0.0, posinf=0.0, neginf=0.0)
        delta_target = target - mu_HR

        # Diffusion forward
        t0 = time.time()
        with torch.amp.autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu", dtype=torch.bfloat16):
            loss = diffusion.compute_loss_edm(
                target=delta_target,
                conditioning=None, conditioning_spatial=None,
                mu_HR=mu_HR, baseline_log=baseline_log,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fwd_total += time.time() - t0

        # Diffusion backward
        t0 = time.time()
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        bwd_total += time.time() - t0

    # Reset grads
    for p in diffusion.parameters():
        if p.grad is not None:
            p.grad = None

    t_stage1.append(s1_total)
    t_diff_fwd.append(fwd_total)
    t_diff_bwd.append(bwd_total)

    print(f"  batch {n_batches+1}: I/O={t_io[-1]:.2f}s  S1={s1_total:.2f}s  fwd={fwd_total:.2f}s  bwd={bwd_total:.2f}s  total={t_io[-1]+s1_total+fwd_total+bwd_total:.2f}s ({len(converted)} micros)")
    n_batches += 1

import statistics
print()
print("=" * 70)
print("BREAKDOWN MOYEN PAR LOGICAL BATCH")
print("=" * 70)
mean_io = statistics.mean(t_io)
mean_s1 = statistics.mean(t_stage1)
mean_fwd = statistics.mean(t_diff_fwd)
mean_bwd = statistics.mean(t_diff_bwd)
total = mean_io + mean_s1 + mean_fwd + mean_bwd
n_micros = statistics.mean(n_micros_per_batch)

print(f"  I/O + convert_sample_to_batch  : {mean_io:6.2f}s ({mean_io/total*100:5.1f}%)")
print(f"  Stage 1 forward (×{n_micros:.0f} micros)    : {mean_s1:6.2f}s ({mean_s1/total*100:5.1f}%)")
print(f"  Diffusion forward (×{n_micros:.0f} bs=1)    : {mean_fwd:6.2f}s ({mean_fwd/total*100:5.1f}%)")
print(f"  Diffusion backward (×{n_micros:.0f} bs=1)   : {mean_bwd:6.2f}s ({mean_bwd/total*100:5.1f}%)")
print(f"  TOTAL                          : {total:6.2f}s ({total/n_micros*1000:5.0f} ms/sample)")
print()
print("DIAGNOSTIC :")
maxc = max([("I/O", mean_io), ("Stage1", mean_s1), ("DiffFwd", mean_fwd), ("DiffBwd", mean_bwd)], key=lambda x: x[1])
print(f"  Bottleneck dominant : {maxc[0]} ({maxc[1]:.1f}s = {maxc[1]/total*100:.0f}%)")

if maxc[0] == "I/O":
    print("  → BS32 (Drive→SSD) doit être activé ; ré-exécuter cell 16.")
elif maxc[0] == "Stage1":
    print("  → BS32b (pre-cache mu_HR) sera la cible.")
elif maxc[0] in ("DiffFwd", "DiffBwd"):
    print("  → BS32a (batchage tensoriel diffusion) sera la cible.")
'''


if __name__ == "__main__":
    print("Copier-coller le contenu suivant dans une cellule Colab :")
    print()
    print(DIAG_CELL)
