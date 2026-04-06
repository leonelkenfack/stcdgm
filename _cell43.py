# Entraînement strict: split train/val + early stopping + scheduler + DataLoader robuste
import copy
import math
import torch.nn.functional as F
from torch.utils.data import IterableDataset, random_split

history = {k: [] for k in ["loss", "loss_gen", "loss_rec", "loss_dag", "val_loss", "lr"]}

print("🏋️ Début entraînement strict (train + validation à chaque epoch)...")
print("=" * 80)

# Split train/val déterministe (si dataset indexable)
seed = int(CONFIG.training.get("seed", 42))
val_fraction = float(CONFIG.training.get("val_fraction", 0.1))
val_fraction = min(max(val_fraction, 0.05), 0.5)

try:
    total_samples = len(dataset)
except TypeError:
    total_samples = None

if total_samples is None:
    # ResDiffIterableDataset n'est pas indexable -> pas de random_split possible.
    train_dataset = dataset
    val_dataset = None
    train_size = None
    val_size = None
    print("⚠️ Dataset de type IterableDataset détecté: split train/val désactivé.")
    print("   Validation, scheduler (sur val_loss) et early stopping sont ignorés.")
else:
    if total_samples < 2:
        raise RuntimeError("Dataset trop petit pour un split train/val (len < 2).")

    val_size = max(1, int(total_samples * val_fraction))
    if val_size >= total_samples:
        val_size = total_samples - 1
    train_size = total_samples - val_size

    split_generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split_generator)

    print(f"📚 Dataset split: train={train_size}, val={val_size}, total={total_samples}")

# DataLoader config basée sur CONFIG et compatible CPU/GPU
BATCH_SIZE = int(CONFIG.training.get("batch_size", 1))
NUM_WORKERS = int(CONFIG.training.get("num_workers", 0))
# Bon point de départ sur CPU : 4 (évite la saturation avec OMP). Augmentez avec précaution.
if NUM_WORKERS < 0:
    NUM_WORKERS = 0
PIN_MEMORY = bool(DEVICE.type == "cuda")  # False sur CPU (inutile sans GPU)

if NUM_WORKERS == 0:
    PREFETCH_FACTOR = None
    PERSISTENT_WORKERS = False
else:
    PREFETCH_FACTOR = int(CONFIG.training.get("prefetch_factor", 2))
    PERSISTENT_WORKERS = bool(CONFIG.training.get("persistent_workers", True))

print("⚙️  DataLoader Configuration:")
print(f"   - Batch size: {BATCH_SIZE}")
print(f"   - Num workers: {NUM_WORKERS}")
print(f"   - Prefetch factor: {PREFETCH_FACTOR}")
print(f"   - Persistent workers: {PERSISTENT_WORKERS}")
print(f"   - Effective batch: {BATCH_SIZE * len(CONFIG.training.multi_gpu.get('gpus', [0]))} (across GPUs)")


def build_loader(ds, shuffle):
    kwargs = {
        "dataset": ds,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "persistent_workers": PERSISTENT_WORKERS,
        "collate_fn": lambda x: x,
    }
    # Important: PyTorch interdit shuffle/sampler avec IterableDataset.
    if not isinstance(ds, IterableDataset):
        kwargs["shuffle"] = shuffle
    if PREFETCH_FACTOR is not None:
        kwargs["prefetch_factor"] = PREFETCH_FACTOR
    return DataLoader(**kwargs)


train_dataloader = build_loader(train_dataset, shuffle=True)
val_dataloader = build_loader(val_dataset, shuffle=False) if val_dataset is not None else None


def _extract_state_dict(model):
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def _load_state_dict(model, state_dict):
    base = model.module if hasattr(model, "module") else model
    base.load_state_dict(state_dict)


@torch.no_grad()
def compute_validation_loss(dataloader):
    if dataloader is None:
        return float("nan")

    encoder.eval()
    rcn_runner.cell.eval()
    diffusion.eval()

    losses = []
    for converted_batches in iterate_batches(dataloader, builder, DEVICE):
        for batch in converted_batches:
            lr_data = batch["lr"].to(DEVICE)
            hetero_data = batch["hetero"]
            target = batch["residual"][-1].to(DEVICE)

            H_init = encoder.init_state(hetero_data).to(DEVICE)
            drivers = [lr_data[t] for t in range(lr_data.shape[0])]
            seq_output = rcn_runner.run(H_init, drivers, reconstruction_sources=None)

            conditioning = encoder.project_state_tensor(seq_output.states[-1]).to(DEVICE)
            generated = diffusion.sample(
                conditioning,
                num_steps=int(CONFIG.diffusion.get("val_num_steps", 15)),
                scheduler_type=CONFIG.diffusion.get("scheduler_type", "ddpm"),
                apply_constraints=False,
            )

            pred = generated.residual
            if pred.dim() == 4:
                pred = pred.squeeze(0)

            if pred.shape != target.shape:
                pred = F.interpolate(
                    pred.unsqueeze(0),
                    size=target.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                ).clamp(min=0.0).squeeze(0)

            pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
            losses.append(torch.mean((pred - target) ** 2).item())

    return float(np.mean(losses)) if losses else float("nan")


# Scheduler optionnel piloté par val_loss
scheduler = None
scheduler_cfg = CONFIG.training.get("lr_scheduler", {})
if val_dataloader is not None and scheduler_cfg.get("enabled", False):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_cfg.get("mode", "min"),
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", 3)),
        min_lr=float(scheduler_cfg.get("min_lr", 1e-7)),
    )

# Early stopping piloté par val_loss
es_cfg = CONFIG.training.get("early_stopping", {})
early_enabled = bool(es_cfg.get("enabled", False)) and (val_dataloader is not None)
es_patience = int(es_cfg.get("patience", 7))
es_min_delta = float(es_cfg.get("min_delta", 0.0))
es_restore_best = bool(es_cfg.get("restore_best", True))

best_val_loss = math.inf
best_epoch = 0
no_improve_epochs = 0
stopped_early = False
BEST_MODEL_STATES = None

for epoch in range(int(CONFIG.training.epochs)):
    if epoch % 10 == 0:
        print_resource_usage()

    encoder.train()
    rcn_runner.cell.train()
    diffusion.train()

    train_metrics = train_epoch(
        encoder=encoder,
        rcn_runner=rcn_runner,
        diffusion_decoder=diffusion,
        optimizer=optimizer,
        data_loader=iterate_batches(train_dataloader, builder, DEVICE),
        lambda_gen=CONFIG.loss.lambda_gen,
        beta_rec=CONFIG.loss.beta_rec,
        gamma_dag=CONFIG.loss.gamma_dag,
        conditioning_fn=None,
        device=DEVICE,
        use_amp=CONFIG.training.get("use_amp", True),
        gradient_clipping=CONFIG.training.gradient_clipping,
        log_interval=CONFIG.training.log_every,
        dag_method=CONFIG.loss.get("dag_method", "dagma"),
        dagma_s=CONFIG.loss.get("dagma_s", 1.0),
        reconstruction_loss_type=CONFIG.loss.get("reconstruction_loss_type", "mse"),
    )

    val_loss = compute_validation_loss(val_dataloader)

    if val_dataloader is None:
        val_loss = float("nan")

    if scheduler is not None and np.isfinite(val_loss):
        scheduler.step(val_loss)

    current_lr = float(optimizer.param_groups[0]["lr"])

    history["loss"].append(train_metrics["loss"])
    history["loss_gen"].append(train_metrics["loss_gen"])
    history["loss_rec"].append(train_metrics["loss_rec"])
    history["loss_dag"].append(train_metrics["loss_dag"])
    history["val_loss"].append(val_loss)
    history["lr"].append(current_lr)

    improved = np.isfinite(val_loss) and (val_loss < (best_val_loss - es_min_delta))
    if improved:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        no_improve_epochs = 0
        BEST_MODEL_STATES = {
            "encoder_state_dict": copy.deepcopy(_extract_state_dict(encoder)),
            "rcn_cell_state_dict": copy.deepcopy(_extract_state_dict(rcn_cell)),
            "diffusion_state_dict": copy.deepcopy(_extract_state_dict(diffusion)),
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        }
    else:
        no_improve_epochs += 1

    print(
        f"Epoch {epoch + 1}/{CONFIG.training.epochs} | "
        f"TrainLoss: {train_metrics['loss']:.4f} | "
        f"ValLoss: {val_loss:.4f} | "
        f"LR: {current_lr:.6g}"
    )

    if early_enabled and no_improve_epochs >= es_patience:
        stopped_early = True
        print(f"⏹️  Early stopping déclenché à l'epoch {epoch + 1} (patience={es_patience}).")
        break

if es_restore_best and BEST_MODEL_STATES is not None:
    _load_state_dict(encoder, BEST_MODEL_STATES["encoder_state_dict"])
    _load_state_dict(rcn_cell, BEST_MODEL_STATES["rcn_cell_state_dict"])
    _load_state_dict(diffusion, BEST_MODEL_STATES["diffusion_state_dict"])
    optimizer.load_state_dict(BEST_MODEL_STATES["optimizer_state_dict"])

if val_dataloader is None:
    best_epoch = len(history["loss"])
    best_val_loss = float("nan")

TRAINING_RUN = {
    "final_epoch": len(history["loss"]),
    "best_epoch": best_epoch,
    "best_val_loss": best_val_loss,
    "stopped_early": stopped_early,
    "train_size": train_size,
    "val_size": val_size,
    "seed": seed,
}

print("\n" + "=" * 80)
print("✅ Entraînement terminé!")
print(f"   - Best epoch: {TRAINING_RUN['best_epoch']}")
print(f"   - Best val loss: {TRAINING_RUN['best_val_loss']:.6f}")
print(f"   - Stopped early: {TRAINING_RUN['stopped_early']}")

print("\n📊 Final resource usage:")
print_resource_usage()
