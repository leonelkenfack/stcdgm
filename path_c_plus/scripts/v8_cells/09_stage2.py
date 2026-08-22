# >>> Cell 9 : etage 2 - diffusion EDM sur le residu
from torch.utils.data import TensorDataset, DataLoader
from st_cdgm.models.diffusion_decoder import CausalDiffusionDecoder
from st_cdgm.training.two_stage import train_epoch_stage2_cached

torch.manual_seed(SEED)
diffusion = CausalDiffusionDecoder(
    height=HR_SHAPE[0], width=HR_SHAPE[1],
    conditioning_channels=int(CONFIG.diffusion.get("conditioning_channels", 2)),
    sigma_data=SIGMA_DATA,
).to(DEVICE)
print(f"parametres etage 2 : {sum(p.numel() for p in diffusion.parameters()):,}")

cached = TensorDataset(cache["mu_HR"], cache["baseline_log"],
                       cache["delta_target"], cache["valid_mask"])
BS = int(CONFIG.training.get("batch_size", 8))

def cached_loader():
    for mu, bl, dt, vm in DataLoader(cached, batch_size=BS, shuffle=True, drop_last=True):
        yield {"mu_HR": mu.to(DEVICE), "baseline_log": bl.to(DEVICE),
               "delta_target": dt.to(DEVICE), "valid_mask": vm.to(DEVICE)}

opt_s2 = torch.optim.AdamW(
    diffusion.parameters(),
    lr=float(CONFIG.training.get("learning_rate_stage2", CONFIG.training.learning_rate)),
    betas=(0.9, 0.99))

hist2 = []
for ep in range(EPOCHS_S2):
    t_ep = time.time()
    m2 = train_epoch_stage2_cached(
        diffusion_decoder=diffusion, optimizer=opt_s2,
        cached_dataloader=cached_loader(), device=DEVICE,
        use_amp=(DEVICE.type == "cuda"), gradient_clipping=1.0,
        # conditioning dropout : entraine la branche inconditionnelle, requise
        # pour cfg_scale > 1 a l'inference (CorrDiff, Mardani 2024 sec 4.2).
        conditioning_dropout_prob=0.13,
        verbose=(ep == 0))
    m2 = dict(m2); m2["epoch"] = ep; m2["seconds"] = round(time.time() - t_ep, 1)
    hist2.append(m2)
    print(f"[S2 {ep + 1:2d}/{EPOCHS_S2}] {m2}")
    torch.save({"epoch": ep, "diffusion_state_dict": diffusion.state_dict(),
                "sigma_data": SIGMA_DATA}, CKPT_DIR / "stage2_last.pth")

json.dump(hist2, open(RESULTS_DIR / "v8_stage2_history.json", "w"),
          indent=2, default=float)
