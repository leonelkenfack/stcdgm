# >>> Cell 9 : etage 2 - diffusion EDM sur le residu
from torch.utils.data import TensorDataset, DataLoader
from st_cdgm.models.diffusion_decoder import CausalDiffusionDecoder
from st_cdgm.models.edm_preconditioner import EDMConfig
from st_cdgm.training.two_stage import train_epoch_stage2_cached

torch.manual_seed(SEED)
S2 = CONFIG.two_stage.stage2
# sigma_data ne se passe PAS au constructeur : il vit dans l'EDMConfig. On part
# du bloc `diffusion.edm` du YAML et on y injecte la valeur CALIBREE en Cell 8,
# sinon le preconditionneur travaillerait avec la valeur figee du fichier
# (0,1) alors que l'echelle reelle du residu est mesuree a l'execution.
_edm = dict(CONFIG.diffusion.get("edm", {}))
_edm["sigma_data"] = SIGMA_DATA
diffusion = CausalDiffusionDecoder(
    in_channels=1,
    conditioning_dim=int(CONFIG.diffusion.conditioning_dim),
    height=int(CONFIG.diffusion.height), width=int(CONFIG.diffusion.width),
    unet_kwargs=OmegaConf.to_container(CONFIG.diffusion.unet_kwargs, resolve=True),
    scheduler_type=str(CONFIG.diffusion.scheduler_type),
    edm_config=EDMConfig.from_yaml_dict(_edm),
    # causal_concat=True est OBLIGATOIRE : sans lui l'UNet est bati avec 1 seul
    # canal d'entree et compute_loss_edm(mu_HR=..., baseline_log=...) leve.
    causal_concat=True,
    conv_padding_mode=str(CONFIG.diffusion.get("conv_padding_mode", "zeros")),
    anti_checkerboard=bool(CONFIG.diffusion.get("anti_checkerboard", False)),
    # Requis a batch=64 : recompute les activations au backward, ~30 % plus lent
    # et ~50 % de VRAM en moins. Sans lui, OOM sur T4 16 Go.
    use_gradient_checkpointing=True,
).to(DEVICE)
print(f"parametres etage 2 : {sum(p.numel() for p in diffusion.parameters()):,}")

cached = TensorDataset(cache["mu_HR"], cache["baseline_log"],
                       cache["delta_target"], cache["valid_mask"])
BS = int(CONFIG.training.batch_size)
# drop_last=True : si le cache contient MOINS que batch_size, le DataLoader rend
# zero batch et train_epoch_stage2_cached tourne a vide en renvoyant n_batches=0
# sans lever. Observe sur un smoke : l'etage 2 n'avait rien entraine et rien ne
# le disait. On refuse plutot que d'entrainer dans le vide.
if len(cached) < BS:
    raise ValueError(
        f"cache de {len(cached)} echantillons pour un batch de {BS} : avec "
        f"drop_last=True le loader serait VIDE et l'etage 2 ne s'entrainerait "
        f"pas. Reduire CONFIG.training.batch_size ou augmenter le jeu.")

def cached_loader():
    for mu, bl, dt, vm in DataLoader(cached, batch_size=BS, shuffle=True, drop_last=True):
        yield {"mu_HR": mu.to(DEVICE), "baseline_log": bl.to(DEVICE),
               "delta_target": dt.to(DEVICE), "valid_mask": vm.to(DEVICE)}

opt_s2 = torch.optim.AdamW(diffusion.parameters(), lr=float(S2.lr),
                           betas=(0.9, 0.99))
print(f"etage 2 : lr={float(S2.lr):.1e} | sigma_data={SIGMA_DATA:.5f}")

_LAST2 = CKPT_DIR / "stage2_last.pth"
hist2, _start2 = [], 0
if _LAST2.exists():
    _r2 = torch.load(_LAST2, map_location=DEVICE, weights_only=False)
    diffusion.load_state_dict(_r2["diffusion_state_dict"])
    if "optimizer_state_dict" in _r2:
        opt_s2.load_state_dict(_r2["optimizer_state_dict"])
    hist2, _start2 = _r2.get("history", []), _r2["epoch"] + 1
    print(f"REPRISE etage 2 a l'epoque {_start2 + 1}/{EPOCHS_S2}")

for ep in range(_start2, EPOCHS_S2):
    t_ep = time.time()
    m2 = train_epoch_stage2_cached(
        diffusion_decoder=diffusion, optimizer=opt_s2,
        cached_dataloader=cached_loader(), device=DEVICE,
        use_amp=(DEVICE.type == "cuda"), gradient_clipping=1.0,
        # conditioning dropout : entraine la branche inconditionnelle, requise
        # pour cfg_scale > 1 a l'inference (CorrDiff, Mardani 2024 sec 4.2).
        conditioning_dropout_prob=float(
            CONFIG.diffusion.get("conditioning_dropout_prob", 0.13)),
        verbose=(ep == 0))
    m2 = dict(m2); m2["epoch"] = ep; m2["seconds"] = round(time.time() - t_ep, 1)
    hist2.append(m2)
    print(f"[S2 {ep + 1:2d}/{EPOCHS_S2}] {m2}")
    if int(m2.get("n_batches", 0)) == 0:
        raise RuntimeError("epoque etage 2 sans aucun batch : rien n'a ete "
                           "entraine, verifier la taille du cache.")
    torch.save({"epoch": ep, "diffusion_state_dict": diffusion.state_dict(),
                "optimizer_state_dict": opt_s2.state_dict(),
                "sigma_data": SIGMA_DATA, "history": hist2}, _LAST2)

json.dump(hist2, open(RESULTS_DIR / "v8_stage2_history.json", "w"),
          indent=2, default=float)
