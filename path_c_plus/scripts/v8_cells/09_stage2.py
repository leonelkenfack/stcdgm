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
# CONFIG.training.batch_size = 64 vise l'A100 80 Go : le YAML de reference dit
# lui-meme que 64 sur cet UNet "OOM meme sur A100" sans recompute d'activations.
# Sur T4 16 Go on descend a 32. Cela ne change PAS le budget d'entrainement :
# une epoque reste une passe sur le cache, donc le nombre de tirages est le
# meme — seul le nombre de pas double.
_VRAM = (torch.cuda.get_device_properties(0).total_memory / 2 ** 30
         if DEVICE.type == "cuda" else 0.0)
BS = int(os.environ.get("V8_BS_S2", 0)) or (64 if _VRAM > 40 else 32)
print(f"batch etage 2 : {BS} ({_VRAM:.0f} GiB de VRAM)")
# drop_last=True : si le cache contient MOINS que batch_size, le DataLoader rend
# zero batch et train_epoch_stage2_cached tourne a vide en renvoyant n_batches=0
# sans lever. Observe sur un smoke : l'etage 2 n'avait rien entraine et rien ne
# le disait. On refuse plutot que d'entrainer dans le vide.
if len(cached) < BS:
    raise ValueError(
        f"cache de {len(cached)} echantillons pour un batch de {BS} : avec "
        f"drop_last=True le loader serait VIDE et l'etage 2 ne s'entrainerait "
        f"pas. Baisser V8_BS_S2 ou augmenter le jeu.")

def cached_loader():
    for mu, bl, dt, vm in DataLoader(cached, batch_size=BS, shuffle=True, drop_last=True):
        yield {"mu_HR": mu.to(DEVICE), "baseline_log": bl.to(DEVICE),
               "delta_target": dt.to(DEVICE), "valid_mask": vm.to(DEVICE)}


# Nombre d'epoques DERIVE du budget en tirages. Une epoque = une passe sur le
# cache, donc EPOCHS = tirages_vises / taille_du_cache — independamment du
# batch et du stride. C'est cette derivation qui empeche de refaire l'erreur
# d'origine : une constante d'epoques ne veut rien dire tant que la taille du
# cache peut bouger sous elle.
EPOCHS_S2 = int(os.environ.get("V8_EPOCHS_S2", 0)) or max(
    1, -(-TARGET_DRAWS_S2 // len(cached)))
print(f"budget etage 2 : {TARGET_DRAWS_S2:,} tirages / {len(cached)} fenetres "
      f"= {EPOCHS_S2} epoques de {len(cached) // BS} pas")

opt_s2 = torch.optim.AdamW(diffusion.parameters(), lr=float(S2.lr),
                           betas=(0.9, 0.99))

# EMA des poids (V5 Track B1, convention Karras EDM2). V5 et CorrDiff evaluent
# tous les deux la moyenne mobile, pas les poids du dernier pas ; s'en passer
# ici desavantagerait V8 sur un facteur qui n'a rien d'architectural.
# decay=0.999 et non 0.9999 : le shadow n'est PAS mis a jour pendant le warmup,
# il repart donc de l'init ALEATOIRE, dont le poids residuel apres n pas vaut
# decay^n. A 0,9999 sur 20 000 pas il resterait ~13 % d'init aleatoire dans les
# poids evalues ; a 0,999 il n'en reste rien des le dixieme de ce budget.
# Horizon d'averaging : ~1 000 pas.
import copy
ema = copy.deepcopy(diffusion)
EMA_DECAY = 0.999

# Programme de pas : warmup lineaire puis cosinus (Karras EDM §5, comme V5).
# Sur ~21 000 pas un lr constant laisse le modele osciller autour du minimum en
# fin de course ; le warmup evite que les premiers pas, sur des poids
# aleatoires, ne detruisent l'echelle de l'UNet.
_WARM = max(1, min(5, EPOCHS_S2 // 50))
sched_s2 = torch.optim.lr_scheduler.SequentialLR(
    opt_s2,
    [torch.optim.lr_scheduler.LinearLR(opt_s2, start_factor=0.1,
                                       total_iters=_WARM),
     torch.optim.lr_scheduler.CosineAnnealingLR(
         opt_s2, T_max=max(1, EPOCHS_S2 - _WARM), eta_min=float(S2.lr) / 50)],
    milestones=[_WARM])
print(f"etage 2 : lr={float(S2.lr):.1e} | sigma_data={SIGMA_DATA:.5f} | "
      f"EMA decay={EMA_DECAY} | warmup {_WARM} ep puis cosinus")

_LAST2 = CKPT_DIR / "stage2_last.pth"
_ARCH_S2 = list(CONFIG.diffusion.unet_kwargs.block_out_channels)


def _incompatible(r):
    """Pourquoi ce checkpoint ne peut PAS servir a reprendre — ou None.

    Verifie AVANT tout load_state_dict : `strict=True` copie les tenseurs qui
    correspondent avant de lever sur les autres, et laisserait donc le modele
    a moitie ecrase par les poids d'un run etranger.
    """
    _a = r.get("unet_block_out_channels")
    if _a is None:                       # checkpoint anterieur a ce champ
        _w = r.get("diffusion_state_dict", {}).get("unet.conv_in.weight")
        _a = [int(_w.shape[0])] if _w is not None else None
    if _a is not None and list(_a)[:1] != _ARCH_S2[:1]:
        return f"UNet {list(_a)} au lieu de {_ARCH_S2}"
    # sigma_data alimente c_skip/c_out/c_in du preconditionneur EDM : reprendre
    # avec une autre valeur entrainerait un modele autrement preconditionne,
    # sans que rien ne le signale.
    _s = r.get("sigma_data")
    if _s is not None and abs(float(_s) - SIGMA_DATA) > 0.05 * SIGMA_DATA:
        return f"sigma_data {float(_s):.5f} au lieu de {SIGMA_DATA:.5f}"
    return None


hist2, _start2, _r2 = [], 0, None
if _LAST2.exists():
    _r2 = torch.load(_LAST2, map_location=DEVICE, weights_only=False)
    _pourquoi = _incompatible(_r2)
    if _pourquoi:
        # Le checkpoint vient d'une AUTRE configuration : le charger leverait
        # un mur de "size mismatch". On l'ecarte sans le detruire et on repart
        # de zero — c'est la seule chose correcte, l'entrainement precedent ne
        # decrit pas ce modele-ci.
        _vieux = _LAST2.with_name(f"stage2_last.perime_{int(time.time())}.pth")
        _LAST2.rename(_vieux)
        print(f"CHECKPOINT ECARTE : {_pourquoi}")
        print(f"  conserve sous {_vieux.name} ; l'etage 2 repart de zero.")
        _r2 = None
if _r2 is not None:
    diffusion.load_state_dict(_r2["diffusion_state_dict"])
    # ORDRE NON NEGOCIABLE : l'optimiseur AVANT le scheduler. `SequentialLR`
    # remet le lr a sa valeur de warmup a la construction, et son
    # `load_state_dict` ne restaure que des compteurs — c'est le state_dict de
    # l'OPTIMISEUR qui porte le lr courant. Inverser les deux lignes ferait
    # reprendre une epoque a 2e-5 au lieu de 1,3e-4, sans rien signaler.
    if "optimizer_state_dict" in _r2:
        opt_s2.load_state_dict(_r2["optimizer_state_dict"])
    if "ema_state_dict" in _r2:
        ema.load_state_dict(_r2["ema_state_dict"])
    hist2, _start2 = _r2.get("history", []), _r2["epoch"] + 1
    if "scheduler_state_dict" in _r2:
        sched_s2.load_state_dict(_r2["scheduler_state_dict"])
    else:
        # Checkpoint anterieur au programme de pas : le rejouer a vide, sinon
        # la reprise repartirait au lr du debut au lieu du lr courant.
        for _ in range(_start2):
            sched_s2.step()
    print(f"REPRISE etage 2 a l'epoque {_start2 + 1}/{EPOCHS_S2} "
          f"(lr={opt_s2.param_groups[0]['lr']:.2e})")

def _epoque(ep):
    return train_epoch_stage2_cached(
        diffusion_decoder=diffusion, optimizer=opt_s2,
        cached_dataloader=cached_loader(), device=DEVICE,
        use_amp=(DEVICE.type == "cuda"), gradient_clipping=1.0,
        # conditioning dropout : entraine la branche inconditionnelle, requise
        # pour cfg_scale > 1 a l'inference (CorrDiff, Mardani 2024 sec 4.2).
        conditioning_dropout_prob=float(
            CONFIG.diffusion.get("conditioning_dropout_prob", 0.13)),
        ema_model=ema, ema_decay=EMA_DECAY, ema_warmup_steps=0,
        verbose=(ep == _start2))


for ep in range(_start2, EPOCHS_S2):
    t_ep = time.time()
    while True:
        try:
            m2 = _epoque(ep)
            break
        except torch.cuda.OutOfMemoryError:
            # L'UNet de reference tient sur T4 a batch 32 d'apres l'estimation,
            # pas d'apres une mesure. Si elle est fausse, l'OOM tombe apres
            # l'etage 1 — plusieurs heures deja payees. On divise le batch et
            # on refait l'epoque plutot que de perdre le run. Le budget en
            # tirages ne bouge pas ; seul le nombre de pas augmente.
            if BS <= 4:
                raise
            torch.cuda.empty_cache()
            BS //= 2
            print(f"  OOM -> batch reduit a {BS}, epoque refaite")
    m2 = dict(m2); m2["epoch"] = ep; m2["seconds"] = round(time.time() - t_ep, 1)
    m2["batch_size"] = BS
    m2["lr"] = opt_s2.param_groups[0]["lr"]
    sched_s2.step()
    hist2.append(m2)
    print(f"[S2 {ep + 1:2d}/{EPOCHS_S2}] {m2}")
    if int(m2.get("n_batches", 0)) == 0:
        raise RuntimeError("epoque etage 2 sans aucun batch : rien n'a ete "
                           "entraine, verifier la taille du cache.")
    if ep == _start2:
        # Le budget se lit en PAS, et le temps total se mesure a la premiere
        # epoque plutot que de se decouvrir a la huitieme heure.
        _nb = int(m2["n_batches"])
        print(f"  -> {_nb} batches/epoque, {_nb * EPOCHS_S2:,} pas au total, "
              f"~{m2['seconds'] * (EPOCHS_S2 - _start2) / 3600:.1f} h restantes "
              f"(interruptible : la reprise est en place)")
    torch.save({"epoch": ep, "diffusion_state_dict": diffusion.state_dict(),
                "optimizer_state_dict": opt_s2.state_dict(),
                "scheduler_state_dict": sched_s2.state_dict(),
                "ema_state_dict": ema.state_dict(),
                # Signature d'architecture : sans elle, un checkpoint d'un run
                # a l'UNet different se charge (mal) au lieu d'etre ecarte.
                "unet_block_out_channels": _ARCH_S2,
                "sigma_data": SIGMA_DATA, "history": hist2}, _LAST2)

# L'evaluation porte sur les poids EMA, comme V5 et CorrDiff. Les poids vifs
# restent dans le checkpoint : une reprise repart de la vraie trajectoire.
diffusion.load_state_dict(ema.state_dict())
print("poids EMA charges pour l'evaluation")

json.dump(hist2, open(RESULTS_DIR / "v8_stage2_history.json", "w"),
          indent=2, default=float)
