# >>> Cell 5 : entrainement etage 1
from torch.optim import AdamW
from st_cdgm.training.training_loop import train_epoch_stage1

def iterate_batches(ds, bld=None, dev=None):
    """Signature (data_loader, builder, device) attendue par les helpers."""
    for s in ds:
        yield [convert_sample_v8(s, bld, dev)]

# bg_head fait partie des parametres optimises : l'oublier laisserait la tete
# a son initialisation et la NLL ne descendrait jamais. L'encodeur, lui, a ete
# materialise en Cell 4 - sinon ses poids lazy seraient absents d'ici.
assert all(sum(1 for _ in m.parameters()) > 0 for m in STAGE1_MODULES),     "un module n'expose aucun parametre - voir la materialisation en Cell 4"
# Les hyperparametres de l'etage 1 vivent sous two_stage.stage1, PAS a la
# racine de `training`. Les lire avec .get() depuis le mauvais niveau
# retomberait en silence sur des defauts codes en dur — le YAML serait ignore
# sans que rien ne le signale.
S1 = CONFIG.two_stage.stage1
opt_s1 = AdamW([p for m in STAGE1_MODULES for p in m.parameters()],
               lr=float(S1.lr), betas=(0.9, 0.99),
               weight_decay=float(S1.get("weight_decay", 0.0)))

# Le YAML declare `scheduler: cosine`, `warmup_epochs: 1` et
# `early_stop_patience: 3` (config/training_config.yaml:545-550). La cellule
# les ignorait tous les trois : lr constant sur 30 epoques, et aucun arret
# quand la validation cesse de progresser. Sur le run precedent elle plafonnait
# des la cinquieme, les vingt-cinq suivantes n'ont fait que memoriser.
_W1 = max(1, int(S1.get("warmup_epochs", 1)))
sched_s1 = torch.optim.lr_scheduler.SequentialLR(
    opt_s1,
    [torch.optim.lr_scheduler.LinearLR(opt_s1, start_factor=0.1, total_iters=_W1),
     torch.optim.lr_scheduler.CosineAnnealingLR(
         opt_s1, T_max=max(1, EPOCHS_S1 - _W1), eta_min=float(S1.lr) / 50)],
    milestones=[_W1])
PATIENCE = int(S1.get("early_stop_patience", 3))
print(f"          warmup {_W1} ep puis cosinus | early stop apres {PATIENCE} "
      f"epoques sans gain de validation")
print(f"etage 1 : lr={float(S1.lr):.1e} lambda_reg={float(S1.lambda_reg)} "
      f"beta_rec={float(S1.beta_rec)} gamma_dag_max={float(S1.gamma_dag_max)} "
      f"lambda_l1={float(S1.lambda_l1)}")

# Une perte de validation. Sans elle, 30 epoques tournent a l'aveugle et le
# "meilleur" checkpoint choisi sur la perte d'ENTRAINEMENT revient a prendre la
# derniere epoque — aucune protection contre le sur-apprentissage, alors que le
# collapse du DAG et l'apprentissage du relief au lieu de la meteo sont des
# modes de defaillance documentes de cette architecture.
from st_cdgm.models.bernoulli_gamma import decode_bg_params, stage1_bg_loss

# UN seul seuil humide, lu aux deux endroits. Le laisser en dur a deux endroits
# est exactement ce qui a fait diverger l'entrainement (0,1) de la validation
# (defaut 1,0) sans que rien ne le signale.
BG_WET_SEUIL = 0.1


@torch.no_grad()
def validation_loss(ds):
    for _m in STAGE1_MODULES:
        _m.eval()
    tot, n = 0.0, 0
    for s_ in ds:
        b = convert_sample_v8(s_, builder, DEVICE)
        t = b["residual"][-1].to(DEVICE)
        if t.dim() == 3:
            t = t.unsqueeze(0)
        bl = b["baseline"][-1].to(DEVICE)
        if bl.dim() == 3:
            bl = bl.unsqueeze(0)
        lr_data = b["lr"].to(DEVICE)
        H0 = encoder.init_state(b["hetero"])
        seq = rcn_runner.run(H0, [lr_data[k] for k in range(lr_data.shape[0])],
                             reconstruction_sources=None)
        H_T = seq.states[-1]
        if bg_head is not None:
            pb, ab, bb = decode_bg_params(regression_head, bg_head, H_T,
                                          target_shape=t.shape[-2:],
                                          baseline_log=bl)
            # wet_threshold EXPLICITE. Le defaut de stage1_bg_loss est 1,0 et
            # l'entrainement passe 0,1 : sans cet argument, la validation
            # notait une AUTRE vraisemblance que celle optimisee — elles
            # divergent sur toute la bande 0,1 a 1 mm/j, precisement celle que
            # A3 existe pour traiter. Le checkpoint "meilleur" etait donc
            # selectionne sur un critere que le modele n'a jamais minimise, et
            # la courbe de validation plate n'etait pas interpretable.
            v, _ = stage1_bg_loss(pb, ab, bb, t, bl, wet_threshold=BG_WET_SEUIL)
        else:
            mu = regression_head(H_T)
            if mu.shape != t.shape:
                mu = torch.nn.functional.interpolate(
                    mu, size=t.shape[-2:], mode="bilinear", align_corners=False)
            m_ = torch.isfinite(t)
            v = ((mu - torch.nan_to_num(t))[m_] ** 2).mean()
        tot += float(v)
        n += 1
    for _m in STAGE1_MODULES:
        _m.train()
    return tot / max(n, 1)


# REPRISE. Un run de plusieurs heures sur Colab SERA interrompu : limite de
# session, deconnexion, onglet ferme. Sans point de reprise il faut tout
# recommencer.
def _incompatible_s1(ck):
    """Pourquoi ce checkpoint ne decrit PAS ce modele-ci — ou None.

    Certains interrupteurs V8 changent les formes et feraient lever
    load_state_dict ; d'autres non, et la reprise serait alors silencieusement
    fausse. On compare donc la configuration, pas seulement les formes. Et on
    verifie AVANT tout chargement : `strict=True` copie les tenseurs qui
    correspondent avant de lever sur les autres, donc l'encodeur et le RCN — que
    le conditionnement de la tete n'a pas fait bouger — seraient deja ecrases
    par les poids d'un run etranger au moment de l'exception.
    """
    if ck.get("node_types") != NODE_TYPES:
        return f"{len(ck.get('node_types') or [])} noeuds au lieu de {len(NODE_TYPES)}"
    if ck.get("v8") != OmegaConf.to_container(V8):
        return "interrupteurs V8 differents"
    if bg_head is not None and "bg_head_state_dict" in ck:
        _w_ck = ck["bg_head_state_dict"].get("proj.weight")
        _w_now = bg_head.state_dict()["proj.weight"]
        if _w_ck is not None and tuple(_w_ck.shape) != tuple(_w_now.shape):
            return (f"tete BG {tuple(_w_ck.shape)} au lieu de "
                    f"{tuple(_w_now.shape)} (conditionnement different)")
    return None


def _ecarter(chemin, pourquoi):
    """Renomme un checkpoint perime au lieu de le detruire, et le dit."""
    _v = chemin.with_name(f"{chemin.stem}.perime_{int(time.time())}.pth")
    chemin.rename(_v)
    print(f"CHECKPOINT ECARTE : {pourquoi}")
    print(f"  conserve sous {_v.name}")


_LAST = CKPT_DIR / "stage1_last.pth"
history, best, _start, _r = [], float("inf"), 0, None
if _LAST.exists():
    _r = torch.load(_LAST, map_location=DEVICE, weights_only=False)
    _dif = _incompatible_s1(_r)
    if _dif:
        _ecarter(_LAST, _dif)
        print("  l'etage 1 repart de zero.")
        _r = None
if _r is not None:
    encoder.load_state_dict(_r["encoder_state_dict"])
    rcn_cell.load_state_dict(_r["rcn_cell_state_dict"])
    regression_head.load_state_dict(_r["regression_head_state_dict"])
    if bg_head is not None and "bg_head_state_dict" in _r:
        bg_head.load_state_dict(_r["bg_head_state_dict"])
    # L'optimiseur AVANT le scheduler : `SequentialLR` remet le lr a sa valeur
    # de warmup a la construction et son load_state_dict ne restaure que des
    # compteurs. C'est l'optimiseur qui porte le lr courant.
    opt_s1.load_state_dict(_r["optimizer_state_dict"])
    history, best, _start = _r["history"], _r["best"], _r["epoch"] + 1
    if "scheduler_state_dict" in _r:
        sched_s1.load_state_dict(_r["scheduler_state_dict"])
    else:
        for _ in range(_start):
            sched_s1.step()
    _sans_gain = int(_r.get("sans_gain", 0))
    print(f"REPRISE a l'epoque {_start + 1}/{EPOCHS_S1} "
          f"(meilleure val = {best:.5f}, lr={opt_s1.param_groups[0]['lr']:.2e})")
else:
    _sans_gain = 0

for ep in range(_start, EPOCHS_S1):
    t_ep = time.time()
    m = train_epoch_stage1(
        encoder=encoder, rcn_runner=rcn_runner, regression_head=regression_head,
        optimizer=opt_s1, data_loader=iterate_batches(train_dataset),
        device=DEVICE, epoch_idx=ep,
        lambda_reg=float(S1.lambda_reg), beta_rec=float(S1.beta_rec),
        gamma_dag_max=float(S1.gamma_dag_max),
        gamma_dag_warmup_epochs=int(S1.gamma_dag_warmup_epochs),
        lambda_l1=float(S1.lambda_l1),
        abort_on_collapse=bool(S1.abort_on_collapse),
        collapse_threshold=float(S1.collapse_threshold),
        dag_floor_projection=bool(S1.dag_floor_projection),
        dag_floor_min_norm=float(S1.dag_floor_min_norm),
        gradient_clipping=float(CONFIG.training.gradient_clipping),
        use_amp=(DEVICE.type == "cuda"),
        # --- A3 : la NLL Bernoulli-Gamma remplace la MSE ------------------
        bg_head=bg_head,
        # --- C7 : prior 3 niveaux, annele PAR NIVEAU ----------------------
        edge_prior=edge_prior, prior_node_order=NODE_TYPES,
        prior_level_floors=DEFAULT_LEVEL_FLOORS, prior_anneal_epochs=EPOCHS_S1,
        # --- exclusions imposees par bg_head (le loop les verifie) --------
        # Le seuil du LOOP prime sur celui de la tete : le laisser a 1,0
        # ferait ajuster la Gamma au-dessus de 1 mm/j, ce qui donne alpha=1,05
        # au lieu de 0,55 - la forme fausse d'un facteur 2, et c'est elle qui
        # gouverne les extremes.
        bg_wet_threshold=BG_WET_SEUIL,
        skip_block=skip_block, p1_tail_alpha=0.0, p3_k_samples=0,
        verbose=(ep == 0),
    )
    m["epoch"] = ep
    m["seconds"] = round(time.time() - t_ep, 1)
    history.append(m)
    print(f"[S1 {ep + 1:2d}/{EPOCHS_S1}] loss={m['loss']:.5f} "
          f"reg={m['loss_reg']:.5f} dag={m.get('loss_dag', 0.0):.4f} "
          f"({m['seconds']:.0f}s)")
    # POIDS NON FINIS : arret immediat. Le gradient d'un poids de convolution
    # somme sur tout le domaine, donc une seule cellule divergente suffit a
    # rendre NaN toute la tete — et plus rien ne le signale ensuite : la perte
    # affiche nan, l'entrainement continue, et l'anomalie ne se decouvre que
    # trois cellules plus loin sous la forme "0 pixel exploitable". Detecter
    # ici coute un balayage des parametres par epoque.
    _mauvais = [n for _mod, _pref in zip(STAGE1_MODULES,
                                         ("encodeur", "RCN", "decodeur", "tete BG"))
                for n, _p in _mod.named_parameters()
                if not torch.isfinite(_p).all() for n in (f"{_pref}.{n}",)]
    if _mauvais:
        raise RuntimeError(
            f"epoque {ep + 1} : parametres NON FINIS -> {_mauvais[:6]}"
            + (f" (+{len(_mauvais) - 6} autres)" if len(_mauvais) > 6 else "")
            + ". L'entrainement a diverge ; poursuivre ne produirait que des "
              "NaN. Verifier alpha_min et le plancher de la tete.")

    m["val_loss"] = validation_loss(val_dataset)
    m["lr"] = opt_s1.param_groups[0]["lr"]
    sched_s1.step()
    print(f"          val={m['val_loss']:.5f} (lr={m['lr']:.2e})")

    _poids = {"encoder_state_dict": encoder.state_dict(),
              "rcn_cell_state_dict": rcn_cell.state_dict(),
              "regression_head_state_dict": regression_head.state_dict(),
              "node_types": NODE_TYPES, "v8": OmegaConf.to_container(V8)}
    if bg_head is not None:
        _poids["bg_head_state_dict"] = bg_head.state_dict()

    # A CHAQUE epoque : point de reprise, ETAT DE L'OPTIMISEUR compris. Sans
    # lui, reprendre repartirait avec des moments Adam nuls — ce ne serait pas
    # la meme trajectoire d'optimisation.
    torch.save({**_poids, "epoch": ep,
                "optimizer_state_dict": opt_s1.state_dict(),
                "scheduler_state_dict": sched_s1.state_dict(),
                "sans_gain": _sans_gain,
                "history": history, "best": best}, _LAST)

    # Selection sur la VALIDATION, jamais sur l'entrainement.
    if m["val_loss"] < best:
        best = m["val_loss"]
        torch.save({"epoch": ep,
                    "encoder_state_dict": encoder.state_dict(),
                    "rcn_cell_state_dict": rcn_cell.state_dict(),
                    "regression_head_state_dict": regression_head.state_dict(),
                    **({"bg_head_state_dict": bg_head.state_dict()} if bg_head else {}),
                    "node_types": NODE_TYPES,
                    "v8": OmegaConf.to_container(V8)},
                   CKPT_DIR / "stage1_best.pth")
        _sans_gain = 0
    else:
        _sans_gain += 1
        if _sans_gain >= PATIENCE:
            print(f"ARRET ANTICIPE : {PATIENCE} epoques sans gain de validation "
                  f"(meilleure = {best:.5f} a l'epoque "
                  f"{1 + max(range(len(history)), key=lambda i: -history[i].get('val_loss', 9e9))}). "
                  f"Les epoques suivantes ne feraient que memoriser.")
            break

json.dump(history, open(RESULTS_DIR / "v8_stage1_history.json", "w"),
          indent=2, default=float)
# Trajectoire, en trois lignes. Sur une reprise ou la boucle ne tourne pas, la
# question "l'etage 1 a-t-il seulement appris ?" se poserait sinon sans reponse
# a l'ecran — et c'est la premiere a se poser quand mu_HR se revele inutilisable.
if history:
    _tr = [float(h["loss"]) for h in history]
    _va = [float(h["val_loss"]) for h in history if "val_loss" in h]
    print(f"trajectoire etage 1 sur {len(history)} epoques :")
    print(f"  entrainement {_tr[0]:.5f} -> {_tr[-1]:.5f} (min {min(_tr):.5f})")
    if _va:
        print(f"  validation   {_va[0]:.5f} -> {_va[-1]:.5f} (min {min(_va):.5f}"
              f" a l'epoque {_va.index(min(_va)) + 1})")
# CHARGER le meilleur checkpoint. Sans cela, les cellules suivantes
# travailleraient sur les poids de la DERNIERE epoque et le checkpoint
# "meilleur" ne serait qu'un fichier decoratif.
# MEME CONTROLE que la reprise : ce fichier-ci survit aussi d'un run a l'autre,
# et il etait charge sans rien verifier. Un checkpoint d'une configuration
# anterieure levait donc ici, apres l'entrainement complet.
_BEST = CKPT_DIR / "stage1_best.pth"
_ck = torch.load(_BEST, map_location=DEVICE, weights_only=False) if _BEST.exists() else None
if _ck is not None:
    _dif_b = _incompatible_s1(_ck)
    if _dif_b:
        _ecarter(_BEST, _dif_b)
        _ck = None
if _ck is not None:
    encoder.load_state_dict(_ck["encoder_state_dict"])
    rcn_cell.load_state_dict(_ck["rcn_cell_state_dict"])
    regression_head.load_state_dict(_ck["regression_head_state_dict"])
    if bg_head is not None:
        bg_head.load_state_dict(_ck["bg_head_state_dict"])
    print(f"meilleure perte de VALIDATION : {best:.5f} (epoque {_ck['epoch'] + 1}) "
          f"- poids recharges")
elif history:
    # La boucle a tourne : les poids en memoire sont ceux de la DERNIERE
    # epoque, pas de la meilleure. C'est utilisable, mais ce n'est pas la meme
    # chose et le silence serait trompeur.
    print("ATTENTION : aucun checkpoint 'meilleur' exploitable. Les cellules "
          "suivantes utilisent les poids de la DERNIERE epoque.")
else:
    raise RuntimeError(
        "Ni entrainement ni checkpoint 'meilleur' utilisable : les poids en "
        "memoire sont ceux de l'INITIALISATION. Relancer la Cell 5 apres avoir "
        "verifie EPOCHS_S1 et le contenu de " + str(CKPT_DIR))
