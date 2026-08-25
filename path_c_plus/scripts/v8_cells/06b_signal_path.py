# >>> Cell 6b : ou meurt la dependance au jour ?
# mu_HR s'est revele sans correlation avec sa cible alors que la perte
# d'entrainement avait baisse de 45 %. Une perte qui descend sans que la
# validation bouge decrit un modele qui apprend la CLIMATOLOGIE puis memorise :
# un champ juste en moyenne, immobile d'un jour a l'autre.
#
# Reste a savoir OU la meteo se perd le long de la chaine. On mesure donc, a
# chaque etage, la part de variance portee par le JOUR plutot que par le lieu
# ou la variable. L'entree sert de temoin : c'est elle qui fixe combien il y
# avait a transmettre.
import itertools

_K_SONDE = 24
_ech = list(itertools.islice(train_dataset, 0, 10 * _K_SONDE, 10))
print(f"sonde sur {len(_ech)} jours espaces de 10 fenetres")

_etages = {"entree LR (temoin)": [], "etat RCN H_T": [],
           "features decodeur": [], "ancre E[log1p]": []}
_champs = []          # (ancre, HR vrai) par jour, pour la decomposition ci-dessous
for _m in STAGE1_MODULES:
    _m.eval()
with torch.no_grad():
    for _s in _ech:
        _b = convert_sample_v8(_s, builder, DEVICE)
        _lr = _b["lr"].to(DEVICE)
        _H0 = encoder.init_state(_b["hetero"])
        _sq = rcn_runner.run(_H0, [_lr[k] for k in range(_lr.shape[0])],
                             reconstruction_sources=None)
        _HT = _sq.states[-1]
        _f = regression_head(_HT, return_features=True)
        _etages["entree LR (temoin)"].append(_lr[-1].flatten().cpu())
        _etages["etat RCN H_T"].append(_HT.flatten().cpu())
        _etages["features decodeur"].append(_f.flatten().cpu())
        if bg_head is not None:
            _p, _a, _bb = bg_head(_f)
            _anc_j = BernoulliGammaHead.mean_log1p(_p, _a, _bb)
            _etages["ancre E[log1p]"].append(_anc_j.flatten().cpu())
            _tj = _b["residual"][-1].to(DEVICE)
            _blj = _b["baseline"][-1].to(DEVICE)
            if _tj.dim() == 3:
                _tj, _blj = _tj.unsqueeze(0), _blj.unsqueeze(0)
            _champs.append((_anc_j.squeeze().cpu(), (_blj + _tj).squeeze().cpu()))
for _m in STAGE1_MODULES:
    _m.train()


def _part_jour(xs):
    """Fraction de la variance totale qui vient du jour."""
    if not xs:
        return float("nan")
    _X = torch.stack(xs).double()
    _v = float(_X.var())
    return float(_X.var(dim=0).mean() / _v) if _v > 1e-12 else float("nan")


_parts = {k: _part_jour(v) for k, v in _etages.items() if v}
print("part de variance portee par le jour :")
for _k, _v in _parts.items():
    print(f"  {_k:22s} {100 * _v:6.2f} %")

# Le temoin borne ce qui etait transmissible ; on cherche le premier etage qui
# effondre cette part. Un facteur 10 est deliberement grossier : il s'agit de
# reperer une rupture, pas de mesurer une attenuation.
_ref = _parts.get("entree LR (temoin)", float("nan"))
_mort = None
_prec = _ref
for _k, _v in list(_parts.items())[1:]:
    if _v == _v and _prec == _prec and _v < _prec / 10.0:
        _mort = _k
        break
    _prec = _v
_fin = list(_parts.values())[-1]
if _mort:
    print(f"RUPTURE : la dependance au jour s'effondre a l'etage << {_mort} >>.")
    print("  C'est la qu'il faut chercher, pas en aval.")
elif _fin == _fin and _ref == _ref and _fin < _ref / 10.0:
    # Angle mort du critere par etage : une chaine qui perd un facteur 3 a
    # chaque passage n'accuse aucun etage et n'en transmet pourtant rien.
    print(f"ATTENUATION CUMULATIVE : aucun etage ne casse seul, mais la sortie "
          f"ne garde que {100 * _fin / _ref:.1f} % de la part temporelle de "
          f"l'entree. Le defaut est reparti sur toute la chaine.")
elif _ref == _ref:
    print("Aucune rupture franche : la meteo traverse toute la chaine.")

# --- Niveau ou motif ? -----------------------------------------------------
# Une part temporelle elevee dit que l'ancre BOUGE, pas qu'elle bouge bien. On
# separe donc le NIVEAU du jour (sa moyenne spatiale) du MOTIF (l'anomalie
# autour de ce niveau), et on regarde lequel des deux suit la verite. Predire
# la pluie moyenne du domaine sans son organisation spatiale, et predire une
# organisation sans le bon niveau, sont deux echecs opposes.
if _champs:
    _AN = torch.stack([c[0] for c in _champs]).double()      # [K, H, W]
    _HR = torch.stack([c[1] for c in _champs]).double()
    _ok = torch.isfinite(_HR).all(dim=0)                     # pixels valides partout
    _AN, _HR = _AN[:, _ok], _HR[:, _ok]                      # [K, P]
    _niv_a, _niv_h = _AN.mean(dim=1), _HR.mean(dim=1)        # niveau du jour
    _mot_a, _mot_h = _AN - _niv_a[:, None], _HR - _niv_h[:, None]

    def _c(x, y):
        x, y = x.flatten() - x.mean(), y.flatten() - y.mean()
        _d = x.norm() * y.norm()
        return float(x @ y / _d) if _d > 1e-12 else float("nan")

    print("decomposition de l'ancre (niveau du jour / motif spatial) :")
    print(f"  ecart-type   niveau {float(_niv_a.std()):.4f}  "
          f"motif {float(_mot_a.std()):.4f}   "
          f"(verite : {float(_niv_h.std()):.4f} / {float(_mot_h.std()):.4f})")
    print(f"  correlation  niveau {_c(_niv_a, _niv_h):+.3f}  "
          f"motif {_c(_mot_a, _mot_h):+.3f}")
    if float(_mot_a.std()) < 0.1 * float(_mot_h.std()):
        print("  -> la tete ne produit quasiment PAS de motif spatial : elle "
              "predit un niveau journalier, pas un champ de pluie.")
    elif _c(_mot_a, _mot_h) < 0.1:
        print("  -> le motif spatial existe mais ne suit pas la verite.")
    del _AN, _HR, _mot_a, _mot_h
del _etages, _ech, _champs
