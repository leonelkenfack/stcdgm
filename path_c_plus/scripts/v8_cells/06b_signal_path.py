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
            _champs.append((_anc_j.squeeze().cpu(), (_blj + _tj).squeeze().cpu(),
                            _blj.squeeze().cpu()))
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

# --- Niveau ou motif, et surtout : MIEUX QUE LA BASELINE ? -----------------
# Une part temporelle elevee dit que l'ancre BOUGE, pas qu'elle bouge bien. On
# separe donc le NIVEAU du jour (sa moyenne spatiale) du MOTIF (l'anomalie
# autour de ce niveau), et on correle chacun a la verite decomposee pareil.
#
# LE TEMOIN EST LA BASELINE, jamais zero. Correler l'ancre au HR vrai flatte :
# le HR ressemble deja beaucoup a la baseline (c'est son interpolation), donc
# recopier l'entree suffit a decrocher +0,9. Ce que l'etage 1 doit produire
# n'est pas HR, c'est HR MOINS la baseline — la part que l'interpolation ne
# donne pas. La ligne qui compte est donc la derniere : le residu.
if _champs:
    _AN = torch.stack([c[0] for c in _champs]).double()      # ancre  [K, H, W]
    _HR = torch.stack([c[1] for c in _champs]).double()      # verite
    _BL = torch.stack([c[2] for c in _champs]).double()      # baseline
    _ok = torch.isfinite(_HR).all(dim=0)                     # pixels valides partout
    _AN, _HR, _BL = _AN[:, _ok], _HR[:, _ok], _BL[:, _ok]    # [K, P]

    def _c(x, y):
        x, y = x.flatten() - x.mean(), y.flatten() - y.mean()
        _d = x.norm() * y.norm()
        return float(x @ y / _d) if _d > 1e-12 else float("nan")

    def _decomp(x):
        _n = x.mean(dim=1)
        return _n, x - _n[:, None]

    _na, _ma = _decomp(_AN)
    _nh, _mh = _decomp(_HR)
    _nb, _mb = _decomp(_BL)
    print("niveau du jour / motif spatial, correles a la verite :")
    print(f"  ancre     niveau {_c(_na, _nh):+.3f} ({float(_na.std()):.4f})  "
          f"motif {_c(_ma, _mh):+.3f} ({float(_ma.std()):.4f})")
    print(f"  BASELINE  niveau {_c(_nb, _nh):+.3f} ({float(_nb.std()):.4f})  "
          f"motif {_c(_mb, _mh):+.3f} ({float(_mb.std()):.4f})   <- le temoin")
    print(f"  verite                    ({float(_nh.std()):.4f})  "
          f"           ({float(_mh.std()):.4f})")

    # Ce que l'etage 2 recevra vraiment : mu = ancre - baseline contre
    # t = HR - baseline. C'est la meme quantite que la Cell 8, sur 24 jours.
    _mu, _t = _AN - _BL, _HR - _BL
    _r2 = 1.0 - float((_t - _mu).pow(2).sum() / _t.pow(2).sum())
    print(f"  RESIDU    corr(mu, t) = {_c(_mu, _t):+.3f} | R2 = {_r2:+.3f} "
          f"| sigma {float(_mu.std()):.4f} contre {float(_t.std()):.4f}")
    if _c(_ma, _mh) <= _c(_mb, _mh) + 0.02:
        print("  -> l'ancre ne fait PAS mieux que la baseline sur le motif : "
              "l'etage 1 recopie son entree au lieu d'ajouter du fin.")
    elif float(_ma.std()) < 0.1 * float(_mh.std()):
        print("  -> la tete ne produit quasiment PAS de motif spatial.")
    del _AN, _HR, _BL, _ma, _mh, _mb, _mu, _t
del _etages, _ech, _champs
