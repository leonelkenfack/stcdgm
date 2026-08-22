# >>> Cell 6 : ce que le DAG a appris - decouvert contre impose
A_lag = rcn_cell.dag_matrix(masked=True).detach().cpu().numpy()
print(f"A(tau>=1) : norme={np.linalg.norm(A_lag):.4f} "
      f"asymetrie={np.abs(A_lag - A_lag.T).mean():.4f}")
if rcn_cell.A_inst is not None:
    A0 = rcn_cell.dag_matrix(masked=True, lag=0).detach().cpu().numpy()
    print(f"A(0)      : norme={np.linalg.norm(A0):.4f} "
          f"asymetrie={np.abs(A0 - A0.T).mean():.4f}")

def _score(A, lag_v, label):
    """Confronte UN operateur a la partie du prior qui lui revient.
    Melanger les lags rendrait le bilan faux sur 24 des 33 aretes."""
    P = edge_prior.matrix(lag=lag_v, node_order=NODE_TYPES)
    support = P != 0
    if not support.any():
        return None
    # Seuil sur le quantile 80 : on garde le meme budget d'aretes que le prior
    # plutot qu'un seuil absolu arbitraire qui dependrait de l'echelle de A.
    thr = float(np.quantile(np.abs(A), 0.80))
    found = np.abs(A) > thr
    kept, dropped = int((found & support).sum()), int((~found & support).sum())
    novel = int((found & ~support).sum())
    print(f"\nprior : {int(support.sum())} aretes | seuil |A| > {thr:.4f}")
    print(f"  retenues par les donnees : {kept}")
    print(f"  REJETEES                 : {dropped}   <- signal de decouverte")
    print(f"                                            NEGATIF, a rapporter (C7)")
    print(f"  hors prior               : {novel}   <- decouverte au-dela du prior")

    # Par niveau : le niveau 3 DOIT pouvoir tomber. C'est tout l'objet de
    # l'annealing par niveau - un prior speculatif que les donnees ne peuvent
    # pas rejeter n'est plus un prior, c'est une contrainte.
    print("\n  survie par niveau de credibilite :")
    per_level = {}
    for lvl, mask in edge_prior.level_masks(node_order=NODE_TYPES).items():
        mask = mask & support        # restreindre au lag de CET operateur
        if mask.any():
            r = float((found & mask).sum() / mask.sum())
            per_level[int(lvl)] = r
            print(f"    niveau {lvl} ({int(mask.sum()):2d} aretes) : {100 * r:4.0f} %")

    # Les aretes rejetees, nommees : c'est le livrable scientifique de C7.
    rejected = [(NODE_TYPES[i], NODE_TYPES[j])
                for i, j in zip(*np.where(~found & support))]
    if rejected:
        print("\n  aretes du prior REJETEES par les donnees :")
        for a, b in rejected:
            print(f"    {a} -> {b}")

    return {"operator": label, "n_prior": int(support.sum()), "kept": kept,
            "dropped": dropped, "novel": novel, "threshold": thr,
            "survival_by_level": per_level, "rejected_edges": rejected}


if edge_prior is not None:
    report = [r for r in (_score(A_lag, 1, "A(tau>=1)"),
                          (_score(A0, 0, "A(0)") if rcn_cell.A_inst is not None else None))
              if r is not None]
    json.dump(report, open(RESULTS_DIR / "v8_dag_vs_prior.json", "w"), indent=2)
