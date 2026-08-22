# >>> Cell 5 : entrainement etage 1
from torch.optim import AdamW
from st_cdgm.training.training_loop import train_epoch_stage1

def iterate_batches(ds, bld=None, dev=None):
    """Signature (data_loader, builder, device) attendue par les helpers."""
    bld = bld if bld is not None else builder
    dev = dev if dev is not None else DEVICE
    for s in ds:
        yield [convert_sample_to_batch(s, bld, dev)]

# bg_head fait partie des parametres optimises : l'oublier laisserait la tete
# a son initialisation et la NLL ne descendrait jamais. L'encodeur, lui, a ete
# materialise en Cell 4 - sinon ses poids lazy seraient absents d'ici.
assert all(sum(1 for _ in m.parameters()) > 0 for m in STAGE1_MODULES),     "un module n'expose aucun parametre - voir la materialisation en Cell 4"
opt_s1 = AdamW([p for m in STAGE1_MODULES for p in m.parameters()],
               lr=float(CONFIG.training.learning_rate), betas=(0.9, 0.99))

history, best = [], float("inf")
for ep in range(EPOCHS_S1):
    t_ep = time.time()
    m = train_epoch_stage1(
        encoder=encoder, rcn_runner=rcn_runner, regression_head=regression_head,
        optimizer=opt_s1, data_loader=iterate_batches(train_dataset),
        device=DEVICE, epoch_idx=ep,
        lambda_reg=float(CONFIG.two_stage.get("lambda_reg", 1.0)),
        beta_rec=float(CONFIG.two_stage.get("beta_rec", 0.05)),
        gamma_dag_max=float(CONFIG.two_stage.get("gamma_dag_max", 0.10)),
        lambda_l1=float(CONFIG.two_stage.get("lambda_l1", 0.01)),
        gradient_clipping=1.0, use_amp=(DEVICE.type == "cuda"),
        # --- A3 : la NLL Bernoulli-Gamma remplace la MSE ------------------
        bg_head=bg_head,
        # --- C7 : prior 3 niveaux, annele PAR NIVEAU ----------------------
        edge_prior=edge_prior, prior_node_order=NODE_TYPES,
        prior_level_floors=DEFAULT_LEVEL_FLOORS, prior_anneal_epochs=EPOCHS_S1,
        # --- exclusions imposees par bg_head (le loop les verifie) --------
        skip_block=skip_block, p1_tail_alpha=0.0, p3_k_samples=0,
        verbose=(ep == 0),
    )
    m["epoch"] = ep
    m["seconds"] = round(time.time() - t_ep, 1)
    history.append(m)
    print(f"[S1 {ep + 1:2d}/{EPOCHS_S1}] loss={m['loss']:.5f} "
          f"reg={m['loss_reg']:.5f} dag={m.get('loss_dag', 0.0):.4f} "
          f"({m['seconds']:.0f}s)")
    if m["loss"] < best:
        best = m["loss"]
        torch.save({"epoch": ep,
                    "encoder_state_dict": encoder.state_dict(),
                    "rcn_cell_state_dict": rcn_cell.state_dict(),
                    "regression_head_state_dict": regression_head.state_dict(),
                    **({"bg_head_state_dict": bg_head.state_dict()} if bg_head else {}),
                    "node_types": NODE_TYPES,
                    "v8": OmegaConf.to_container(V8)},
                   CKPT_DIR / "stage1_best.pth")

json.dump(history, open(RESULTS_DIR / "v8_stage1_history.json", "w"),
          indent=2, default=float)
print(f"meilleure perte etage 1 : {best:.5f}")
