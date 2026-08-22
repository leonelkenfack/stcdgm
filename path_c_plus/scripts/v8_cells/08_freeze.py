# >>> Cell 8 : gel de l'etage 1 + cache pour l'etage 2
from st_cdgm.training.two_stage import freeze_stage1, precompute_stage1_outputs
from st_cdgm.training.stage1_paths import calibrate_sigma_data_variant

sig = calibrate_sigma_data_variant(
    variant="causal", regression_head=regression_head, data_loader=train_dataset,
    iterate_batches_fn=iterate_batches, builder=builder, device=DEVICE,
    encoder=encoder, rcn_runner=rcn_runner, max_samples=200)
SIGMA_DATA = float(sig["sigma_data"])
print(f"sigma_data = {SIGMA_DATA:.5f}")

freeze_stage1(encoder, rcn_cell, regression_head, *([bg_head] if bg_head else []))
rcn_cell.A_dag.requires_grad_(False)
if rcn_cell.A_inst is not None:
    rcn_cell.A_inst.requires_grad_(False)
print("etage 1 gele, A_dag et A(0) compris - le DAG devient une feature OOD assumee")

# L'ancre mise en cache est mu = p*alpha*beta reexprimee en residu log1p,
# pas la projection 1 canal du decodeur (qui n'est plus entrainee sous A3).
cache = precompute_stage1_outputs(
    encoder=encoder, rcn_runner=rcn_runner, regression_head=regression_head,
    train_dataset=train_dataset,
    iterate_batches_fn=lambda s: convert_sample_to_batch(s, builder, DEVICE),
    device=DEVICE, bg_head=bg_head)
print({k: tuple(v.shape) for k, v in cache.items()})

torch.save({"sigma_data": SIGMA_DATA, "node_types": NODE_TYPES,
            "v8": OmegaConf.to_container(V8)},
           CKPT_DIR / "stage1_frozen_meta.pth")
