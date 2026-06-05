"""Fix build_stack avec la vraie API : IntelligibleVariableConfig(name, meta_path, pool),
RCNCell(num_vars=...), CausalDiffusionDecoder avec unet_kwargs converti en tuples."""
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

NB = Path("st_cdgm_v5_evaluation.ipynb")
with NB.open(encoding="utf-8") as f:
    nb = json.load(f)

BACKUP = Path("st_cdgm_v5_evaluation.ipynb.bak8")
if not BACKUP.exists():
    BACKUP.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] Backup #8 cree : {BACKUP}")

# Source actuel de Cell 4
src = "".join(nb["cells"][4]["source"])

OLD_BUILD_STACK = '''def build_stack(ckpt_path, name):
    print(f"  [{name}] {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    enc_cfg = IntelligibleVariableConfig(
        num_variables=len(CONFIG.data.lr_variables),
        hidden_dim=CONFIG.encoder.hidden_dim,
        conditioning_dim=CONFIG.encoder.conditioning_dim,
        num_dag_tokens=int(CONFIG.encoder.get("num_dag_tokens", 2)),
        causal_conditioning=bool(CONFIG.encoder.get("causal_conditioning", True)),
    )
    enc = IntelligibleVariableEncoder(enc_cfg).to(DEVICE)
    rcn_cell = RCNCell(
        num_variables=enc_cfg.num_variables,
        hidden_dim=CONFIG.rcn.hidden_dim,
        driver_dim=CONFIG.rcn.driver_dim,
        reconstruction_dim=CONFIG.rcn.reconstruction_dim,
        dropout=CONFIG.rcn.dropout,
    ).to(DEVICE)
    rcn_runner = RCNSequenceRunner(rcn_cell, detach_interval=CONFIG.rcn.detach_interval)
    rh = GraphToGridDecoder(
        d_model=CONFIG.encoder.hidden_dim,
        hr_h=CONFIG.graph.hr_shape[0], hr_w=CONFIG.graph.hr_shape[1],
    ).to(DEVICE)
    edm_cfg = EDMConfig.from_yaml_dict(CONFIG.diffusion.get("edm", {}))
    diff = CausalDiffusionDecoder(
        in_channels=CONFIG.diffusion.in_channels,
        conditioning_dim=CONFIG.diffusion.conditioning_dim,
        height=CONFIG.diffusion.height, width=CONFIG.diffusion.width,
        scheduler_type=CONFIG.diffusion.scheduler_type, causal_concat=True,
        edm_config=edm_cfg, unet_kwargs=dict(CONFIG.diffusion.unet_kwargs),
    ).to(DEVICE)'''

NEW_BUILD_STACK = '''def build_stack(ckpt_path, name):
    print(f"  [{name}] {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    # 1. Encoder configs depuis CONFIG.encoder.metapaths (filtre selon allowed_nodes du builder)
    allowed_nodes = set(builder.dynamic_node_types + builder.static_node_types)
    encoder_configs = []
    for _mp in CONFIG.encoder.metapaths:
        _src, _rel, _tgt = _mp.src, _mp.relation, _mp.target
        if _src in allowed_nodes and _tgt in allowed_nodes:
            encoder_configs.append(IntelligibleVariableConfig(
                name=_mp.name,
                meta_path=(_src, _rel, _tgt),
                pool=_mp.get("pool", "mean"),
            ))
    # Ajoute le static metapath si pipeline a static_dataset
    if pipeline_access.get_static_dataset() is not None:
        encoder_configs.append(IntelligibleVariableConfig(
            name="static", meta_path=("SP_HR", "causes", "GP850"), pool="mean",
        ))

    enc = IntelligibleVariableEncoder(
        configs=encoder_configs,
        hidden_dim=CONFIG.encoder.hidden_dim,
        conditioning_dim=CONFIG.encoder.conditioning_dim,
    ).to(DEVICE)
    num_vars = len(encoder_configs)

    rcn_cell = RCNCell(
        num_vars=num_vars,
        hidden_dim=CONFIG.rcn.hidden_dim,
        driver_dim=int(CONFIG.rcn.driver_dim),
        reconstruction_dim=int(CONFIG.rcn.reconstruction_dim),
        dropout=CONFIG.rcn.dropout,
    ).to(DEVICE)
    rcn_runner = RCNSequenceRunner(rcn_cell, detach_interval=CONFIG.rcn.get("detach_interval"))

    rh = GraphToGridDecoder(
        d_model=CONFIG.encoder.hidden_dim,
        hr_h=CONFIG.graph.hr_shape[0], hr_w=CONFIG.graph.hr_shape[1],
    ).to(DEVICE)

    edm_cfg = EDMConfig.from_yaml_dict(CONFIG.diffusion.get("edm", {}))
    # Convertit unet_kwargs en dict + tuples pour block_types
    from omegaconf import OmegaConf as _OC
    _unet_kwargs = _OC.to_container(CONFIG.diffusion.unet_kwargs, resolve=True)
    for _k in ("down_block_types", "up_block_types"):
        if _k in _unet_kwargs and isinstance(_unet_kwargs[_k], list):
            _unet_kwargs[_k] = tuple(_unet_kwargs[_k])

    hr_channels = int(sample["residual"].shape[1])

    diff = CausalDiffusionDecoder(
        in_channels=hr_channels,
        conditioning_dim=CONFIG.diffusion.conditioning_dim,
        height=int(CONFIG.diffusion.height),
        width=int(CONFIG.diffusion.width),
        unet_kwargs=_unet_kwargs,
        scheduler_type=str(CONFIG.diffusion.scheduler_type),
        use_gradient_checkpointing=bool(CONFIG.diffusion.get("use_gradient_checkpointing", False)),
        conv_padding_mode=str(CONFIG.diffusion.get("conv_padding_mode", "zeros")),
        anti_checkerboard=bool(CONFIG.diffusion.get("anti_checkerboard", False)),
        edm_config=edm_cfg,
        causal_concat=True,
    ).to(DEVICE)'''

if OLD_BUILD_STACK not in src:
    print("[ERREUR] Bloc build_stack ancien introuvable.")
    sys.exit(1)

src_new = src.replace(OLD_BUILD_STACK, NEW_BUILD_STACK)

nb["cells"][4]["source"] = [ln + "\n" for ln in src_new.split("\n")[:-1]] + [src_new.split("\n")[-1]]

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[OK] build_stack corrige : signature IntelligibleVariableConfig(name, meta_path, pool)")
print(f"     + RCNCell(num_vars=...) + CausalDiffusionDecoder avec tuples block_types")
print(f"     {len(nb['cells'])} cellules")
