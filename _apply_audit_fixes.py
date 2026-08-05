"""Apply audit fixes from 3-agent review.

phase7_stage2_B_finetune.ipynb :
  - Cell 5  : finite + shape validation on loaded cache
  - Cell 6  : log warm-start sigma_data vs override (warn on mismatch)
  - Cell 7  : (a) resume config validation
              (b) restore ema_step_counter
              (c) reseed before training loop
              (d) drop dead _ema_update fn

phase6_dualpath_final_validation.ipynb :
  - Cell 2  : add STAGE2_CHECKPOINT override + test_dataloader hook
  - Cell 3  : build test_dataloader alongside val/train
  - Cell 4  : EMA fallback (diffusion_ema_state_dict OR diffusion_state_dict),
              read sigma_data from ckpt with hard-coded fallback
  - Cell 6  : iterate test_dataloader (was val)
  - Cell 8  : iterate test_dataloader (was val)
"""
import json
from pathlib import Path

# ============================================================================
# phase7_stage2_B_finetune.ipynb
# ============================================================================
nb_path = Path('path_c_plus/scripts/phase7_stage2_B_finetune.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))

# ---- Cell 5 : assert cache integrity ----
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    if "[Cell 5] Cache rehydrated to dense RAM" in src or "cache = torch.load(LOCAL_CACHE_PATH" in src:
        cell5_idx = i
        break
else:
    raise RuntimeError('Cell 5 not found')

old_5 = "print(f'[Cell 5] delta_target stats : mean={cache[\"delta_target\"].mean():.4f} '\n      f'std={cache[\"delta_target\"].std():.4f}')\n"
new_5 = (
    "print(f'[Cell 5] delta_target stats : mean={cache[\"delta_target\"].mean():.4f} '\n"
    "      f'std={cache[\"delta_target\"].std():.4f}')\n"
    "\n"
    "# Cache integrity validation (audit fix #1 : guard against silently corrupt local copy).\n"
    "for _k in ('mu_HR', 'baseline_log', 'delta_target'):\n"
    "    _v = cache[_k]\n"
    "    if not torch.isfinite(_v).all():\n"
    "        _n_bad = int((~torch.isfinite(_v)).sum().item())\n"
    "        raise RuntimeError(\n"
    "            f'[Cell 5] cache[{_k!r}] has {_n_bad} non-finite values. '\n"
    "            f'DELETE {LOCAL_CACHE_PATH} and {STAGE1_CACHE_PATH} then re-run Cell 5.'\n"
    "        )\n"
    "_n0 = int(cache['mu_HR'].shape[0])\n"
    "_n1 = int(cache['delta_target'].shape[0])\n"
    "assert _n0 == _n1 == int(cache['baseline_log'].shape[0]) == int(cache['valid_mask'].shape[0]), (\n"
    "    f'[Cell 5] cache shape mismatch : mu_HR={_n0} dlt={_n1}')\n"
    "print(f'[Cell 5] integrity check OK : N={_n0}  all-finite  shapes-aligned')\n"
)
src5 = ''.join(nb['cells'][cell5_idx]['source'])
n5 = src5.count(old_5)
print(f'Cell 5 old_5 matched : {n5} (expect 1)')
assert n5 == 1
src5 = src5.replace(old_5, new_5, 1)
try:
    compile(src5, 'cell-5', 'exec')
except SyntaxError as e:
    print(f'Cell 5 SYNTAX ERROR L{e.lineno}: {e.msg}')
    raise
lines5 = src5.split('\n')
nb['cells'][cell5_idx]['source'] = [l + '\n' for l in lines5[:-1]] + ([lines5[-1]] if lines5[-1] else [])

# ---- Cell 6 : log warm-start sigma_data ----
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    if 'def build_stage2' in src:
        cell6_idx = i
        break
else:
    raise RuntimeError('Cell 6 not found')

old_6 = (
    "    if warm_start:\n"
    "        if not CKPT_STAGE2_WARMSTART.exists():\n"
    "            raise FileNotFoundError(f'Warm-start ckpt missing : {CKPT_STAGE2_WARMSTART}')\n"
    "        ck = torch.load(CKPT_STAGE2_WARMSTART, map_location=DEVICE, weights_only=False)\n"
    "        diff_sd = _strip_prefixes(ck.get('diffusion_state_dict'))\n"
    "        if diff_sd is None:\n"
    "            raise RuntimeError(f'diffusion_state_dict absent from {CKPT_STAGE2_WARMSTART}')\n"
    "        info = diff.load_state_dict(diff_sd, strict=False)\n"
    "        print(f'  [build_stage2] warm-start loaded | missing={len(info.missing_keys)} '\n"
    "              f'unexpected={len(info.unexpected_keys)}')\n"
    "        del ck, diff_sd\n"
)
new_6 = (
    "    if warm_start:\n"
    "        if not CKPT_STAGE2_WARMSTART.exists():\n"
    "            raise FileNotFoundError(f'Warm-start ckpt missing : {CKPT_STAGE2_WARMSTART}')\n"
    "        ck = torch.load(CKPT_STAGE2_WARMSTART, map_location=DEVICE, weights_only=False)\n"
    "        diff_sd = _strip_prefixes(ck.get('diffusion_state_dict'))\n"
    "        if diff_sd is None:\n"
    "            raise RuntimeError(f'diffusion_state_dict absent from {CKPT_STAGE2_WARMSTART}')\n"
    "        info = diff.load_state_dict(diff_sd, strict=False)\n"
    "        # Audit fix #4 : warn if warm-start was trained with a different sigma_data.\n"
    "        _ws_sigma = float(ck.get('sigma_data', float('nan')))\n"
    "        if _ws_sigma == _ws_sigma:\n"
    "            _diff_pct = abs(_ws_sigma - SIGMA_DATA_NEW) / max(_ws_sigma, 1e-6) * 100.0\n"
    "            tag = 'OK' if _diff_pct < 5.0 else 'WARN'\n"
    "            print(f'  [build_stage2] warm-start sigma_data : ckpt={_ws_sigma:.4f}  '\n"
    "                  f'override={SIGMA_DATA_NEW:.4f}  delta={_diff_pct:.2f}%  [{tag}]')\n"
    "        else:\n"
    "            print(f'  [build_stage2] warm-start ckpt has no sigma_data key '\n"
    "                  f'(legacy ckpt). Forcing {SIGMA_DATA_NEW:.4f}.')\n"
    "        print(f'  [build_stage2] warm-start loaded | missing={len(info.missing_keys)} '\n"
    "              f'unexpected={len(info.unexpected_keys)}')\n"
    "        del ck, diff_sd\n"
)
src6 = ''.join(nb['cells'][cell6_idx]['source'])
n6 = src6.count(old_6)
print(f'Cell 6 old_6 matched : {n6} (expect 1)')
assert n6 == 1
src6 = src6.replace(old_6, new_6, 1)
try:
    compile(src6, 'cell-6', 'exec')
except SyntaxError as e:
    print(f'Cell 6 SYNTAX ERROR L{e.lineno}: {e.msg}')
    raise
lines6 = src6.split('\n')
nb['cells'][cell6_idx]['source'] = [l + '\n' for l in lines6[:-1]] + ([lines6[-1]] if lines6[-1] else [])

# ---- Cell 7 : resume validation + ema counter restore + reseed + drop _ema_update ----
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    if 'def _ema_update' in src and 'train_epoch_stage2_cached' in src and 'RESUME' in src:
        cell7_idx = i
        break
else:
    raise RuntimeError('Cell 7 not found')

# Replace : RESUME block with config validation + ema counter restore
old_7a = (
    "if RESUME and CKPT_LAST.exists():\n"
    "    print(f'[Cell 7] RESUME from {CKPT_LAST}')\n"
    "    _ck = torch.load(CKPT_LAST, map_location=DEVICE, weights_only=False)\n"
    "    diffusion_decoder.load_state_dict(_strip_prefixes(_ck['diffusion_state_dict']))\n"
    "    if _ck.get('diffusion_ema_state_dict') is not None:\n"
    "        diffusion_ema.load_state_dict(_strip_prefixes(_ck['diffusion_ema_state_dict']))\n"
    "    if _ck.get('optimizer_state_dict') is not None:\n"
    "        try:\n"
    "            optimizer.load_state_dict(_ck['optimizer_state_dict'])\n"
    "        except Exception as _e:\n"
    "            print(f'  [warn] optimizer resume failed : {_e}')\n"
    "    start_epoch = int(_ck.get('epoch', 0)) + 1\n"
    "    best_loss = float(_ck.get('best_loss', float('inf')))\n"
    "    training_history = list(_ck.get('training_history', []))\n"
    "    print(f'  Resumed at epoch {start_epoch}/{EPOCHS_TARGET} | best_loss={best_loss:.5f} '\n"
    "          f'| history={len(training_history)} entries')\n"
    "    del _ck\n"
)
new_7a = (
    "if RESUME and CKPT_LAST.exists():\n"
    "    print(f'[Cell 7] RESUME from {CKPT_LAST}')\n"
    "    _ck = torch.load(CKPT_LAST, map_location=DEVICE, weights_only=False)\n"
    "    # Audit fix #5 : refuse resume if critical hyperparams changed.\n"
    "    _ck_lr        = float(_ck.get('lr',        float('nan')))\n"
    "    _ck_sigma     = float(_ck.get('sigma_data', float('nan')))\n"
    "    _ck_ema_decay = float(_ck.get('ema_decay', float('nan')))\n"
    "    _drift = []\n"
    "    if _ck_lr == _ck_lr and abs(_ck_lr - LR) / max(LR, 1e-12) > 1e-3:\n"
    "        _drift.append(f'lr {_ck_lr} != {LR}')\n"
    "    if _ck_sigma == _ck_sigma and abs(_ck_sigma - SIGMA_DATA_NEW) > 1e-4:\n"
    "        _drift.append(f'sigma_data {_ck_sigma} != {SIGMA_DATA_NEW}')\n"
    "    if _ck_ema_decay == _ck_ema_decay and abs(_ck_ema_decay - EMA_DECAY) > 1e-6:\n"
    "        _drift.append(f'ema_decay {_ck_ema_decay} != {EMA_DECAY}')\n"
    "    if _drift:\n"
    "        raise RuntimeError(\n"
    "            f'[Cell 7] resume aborted -- hyperparam drift detected :\\n  '\n"
    "            + '\\n  '.join(_drift)\n"
    "            + f'\\nEither restore the old config in Cell 2 or '\n"
    "            + f'rm {CKPT_LAST} to start fresh.'\n"
    "        )\n"
    "    diffusion_decoder.load_state_dict(_strip_prefixes(_ck['diffusion_state_dict']))\n"
    "    if _ck.get('diffusion_ema_state_dict') is not None:\n"
    "        diffusion_ema.load_state_dict(_strip_prefixes(_ck['diffusion_ema_state_dict']))\n"
    "    # Audit fix #6 : restore EMA step counter (lost on resume otherwise).\n"
    "    diffusion_ema._ema_step_counter = int(_ck.get('ema_step_counter', 0))\n"
    "    if _ck.get('optimizer_state_dict') is not None:\n"
    "        try:\n"
    "            optimizer.load_state_dict(_ck['optimizer_state_dict'])\n"
    "        except Exception as _e:\n"
    "            print(f'  [warn] optimizer resume failed : {_e}')\n"
    "    start_epoch = int(_ck.get('epoch', 0)) + 1\n"
    "    best_loss = float(_ck.get('best_loss', float('inf')))\n"
    "    training_history = list(_ck.get('training_history', []))\n"
    "    print(f'  Resumed at epoch {start_epoch}/{EPOCHS_TARGET} | best_loss={best_loss:.5f} '\n"
    "          f'| history={len(training_history)} entries '\n"
    "          f'| ema_step_counter={diffusion_ema._ema_step_counter}')\n"
    "    del _ck\n"
)
src7 = ''.join(nb['cells'][cell7_idx]['source'])
n7a = src7.count(old_7a)
print(f'Cell 7 old_7a matched : {n7a} (expect 1)')
assert n7a == 1
src7 = src7.replace(old_7a, new_7a, 1)

# Drop dead _ema_update fn (8 lines).
old_7b = (
    "def _ema_update(ema_model, live_model, decay):\n"
    "    \"\"\"Karras EDM2 EMA : parameters mul+add, buffers copy.\"\"\"\n"
    "    with torch.no_grad():\n"
    "        for p_ema, p_live in zip(ema_model.parameters(), live_model.parameters()):\n"
    "            p_ema.data.mul_(decay).add_(p_live.data, alpha=1.0 - decay)\n"
    "        for b_ema, b_live in zip(ema_model.buffers(), live_model.buffers()):\n"
    "            b_ema.data.copy_(b_live.data)\n"
    "\n"
    "\n"
)
n7b = src7.count(old_7b)
print(f'Cell 7 old_7b matched : {n7b} (expect 1)')
assert n7b == 1
src7 = src7.replace(old_7b, "", 1)

# Reseed before training loop (audit fix #7) + add ema_step_counter to payload.
old_7c = (
    "print(f'\\n[Cell 7] Training from epoch {start_epoch} to {EPOCHS_TARGET}')\n"
    "\n"
    "for ep in range(start_epoch, EPOCHS_TARGET + 1):\n"
)
new_7c = (
    "# Audit fix #7 : reseed RNG before training loop (build_stage2 x2 consumed it).\n"
    "torch.manual_seed(SEED)\n"
    "if torch.cuda.is_available():\n"
    "    torch.cuda.manual_seed_all(SEED)\n"
    "print(f'\\n[Cell 7] Training from epoch {start_epoch} to {EPOCHS_TARGET}')\n"
    "\n"
    "for ep in range(start_epoch, EPOCHS_TARGET + 1):\n"
)
n7c = src7.count(old_7c)
print(f'Cell 7 old_7c matched : {n7c} (expect 1)')
assert n7c == 1
src7 = src7.replace(old_7c, new_7c, 1)

# Add ema_step_counter to payload.
old_7d = (
    "    payload = {\n"
    "        'epoch': ep,\n"
    "        'diffusion_state_dict':     diffusion_decoder.state_dict(),\n"
    "        'diffusion_ema_state_dict': diffusion_ema.state_dict(),\n"
    "        'optimizer_state_dict':     optimizer.state_dict(),\n"
    "        'best_loss':                best_loss,\n"
    "        'training_history':         training_history,\n"
    "        'sigma_data':               float(SIGMA_DATA_NEW),\n"
    "        'warm_start':               bool(WARM_START),\n"
    "        'epochs_target':            int(EPOCHS_TARGET),\n"
    "        'lr':                       float(LR),\n"
    "        'ema_decay':                float(EMA_DECAY),\n"
    "    }\n"
)
new_7d = (
    "    payload = {\n"
    "        'epoch': ep,\n"
    "        'diffusion_state_dict':     diffusion_decoder.state_dict(),\n"
    "        'diffusion_ema_state_dict': diffusion_ema.state_dict(),\n"
    "        'optimizer_state_dict':     optimizer.state_dict(),\n"
    "        'best_loss':                best_loss,\n"
    "        'training_history':         training_history,\n"
    "        'sigma_data':               float(SIGMA_DATA_NEW),\n"
    "        'warm_start':               bool(WARM_START),\n"
    "        'epochs_target':            int(EPOCHS_TARGET),\n"
    "        'lr':                       float(LR),\n"
    "        'ema_decay':                float(EMA_DECAY),\n"
    "        'ema_step_counter':         int(getattr(diffusion_ema, '_ema_step_counter', 0)),\n"
    "    }\n"
)
n7d = src7.count(old_7d)
print(f'Cell 7 old_7d matched : {n7d} (expect 1)')
assert n7d == 1
src7 = src7.replace(old_7d, new_7d, 1)

try:
    compile(src7, 'cell-7', 'exec')
except SyntaxError as e:
    print(f'Cell 7 SYNTAX ERROR L{e.lineno}: {e.msg}')
    raise
lines7 = src7.split('\n')
nb['cells'][cell7_idx]['source'] = [l + '\n' for l in lines7[:-1]] + ([lines7[-1]] if lines7[-1] else [])

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'phase7_stage2_B_finetune.ipynb : 4 fixes applied')


# ============================================================================
# phase6_dualpath_final_validation.ipynb
# ============================================================================
nb2_path = Path('path_c_plus/scripts/phase6_dualpath_final_validation.ipynb')
nb2 = json.loads(nb2_path.read_text(encoding='utf-8'))

# ---- Cell 2 : add STAGE2_CHECKPOINT override (uses globals()) ----
for i, c in enumerate(nb2['cells']):
    src = ''.join(c.get('source', []))
    if 'CKPT_STAGE2    = ORACLE_9N / ' in src:
        cell2_idx = i
        break
else:
    raise RuntimeError('Phase6 Cell 2 not found')

old_2 = "CKPT_STAGE2    = ORACLE_9N / 'epoch_last.pth'\n"
new_2 = (
    "# Audit fix : allow override via globals() so phase7_stage2_B_finetune ckpt can be plugged.\n"
    "CKPT_STAGE2    = Path(str(globals().get('STAGE2_CHECKPOINT',\n"
    "                                          ORACLE_9N / 'epoch_last.pth')))\n"
)
src2 = ''.join(nb2['cells'][cell2_idx]['source'])
n_2 = src2.count(old_2)
print(f'Phase6 Cell 2 matched : {n_2} (expect 1)')
assert n_2 == 1
src2 = src2.replace(old_2, new_2, 1)
try:
    compile(src2, 'p6-cell-2', 'exec')
except SyntaxError as e:
    print(f'Phase6 Cell 2 SYNTAX ERROR L{e.lineno}: {e.msg}')
    raise
lines2 = src2.split('\n')
nb2['cells'][cell2_idx]['source'] = [l + '\n' for l in lines2[:-1]] + ([lines2[-1]] if lines2[-1] else [])

# ---- Cell 3 : add test_dataloader ----
for i, c in enumerate(nb2['cells']):
    src = ''.join(c.get('source', []))
    if "val_dataloader   = _DataLoader(val_dataset, shuffle=False" in src:
        cell3_idx = i
        break
else:
    raise RuntimeError('Phase6 Cell 3 not found')

old_3 = (
    "train_dataset = pipeline.build_sequence_dataset(split='train', seq_len=SEQ_LEN,\n"
    "                                                 stride=int(CONFIG.data.stride), as_torch=True)\n"
    "val_dataset   = pipeline.build_sequence_dataset(split='val',   seq_len=SEQ_LEN,\n"
    "                                                 stride=int(CONFIG.data.stride), as_torch=True)\n"
)
new_3 = (
    "train_dataset = pipeline.build_sequence_dataset(split='train', seq_len=SEQ_LEN,\n"
    "                                                 stride=int(CONFIG.data.stride), as_torch=True)\n"
    "val_dataset   = pipeline.build_sequence_dataset(split='val',   seq_len=SEQ_LEN,\n"
    "                                                 stride=int(CONFIG.data.stride), as_torch=True)\n"
    "# Audit fix : test split (2012-2013) -- the BS30 protocol's actual eval window.\n"
    "test_dataset  = pipeline.build_sequence_dataset(split='test',  seq_len=SEQ_LEN,\n"
    "                                                 stride=int(CONFIG.data.stride), as_torch=True)\n"
)
src3 = ''.join(nb2['cells'][cell3_idx]['source'])
n_3 = src3.count(old_3)
print(f'Phase6 Cell 3 dataset block matched : {n_3} (expect 1)')
assert n_3 == 1
src3 = src3.replace(old_3, new_3, 1)

old_3b = "val_dataloader   = _DataLoader(val_dataset, shuffle=False, **_loader_kw)\n"
new_3b = (
    "val_dataloader   = _DataLoader(val_dataset,  shuffle=False, **_loader_kw)\n"
    "test_dataloader  = _DataLoader(test_dataset, shuffle=False, **_loader_kw)\n"
)
n_3b = src3.count(old_3b)
print(f'Phase6 Cell 3 dataloader matched : {n_3b} (expect 1)')
assert n_3b == 1
src3 = src3.replace(old_3b, new_3b, 1)

# Update the summary print to mention test.
old_3c = "print(f'[Cell 3] train={_n_train} samples  val={_n_val} samples')\n"
new_3c = (
    "_n_test = len(test_dataset) if hasattr(test_dataset, '__len__') else '?'\n"
    "print(f'[Cell 3] train={_n_train} samples  val={_n_val} samples  test={_n_test} samples')\n"
)
n_3c = src3.count(old_3c)
print(f'Phase6 Cell 3 print matched : {n_3c} (expect 1)')
assert n_3c == 1
src3 = src3.replace(old_3c, new_3c, 1)

try:
    compile(src3, 'p6-cell-3', 'exec')
except SyntaxError as e:
    print(f'Phase6 Cell 3 SYNTAX ERROR L{e.lineno}: {e.msg}')
    raise
lines3 = src3.split('\n')
nb2['cells'][cell3_idx]['source'] = [l + '\n' for l in lines3[:-1]] + ([lines3[-1]] if lines3[-1] else [])

# ---- Cell 4 : EMA fallback + sigma_data from ckpt ----
for i, c in enumerate(nb2['cells']):
    src = ''.join(c.get('source', []))
    if "_diff_sd = _strip_prefixes(ck_s2.get('diffusion_state_dict'))" in src:
        cell4_idx = i
        break
else:
    raise RuntimeError('Phase6 Cell 4 not found')

old_4a = (
    "_diff_sd = _strip_prefixes(ck_s2.get('diffusion_state_dict'))\n"
    "if _diff_sd is None:\n"
    "    raise RuntimeError(\n"
    "        f'diffusion_state_dict absent from {CKPT_STAGE2}. Stage 2 ckpt must be '\n"
    "        f'the epoch_last.pth from Path C+ Option C 9-node training.'\n"
    "    )\n"
    "_missing, _unexpected = diffusion_decoder.load_state_dict(_diff_sd, strict=False)\n"
)
new_4a = (
    "# Audit fix #2 : prefer EMA weights (diffusion convention) ; fall back to live.\n"
    "_ema_sd  = _strip_prefixes(ck_s2.get('diffusion_ema_state_dict'))\n"
    "_live_sd = _strip_prefixes(ck_s2.get('diffusion_state_dict'))\n"
    "if _ema_sd is not None:\n"
    "    _diff_sd = _ema_sd\n"
    "    print(f'[Cell 4] using EMA weights (diffusion_ema_state_dict)')\n"
    "elif _live_sd is not None:\n"
    "    _diff_sd = _live_sd\n"
    "    print(f'[Cell 4] EMA absent : using LIVE weights (diffusion_state_dict)')\n"
    "else:\n"
    "    raise RuntimeError(\n"
    "        f'Neither diffusion_ema_state_dict nor diffusion_state_dict present in '\n"
    "        f'{CKPT_STAGE2}.'\n"
    "    )\n"
    "_missing, _unexpected = diffusion_decoder.load_state_dict(_diff_sd, strict=False)\n"
)
src4 = ''.join(nb2['cells'][cell4_idx]['source'])
n_4a = src4.count(old_4a)
print(f'Phase6 Cell 4 EMA load matched : {n_4a} (expect 1)')
assert n_4a == 1
src4 = src4.replace(old_4a, new_4a, 1)

old_4b = (
    "# Override sigma_data for phase6_dualpath regime (mu_total).\n"
    "_SIGMA_DATA_CKPT = float(diffusion_decoder.edm_config.sigma_data)\n"
    "diffusion_decoder.edm_config.sigma_data = float(SIGMA_DATA_NEW)\n"
    "print(f'[Cell 4] sigma_data : ckpt={_SIGMA_DATA_CKPT:.5f} -> new={SIGMA_DATA_NEW:.5f}')\n"
)
new_4b = (
    "# Audit fix #3 : prefer sigma_data recorded in the ckpt payload over the hard-coded constant.\n"
    "# This guarantees we sample with the same sigma_data the model was preconditioned for.\n"
    "_SIGMA_DATA_CKPT = float(ck_s2.get('sigma_data', diffusion_decoder.edm_config.sigma_data))\n"
    "_sigma_to_use    = _SIGMA_DATA_CKPT if _SIGMA_DATA_CKPT == _SIGMA_DATA_CKPT else float(SIGMA_DATA_NEW)\n"
    "diffusion_decoder.edm_config.sigma_data = float(_sigma_to_use)\n"
    "print(f'[Cell 4] sigma_data : ckpt={_SIGMA_DATA_CKPT:.5f}  '\n"
    "      f'override-fallback={SIGMA_DATA_NEW:.5f}  -> using {_sigma_to_use:.5f}')\n"
)
n_4b = src4.count(old_4b)
print(f'Phase6 Cell 4 sigma_data matched : {n_4b} (expect 1)')
assert n_4b == 1
src4 = src4.replace(old_4b, new_4b, 1)

try:
    compile(src4, 'p6-cell-4', 'exec')
except SyntaxError as e:
    print(f'Phase6 Cell 4 SYNTAX ERROR L{e.lineno}: {e.msg}')
    raise
lines4 = src4.split('\n')
nb2['cells'][cell4_idx]['source'] = [l + '\n' for l in lines4[:-1]] + ([lines4[-1]] if lines4[-1] else [])

# ---- Cell 6 : iterate test_dataloader ----
for i, c in enumerate(nb2['cells']):
    src = ''.join(c.get('source', []))
    if "FINAL_VALIDATION protocol on phase6_dualpath" in src:
        cell6_p_idx = i
        break
else:
    raise RuntimeError('Phase6 Cell 6 not found')

old_6p = "    for converted_batches in iterate_batches(val_dataloader, builder, DEVICE):\n"
new_6p = (
    "    # Audit fix #3 : eval on TEST split (2012-2013), same as non-causal 0.815 baseline.\n"
    "    for converted_batches in iterate_batches(test_dataloader, builder, DEVICE):\n"
)
src6p = ''.join(nb2['cells'][cell6_p_idx]['source'])
n_6p = src6p.count(old_6p)
print(f'Phase6 Cell 6 val->test matched : {n_6p} (expect 1)')
assert n_6p == 1
src6p = src6p.replace(old_6p, new_6p, 1)
try:
    compile(src6p, 'p6-cell-6', 'exec')
except SyntaxError as e:
    print(f'Phase6 Cell 6 SYNTAX ERROR L{e.lineno}: {e.msg}')
    raise
lines6p = src6p.split('\n')
nb2['cells'][cell6_p_idx]['source'] = [l + '\n' for l in lines6p[:-1]] + ([lines6p[-1]] if lines6p[-1] else [])

# ---- Cell 8 : iterate test_dataloader ----
for i, c in enumerate(nb2['cells']):
    src = ''.join(c.get('source', []))
    if 'run_aligned_eval' in src and 'val_dataloader' in src:
        cell8_idx = i
        break
else:
    raise RuntimeError('Phase6 Cell 8 not found')

old_8 = "    for _conv in iterate_batches(val_dataloader, builder, DEVICE):\n"
new_8 = (
    "    # Audit fix : test split (same as Cell 6).\n"
    "    for _conv in iterate_batches(test_dataloader, builder, DEVICE):\n"
)
src8 = ''.join(nb2['cells'][cell8_idx]['source'])
n_8 = src8.count(old_8)
print(f'Phase6 Cell 8 val->test matched : {n_8} (expect 1)')
assert n_8 == 1
src8 = src8.replace(old_8, new_8, 1)
try:
    compile(src8, 'p6-cell-8', 'exec')
except SyntaxError as e:
    print(f'Phase6 Cell 8 SYNTAX ERROR L{e.lineno}: {e.msg}')
    raise
lines8 = src8.split('\n')
nb2['cells'][cell8_idx]['source'] = [l + '\n' for l in lines8[:-1]] + ([lines8[-1]] if lines8[-1] else [])

nb2_path.write_text(json.dumps(nb2, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'phase6_dualpath_final_validation.ipynb : 5 fixes applied')
