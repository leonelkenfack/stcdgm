"""Fix Cell 9 indentation broken by previous patch."""
import json
import sys
from pathlib import Path

nb_path = Path('path_c_plus/scripts/phase6_dualpath_training.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))

for c in nb['cells']:
    if c.get('id') != 'cell-9-phase3':
        continue

    src = ''.join(c['source'])

    # Step 1: Remove ALL existing _use_amp_phase_III lines I might have wrongly inserted
    lines = src.split('\n')
    cleaned = []
    skip_next_blank = False
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith('_use_amp_phase_III = (USE_AMP'):
            continue
        if stripped.startswith("print(f'  [Phase III] AMP enabled"):
            skip_next_blank = True
            continue
        if skip_next_blank and stripped == '':
            skip_next_blank = False
            continue
        skip_next_blank = False
        cleaned.append(ln)

    src = '\n'.join(cleaned)

    # Step 2: Insert ONCE at the unique top-level location
    # The unique marker is "_start_ep_III     = 1" (with that exact spacing)
    marker = '_start_ep_III     = 1'
    n_match = src.count(marker)
    print(f'marker "{marker}" found {n_match} times')
    if n_match != 1:
        print('ABORT: expect exactly 1 match')
        sys.exit(1)

    new_block = (
        "_start_ep_III     = 1\n"
        "_use_amp_phase_III = (USE_AMP and DEVICE.type == 'cuda' and not FORCE_FP32_DUALPATH)\n"
        "print(f'  [Phase III] AMP enabled = {_use_amp_phase_III}')"
    )
    src = src.replace(marker, new_block, 1)

    # Validate syntax
    try:
        compile(src, 'cell-9', 'exec')
        print('Syntax OK')
    except SyntaxError as e:
        print(f'SYNTAX ERROR at line {e.lineno}: {e.msg}')
        for i, l in enumerate(src.split('\n'), 1):
            if abs(i - e.lineno) <= 4:
                print(f'  {i:3d}: {l}')
        sys.exit(1)

    lines2 = src.split('\n')
    c['source'] = [l + '\n' for l in lines2[:-1]]
    if lines2[-1]:
        c['source'].append(lines2[-1])
    print('Cell 9 fixed')
    break

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print('Saved')
