"""Fix Cell 9 line 50 indentation (best_rmse_total inside if RESUME block)."""
import json
import sys
from pathlib import Path

nb_path = Path('path_c_plus/scripts/phase6_dualpath_training.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))

for c in nb['cells']:
    if c.get('id') != 'cell-9-phase3':
        continue

    src = ''.join(c['source'])

    # The broken line: `best_rmse_total   = float(_ck_III.get('best_rmse_total', float('inf')))`
    # Should be indented 4 spaces (inside `if RESUME and CKPT_PHASE_III_LAST.exists():`)
    bad = "\nbest_rmse_total   = float(_ck_III.get('best_rmse_total', float('inf')))\n"
    good = "\n    best_rmse_total   = float(_ck_III.get('best_rmse_total', float('inf')))\n"
    n = src.count(bad)
    print(f'Bad line found {n} times')
    if n != 1:
        print('ABORT')
        sys.exit(1)
    src = src.replace(bad, good, 1)

    # Compile check
    try:
        compile(src, 'cell-9', 'exec')
        print('Cell 9 syntax OK now')
    except SyntaxError as e:
        print(f'STILL BROKEN at line {e.lineno}: {e.msg}')
        for i, l in enumerate(src.split('\n'), 1):
            if abs(i - e.lineno) <= 4:
                print(f'  {i:3d}: {l}')
        sys.exit(1)

    lines = src.split('\n')
    c['source'] = [l + '\n' for l in lines[:-1]]
    if lines[-1]:
        c['source'].append(lines[-1])
    print('Fixed')
    break

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print('Saved')
