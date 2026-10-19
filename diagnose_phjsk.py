"""Local, read-only diagnostic for the PHJSK crew-relief failure.
Place beside optimizer.py and run: python diagnose_phjsk.py
Does not print names, mission IDs, dates, routes, or pilot availability.
"""
from collections import Counter
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('crew_optimizer', ROOT / 'optimizer.py')
opt = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = opt
spec.loader.exec_module(opt)

missions, pilots, avail = opt.load_data()
feasible, caps, fos = opt.precompute(missions, pilots, avail)
tours = opt.build_tours(missions)
selected = [(reg, tour) for reg, tour in tours if reg.replace('-', '').upper() == 'PHJSK']
print('Total missions:', len(missions), '| Tours for target aircraft:', len(selected))
if not selected:
    print('No target aircraft found. Check the aircraft registration in the log.')
    raise SystemExit(1)

for number, (reg, tour) in enumerate(selected, 1):
    print(f'\nTarget tour {number}: sectors={len(tour)}')
    for index, mission in enumerate(tour, 1):
        cs = feasible[mission.id]['CAPT']
        fs = feasible[mission.id]['FO']
        compatible = [(c, f) for c in cs for f in fs if pilots[c].experienced or pilots[f].experienced]
        ftl_single = sum(opt.ftl_ok_for_missions([mission], pilots[c].home_base)
                         and opt.ftl_ok_for_missions([mission], pilots[f].home_base)
                         for c, f in compatible)
        print(f'  sector={index} available_capt={len(cs)} available_fo={len(fs)} '
              f'experience_pairs={len(compatible)} individually_ftl_legal_pairs={ftl_single}')
    failures = {}
    outcome = opt.decode([0.0] * max(2, len(tour) * 2), tour, feasible, caps, fos, avail, pilots,
                         failure_stats=failures)
    print('  Isolated tour feasible:', outcome is not None)
    print('  Isolated rejection categories:', [(key[1], key[2]) for key in failures])

# Compare with prefix tours in normal decoder ordering; does not expose source data.
for stop in range(1, len(tours) + 1):
    prefix = [m for _, t in tours[:stop] for m in t]
    if any(reg.replace('-', '').upper() == 'PHJSK' for reg, _ in tours[:stop]):
        failures = {}
        result = opt.decode([0.0] * max(2, len(prefix) * 2), prefix, feasible, caps, fos, avail, pilots,
                            failure_stats=failures)
        print('\nPrefix through target tour:', stop, '| feasible:', result is not None)
        print('Prefix rejection categories:', [(k[1], k[2]) for k in failures])
        break
