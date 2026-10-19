"""Read-only, privacy-preserving diagnostic. Run beside optimizer.py."""
from pathlib import Path
import hashlib
import importlib.util
import random
import sys
from collections import Counter

root = Path(__file__).resolve().parent
source = root / 'optimizer.py'
if not source.is_file():
    raise SystemExit('ERROR: optimizer.py not found beside this script.')
spec = importlib.util.spec_from_file_location('local_crew_optimizer', source)
opt = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = opt
spec.loader.exec_module(opt)
print('Optimizer SHA256:', hashlib.sha256(source.read_bytes()).hexdigest()[:16])
print('Data directory:', opt.DATA.resolve())
missions, pilots, availability = opt.load_data()
print('Missions:', len(missions), 'Pilots:', len(pilots), 'Tours:', len(opt.build_tours(missions)))

# Check overlapping aircraft flights without printing confidential identifiers.
conflicts = 0
for registration in {m.fixed_aircraft for m in missions}:
    seq = sorted((m for m in missions if m.fixed_aircraft == registration), key=lambda m: m.departure)
    for index, a in enumerate(seq):
        for b in seq[index + 1:]:
            if b.departure >= a.arrival:
                break
            conflicts += 1
print('Aircraft overlap pairs:', conflicts)
if conflicts:
    print('STOP: correct fixed-aircraft flight conflicts before evaluating crew feasibility.')
    raise SystemExit(2)

feasible, captains, fos = opt.precompute(missions, pilots, availability)
reasons = Counter()
valid = 0
for i in range(50):
    rng = random.Random(42 + i * 101)
    genes = [rng.random() for _ in range(max(2, len(missions) * 2))]
    try:
        decoded = opt.decode(genes, missions, feasible, captains, fos, availability, pilots,
                             failure_stats=reasons)
    except TypeError as exc:
        if 'failure_stats' not in str(exc):
            raise
        decoded = opt.decode(genes, missions, feasible, captains, fos, availability, pilots)
    valid += decoded is not None
print('Valid schedules out of 50:', valid)
print('Rejection reasons (counts only):')
for (aircraft, reason, detail), count in reasons.most_common(8):
    print(f'  reason={reason} detail={detail} count={count}')
if not reasons and valid == 0:
    print('No decoder rejection details available in this optimizer version.')
