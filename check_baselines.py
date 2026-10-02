"""Check baseline scores from main sweep for comparison."""
import csv, collections, glob, os
import numpy as np

files = sorted(glob.glob('results/master_sweep_*.csv'))
latest = files[-1]
print(f'Using: {os.path.basename(latest)}')

rows = []
with open(latest) as f:
    for r in csv.DictReader(f):
        if r.get('error'):
            continue
        if r['dataset'] in ('glass', 'ecoli', 'segment') and r['eta'] == '0.0':
            if r['method'] in ('P2_Full', 'BADGE', 'CoreSet', 'QBC', 'Random', 'BALD', 'Entropy'):
                rows.append(r)

by = collections.defaultdict(list)
for r in rows:
    key = (r['dataset'], r['method'], r['budget_frac'])
    by[key].append(float(r['ACC']))

header = f'{"dataset":>10s} {"method":>10s} {"bud":>5s} {"meanACC":>8s} {"n":>3s}'
print(header)
print('-' * 45)
for ds in ('glass', 'ecoli', 'segment'):
    for m in ('P2_Full', 'BADGE', 'CoreSet', 'QBC', 'Random', 'BALD', 'Entropy'):
        for b in ('0.1', '0.10', '0.2', '0.20'):
            vals = by.get((ds, m, b), [])
            if vals:
                print(f'{ds:>10s} {m:>10s} {b:>5s} {np.mean(vals):8.4f} {len(vals):3d}')
    print()
