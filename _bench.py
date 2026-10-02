import time
from src.sweep_all import _worker

for ds in ['iris', 'segment']:
    t0 = time.perf_counter()
    rows = _worker((ds, 'P2_Full', 'clean', 0.0, 0))
    t1 = time.perf_counter()
    acc = rows[-1]['ACC'] if rows else 'N/A'
    print(f'{ds} P2_Full: {t1-t0:.1f}s, {len(rows)} rows, ACC@30%={acc}')

for m in ['BADGE', 'Random', 'Entropy']:
    t0 = time.perf_counter()
    rows = _worker(('segment', m, 'clean', 0.0, 0))
    t1 = time.perf_counter()
    print(f'segment {m}: {t1-t0:.1f}s')
