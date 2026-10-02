import pandas as pd
import os

for name, target in [('segment', 0.787), ('glass', 0.479), ('noise', 0.726)]:
    f = f'results/rescue_tuning_{name}.csv'
    if not os.path.exists(f):
        print(f'{name}: NOT FOUND')
        continue
    df = pd.read_csv(f)
    print(f'=== {name} ({len(df)} rows) ===')
    print('columns:', list(df.columns))
    acc_col = 'ACC' if 'ACC' in df.columns else ('acc' if 'acc' in df.columns else None)
    if acc_col is None:
        print(df.head(3))
        print()
        continue
    g = df.groupby([c for c in df.columns if c in ('P2_LAM','lam','alpha_cross_frac','ROBUST_KAPPA_BASE','kappa','use_block')])[acc_col].mean().reset_index()
    g = g.sort_values(acc_col, ascending=False)
    print(g.head(8).to_string(index=False))
    best = g[acc_col].max()
    print(f'>>> best={best:.3f}  target(Random/CoreSet/BADGE)={target}  rescued={best>=target}')
    print()
