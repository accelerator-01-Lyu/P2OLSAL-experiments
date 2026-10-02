# -*- coding: utf-8 -*-
"""Rebuild results/table1_main_results.csv with the canonical equal-weight
caliber: seed mean within each dataset, then ONE equal vote per dataset.

Layout (16 methods):
  rank, method, ACC_mean, NMI_mean, ARI_mean, then ACC_<dataset> x13.
All at budget=10%, eta=0.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import equal_weight as ew

OUT = os.path.join(ew.RESULTS, 'table1_main_results.csv')


def main():
    df = ew.load_merged()
    acc = ew.per_dataset(df, 'ACC')
    nmi = ew.per_dataset(df, 'NMI')
    ari = ew.per_dataset(df, 'ARI')
    acc_mean = acc.mean(axis=1)
    rows = []
    for m in ew.METHODS_16:
        row = {'method': m,
               'ACC_mean': round(float(acc_mean[m]), 6),
               'NMI_mean': round(float(nmi.loc[m].mean()), 6),
               'ARI_mean': round(float(ari.loc[m].mean()), 6)}
        for ds in ew.DATASETS:
            row[f'ACC_{ds}'] = round(float(acc.loc[m, ds]), 6)
        rows.append(row)
    t = pd.DataFrame(rows)
    t.insert(0, 'rank',
             t['ACC_mean'].rank(ascending=False, method='min').astype(int))
    t = t.sort_values('rank').reset_index(drop=True)
    t.to_csv(OUT, index=False)
    print('wrote', OUT, t.shape)
    print(t[['rank', 'method', 'ACC_mean', 'NMI_mean', 'ARI_mean']]
          .to_string(index=False))
    p2 = t[t.method == 'P2_Full'].iloc[0]
    print('\nP2_Full: ACC eq-mean=%.4f rank=%d' % (p2['ACC_mean'], p2['rank']))


if __name__ == '__main__':
    main()
