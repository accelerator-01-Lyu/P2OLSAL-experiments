# -*- coding: utf-8 -*-
"""Canonical dataset-equal-weight aggregation for ALL cross-dataset summaries.

The merged sweep has unequal seeds per dataset (30 seeds for n<=2500, 8 seeds
for the large sets). Pooling every ROW therefore over-weights the large
datasets and gives the WRONG headline mean. The ONLY correct paper caliber is:

    1. average over seeds WITHIN each (method, dataset)  -> per-dataset value
    2. average the 13 per-dataset values with EQUAL weight (one vote each)

Every cross-dataset number (Table 1, abstract, noise curves, waterfall,
regime map, conclusions) MUST go through equal_weight_mean() / per_dataset().
Do not use a plain df['ACC'].mean() over rows.
"""
import os
import pandas as pd

RESULTS = r'D:\P2OLSAL_FSS\experiments\results'
MERGED = os.path.join(RESULTS, 'all_results_merged.csv')

DATASETS = ['iris', 'wine', 'seeds', 'glass', 'ecoli', 'segment', 'balance',
            'aggregation', 'compound', 'letter', 'shuttle', 'usps', 'fashion']
METHODS_16 = ['P2_Full', 'P2_NoRobust', 'P2_NoExploration', 'P2_NoRedundancy',
              'P2_NoFMIS', 'P2_NoBlock', 'BADGE', 'CoreSet', 'BALD', 'QBC',
              'Entropy', 'Random', 'SSFCM', 'CEFCM', 'GRFCM', 'PLCFCMPassive']


def load_merged(path=MERGED):
    return pd.read_csv(path, low_memory=False)


def per_dataset(df, metric='ACC', budget=0.1, eta=0.0):
    """Step 1: mean over seeds -> DataFrame indexed method, columns=datasets."""
    d = df[(df['budget_frac'] == budget) & (df['eta'] == eta)]
    m = (d.groupby(['method', 'dataset'])[metric]
           .mean().unstack('dataset'))
    missing = [x for x in DATASETS if x not in m.columns]
    if missing:
        raise ValueError(f'missing dataset columns for {metric}: {missing}')
    m = m.reindex(columns=DATASETS)
    if m[DATASETS].isna().any().any():
        na = m[DATASETS].isna().stack()
        raise ValueError('NaN per-dataset cells: '
                         + str(na[na].index.tolist()[:10]))
    return m


def equal_weight_mean(df=None, metric='ACC', budget=0.1, eta=0.0):
    """Step 2: one equal vote per dataset -> Series indexed by method."""
    if df is None:
        df = load_merged()
    return per_dataset(df, metric, budget, eta).mean(axis=1)


def ranked(metric='ACC', budget=0.1, eta=0.0, df=None, ascending=False):
    """Return DataFrame method, value, rank (rank 1 = best for ACC)."""
    s = equal_weight_mean(df, metric, budget, eta).sort_values(ascending=ascending)
    out = s.reset_index()
    out.columns = ['method', f'{metric}_eqmean']
    out['rank'] = range(1, len(out) + 1)
    return out


if __name__ == '__main__':
    df = load_merged()
    r = ranked('ACC', df=df)
    print(r.to_string(index=False))
    print('P2_Full ACC@10%% eq-mean = %.4f  rank %d'
          % (equal_weight_mean(df, 'ACC').loc['P2_Full'],
             list(r['method']).index('P2_Full') + 1))
    for e in [0.0, 0.1, 0.2, 0.3]:
        s = equal_weight_mean(df, 'ACC', eta=e)
        print('eta', e, {m: round(100 * s[m], 2)
                         for m in ['P2_Full', 'BADGE', 'Random', 'QBC']})
