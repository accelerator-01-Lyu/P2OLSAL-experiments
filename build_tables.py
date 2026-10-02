"""
统一等权聚合：先每个数据集对种子平均，再跨13数据集平均（数据集等权）。
产出 table1_main_results.csv（@10%, eta=0）与 noise_equalweight.csv（四档噪声）。
所有跨集汇总一律使用 equal_weight_mean，禁止按行直接平均（种子数不等会偏）。
"""
import os, json
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, 'results')
df = pd.read_csv(os.path.join(RES, 'all_results_merged.csv'), low_memory=False)
# 丢弃错误行
df = df[df['error'].isna() | (df['error'].astype(str) == '')].copy()
for c in ['ACC', 'NMI', 'ARI']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

DATASETS = sorted(df['dataset'].unique())
METHODS = list(dict.fromkeys([
    'P2_Full', 'BADGE', 'CoreSet', 'BALD', 'QBC', 'Entropy', 'Random',
    'P2_NoRobust', 'P2_NoRedundancy', 'P2_NoBlock', 'P2_NoExploration', 'P2_NoFMIS',
    'SSFCM', 'CEFCM', 'GRFCM', 'PLCFCMPassive']))
METHODS = [m for m in METHODS if m in set(df['method'])]


def per_dataset_mean(d, metric):
    """每数据集对种子平均，返回 DataFrame(index=dataset, columns=method)"""
    return d.groupby(['method', 'dataset'])[metric].mean().unstack('dataset')


def equal_weight_mean(d, metric):
    """数据集等权平均：先每集平均，再跨集平均"""
    return per_dataset_mean(d, metric).mean(axis=1)


# ===== 主表 @10% eta=0 =====
main = df[(df['budget_frac'] == 0.1) & (df['eta'] == 0)].copy()
acc = per_dataset_mean(main, 'ACC')
nmi = per_dataset_mean(main, 'NMI')
ari = per_dataset_mean(main, 'ARI')
summary = pd.DataFrame({
    'ACC_mean': acc.mean(axis=1),
    'NMI_mean': nmi.mean(axis=1),
    'ARI_mean': ari.mean(axis=1),
}).loc[METHODS]
summary = summary.sort_values('ACC_mean', ascending=False)

rows = []
for rank, (m, r) in enumerate(summary.iterrows(), 1):
    row = {'rank': rank, 'method': m,
           'ACC_mean': round(r['ACC_mean'], 6),
           'NMI_mean': round(r['NMI_mean'], 6),
           'ARI_mean': round(r['ARI_mean'], 6)}
    for ds in DATASETS:
        row[f'ACC_{ds}'] = round(acc.loc[m, ds], 6) if ds in acc.columns else None
    rows.append(row)
table1 = pd.DataFrame(rows)
table1.to_csv(os.path.join(RES, 'table1_main_results.csv'), index=False)

# ===== 噪声四档等权 =====
noise = df[df['budget_frac'] == 0.1].copy()
noise_tab = noise.groupby(['method', 'dataset', 'eta'])['ACC'].mean().groupby(['method', 'eta']).mean().unstack('eta')
noise_tab = noise_tab.loc[[m for m in METHODS if m in noise_tab.index]]
noise_tab.to_csv(os.path.join(RES, 'noise_equalweight.csv'))

# ===== 打印供手稿使用 =====
print('=== 主表 数据集等权 ACC/NMI/ARI@10% eta=0 ===')
for rank, (m, r) in enumerate(summary.iterrows(), 1):
    print(f'{rank:2d}. {m:16s} ACC {r.ACC_mean*100:6.3f}  NMI {r.NMI_mean*100:6.3f}  ARI {r.ARI_mean*100:6.3f}')

print('\n=== 噪声等权 ACC (budget=10%) ===')
etas = sorted(noise_tab.columns)
print('method            ' + ''.join(f'eta={e:<6}' for e in etas) + ' 退化%')
for m in ['P2_Full', 'BADGE', 'Random', 'CoreSet', 'QBC', 'Entropy', 'BALD', 'PLCFCMPassive']:
    if m not in noise_tab.index:
        continue
    vals = noise_tab.loc[m]
    deg = (vals[etas[0]] - vals[etas[-1]]) / vals[etas[0]] * 100
    print(f'{m:16s} ' + ''.join(f'{vals[e]*100:6.2f}  ' for e in etas) + f' {deg:4.1f}%')

# 存 JSON 供后续
out = {
    'main_rank': [{'rank': i, 'method': m,
                   'ACC': round(r.ACC_mean, 5), 'NMI': round(r.NMI_mean, 5), 'ARI': round(r.ARI_mean, 5)}
                  for i, (m, r) in enumerate(summary.iterrows(), 1)],
    'noise': {m: {str(e): round(float(noise_tab.loc[m, e]), 5) for e in etas}
              for m in noise_tab.index},
}
with open(os.path.join(RES, '_summary_for_manuscript.json'), 'w') as f:
    json.dump(out, f, indent=1)
print('\nwritten table1_main_results.csv, noise_equalweight.csv')
