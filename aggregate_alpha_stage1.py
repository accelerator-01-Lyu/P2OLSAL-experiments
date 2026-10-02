"""Aggregate stage-1 A/B/C, paired-seed adoption tests, Welch/PC diagnostics."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys, glob
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, r'D:\P2OLSAL_FSS\experiments')

R = r'D:\P2OLSAL_FSS\experiments\results'
SMALL = ['iris','wine','seeds','glass','ecoli','segment','balance','aggregation','compound']
LARGE = ['letter','shuttle','usps','fashion']
ALL = SMALL + LARGE
NSEED = {**{d:30 for d in SMALL}, **{d:8 for d in LARGE}}
BUD = [0.05,0.10,0.15,0.20]
NI = -0.005  # non-inferiority margin

main = pd.read_csv(os.path.join(R,'all_results_merged.csv'), low_memory=False)
A = main[(main.method=='P2_Full')&(main.eta==0)&
         (main.budget_frac.isin(BUD))].copy()
A['scheme']='A'
frames=[]
for f in glob.glob(os.path.join(R,'_alpha_stage1_*.csv')):
    frames.append(pd.read_csv(f, low_memory=False))
BC = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
print('B/C job-rows:', len(BC), 'datasets:', sorted(BC.dataset.unique()) if len(BC) else None)

# long ACC comparison (A + B + C)
cols=['scheme','dataset','eta','seed','budget_frac','ACC','NMI','ARI','sel_time']
cmp = pd.concat([A[cols], BC[cols]], ignore_index=True)
cmp.to_csv(os.path.join(R,'alpha_scheme_comparison.csv'), index=False, encoding='utf-8')

# trace means per scheme/dataset (B,C), structural for A
tk=['alpha','gamma_raw','d_eff','lambda2','r_within','r_cross','r_W','p','w_B','PC','H_norm']
if len(BC):
    trm = BC.groupby(['scheme','dataset'])[tk].mean().reset_index()
    trm.to_csv(os.path.join(R,'alpha_scheme_tracemeans.csv'), index=False, encoding='utf-8')

def paired(ds, sx, sy='A', bf=0.10):
    x = cmp[(cmp.dataset==ds)&(cmp.scheme==sx)&(abs(cmp.budget_frac-bf)<1e-9)]
    y = cmp[(cmp.dataset==ds)&(cmp.scheme==sy)&(abs(cmp.budget_frac-bf)<1e-9)]
    j = x.merge(y, on='seed', suffixes=('_x','_y'))
    if len(j)<3: return None
    d = (j.ACC_x - j.ACC_y).values
    m=d.mean(); sd=d.std(ddof=1); se=sd/np.sqrt(len(d)); dz=m/sd if sd>0 else np.nan
    try:
        if np.allclose(d,0): wp=1.0
        else: wp=stats.wilcoxon(d, zero_method='wilcox').pvalue
    except Exception:
        wp=np.nan
    return dict(dataset=ds, cmp=f'{sx}-{sy}', n=len(d), diff=m, se=se,
                dz=dz, wilcoxon_p=wp, noninf=bool(m>=NI),
                acc_x=j.ACC_x.mean(), acc_y=j.ACC_y.mean())

print('\n=== Paired tests @ACC10%  (diff = first - A) ===')
rows=[]
for ds in ALL:
    for sx in ['C','B']:
        r=paired(ds,sx)
        if r: rows.append(r)
pt=pd.DataFrame(rows)
pd.set_option('display.width',200)
print(pt.round(4).to_string(index=False))
pt.to_csv(os.path.join(R,'alpha_paired_tests.csv'), index=False, encoding='utf-8')

# scheme 13-dataset equal-weight mean @10%
print('\n=== Scheme means @10% (13-dataset equal weight) ===')
m10 = cmp[abs(cmp.budget_frac-0.10)<1e-9].groupby(['scheme','dataset']).ACC.mean().unstack('scheme')
m10['nseed']=[NSEED[d] for d in m10.index]
print(m10.round(4).to_string())
for s in ['A','B','C']:
    if s in m10: print(f'  {s} mean = {m10[s].mean():.4f}')
m10.to_csv(os.path.join(R,'alpha_scheme_acc10_means.csv'), encoding='utf-8')

# adoption gate (C vs A)
pc = pt[pt.cmp=='C-A'].set_index('dataset')
ok_large = all(pc.loc[d,'diff']>=NI for d in LARGE if d in pc.index)
ok_small = all(pc.loc[d,'diff']>=NI for d in SMALL if d in pc.index)
print(f'\nADOPT GATE: 4-large C>=A non-inf? {ok_large} | 9-small non-inf? {ok_small} '
      f'=> ADOPT={ok_large and ok_small}')
for d in LARGE:
    if d in pc.index:
        print(f'  {d:>8}: C-A={pc.loc[d,"diff"]:+.4f} p={pc.loc[d,"wilcoxon_p"]:.3f}')

# Welch + PC diagnostics from structure table
st = pd.read_csv(os.path.join(R,'alpha_structure.csv'))
rp = stats.pearsonr(st.r_cross, st.r_W); rs = stats.spearmanr(st.r_cross, st.r_W)
rg = stats.pearsonr(st.gamma, st.gamma_W); rgs = stats.spearmanr(st.gamma, st.gamma_W)
mad = (st.gamma-st.gamma_W).abs().mean()
print(f'\nr_cross ~ r_W : Pearson r={rp.statistic:.3f}(p={rp.pvalue:.3f})  '
      f'Spearman rho={rs.statistic:.3f}(p={rs.pvalue:.3f})')
print(f'gamma ~ gamma_W: Pearson={rg.statistic:.3f} Spearman={rgs.statistic:.3f} '
      f'MAD={mad:.3f}')
print('\nPC / entropy (partition degeneracy):')
print(st[['ds','gamma','PC','H_norm','d_eff','r_cross','r_W']].round(3).to_string(index=False))
