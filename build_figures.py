"""Build canonical Fig.1-11 files (uniform naming) for the manuscript.
Fig.2 (advantage-regime gamma vs delta ACC) and Fig.7 (A/D/E/B factorial bars)
are newly computed from CSV; others are copied or composed from existing PNGs.
"""
import os, shutil
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy import stats
R=r'D:\P2OLSAL_FSS\experiments\results'
F=r'D:\P2OLSAL_FSS\experiments\figures'
SMALL=['iris','wine','seeds','glass','ecoli','segment','balance','aggregation','compound']
LARGE=['letter','shuttle','usps','fashion']; ALL=SMALL+LARGE

def cp(src,dst):
    shutil.copyfile(os.path.join(F,src),os.path.join(F,dst)); print('copy',dst)

cp('fig1_learning_curves.png','fig01_learning_curves.png')
cp('fig1_rho_cliff.png','fig03_rho_cliff.png')
cp('fig2_selection_shift.png','fig04_selection_shift.png')
cp('fig2_noise_robustness.png','fig05_noise_robustness.png')
cp('fig5_misfit_proxy.png','fig08_misfit_proxy.png')
cp('fig8_adversarial_regret.png','fig09_adversarial_regret.png')
cp('fig9_alpha_sensitivity.png','fig10_alpha_sensitivity.png')

# ---------- Fig.2 advantage-regime: gamma vs (P2 - best external SOTA) ----------
main=pd.read_csv(os.path.join(R,'all_results_merged.csv'),low_memory=False)
EXT=['BADGE','CoreSet','BALD','QBC','Entropy','Random','SSFCM','CEFCM','GRFCM','PLCFCMPassive']
m10=main[(abs(main.budget_frac-0.10)<1e-9)&(main.eta==0)]
p2=m10[m10.method=='P2_Full'].groupby('dataset').ACC.mean()
ext=m10[m10.method.isin(EXT)].groupby(['dataset','method']).ACC.mean().unstack('method')
best_ext=ext.max(axis=1)
st=pd.read_csv(os.path.join(R,'alpha_structure.csv')).set_index('ds')
rows=[]
for d in ALL:
    rows.append(dict(dataset=d,gamma=st.loc[d,'gamma'],d_eff=st.loc[d,'d_eff'],
                     P2=p2[d],best=best_ext[d],delta=p2[d]-best_ext[d]))
adv=pd.DataFrame(rows)
x=adv.gamma.values; y=adv.delta.values
rp=stats.pearsonr(x,y); rs=stats.spearmanr(x,y)
b1,b0=np.polyfit(x,y,1); xs=np.linspace(x.min(),x.max(),50)
fig,ax=plt.subplots(figsize=(7.2,5.2))
colors=['#1f77b4' if d in SMALL else '#d62728' for d in adv.dataset]
ax.scatter(x,y,s=70,c=colors,edgecolor='k',zorder=3)
for _,r in adv.iterrows():
    ax.annotate(r.dataset,(r.gamma,r.delta),xytext=(4,4),
                textcoords='offset points',fontsize=8)
ax.plot(xs,b1*xs+b0,'k--',lw=1.5,zorder=2,
        label=f'trend slope={b1:.2f}')
ax.axhline(0,color='gray',lw=.8); ax.set_xlim(.6,1.05)
ax.set_xlabel(r'Block contrast $\gamma$ (membership space, current $U$)')
ax.set_ylabel(r'$\Delta$ACC = P2-OLSAL $-$ best external baseline @10%')
ax.set_title(f'Advantage regime (13 datasets)\n'
             f'Pearson r={rp.statistic:.2f} (p={rp.pvalue:.3f}), '
             f'Spearman '+r'$\rho$'+f'={rs.statistic:.2f}', fontsize=11)
ax.legend(); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(os.path.join(F,'fig02_advantage_gamma.png'),dpi=150)
plt.close(fig); print('wrote fig02')
adv.to_csv(os.path.join(R,'fig2_advantage_data.csv'),index=False)
print('Pearson',round(rp.statistic,3),round(rp.pvalue,4),'Spearman',round(rs.statistic,3))

# ---------- Fig.7 A/D/E/B grouped bars: 4 large + 4 representative small ----------
cell=pd.read_csv(os.path.join(R,'factorial_cell_acc.csv')).set_index('dataset')
order=['letter','shuttle','usps','fashion','ecoli','segment','iris','aggregation']
labels=order; k=len(order)
fig,ax=plt.subplots(figsize=(11,5))
w=.2; idx=np.arange(k)
for j,(c,col) in enumerate(zip(['A','D','E','B'],['#7f7f7f','#1f77b4','#2ca02c','#d62728'])):
    ax.bar(idx+(j-1.5)*w,cell.loc[labels,c].values,w,label={'A':'A baseline',
        'D':'D cross=0','E':'E within up','B':'B both'}[c],color=col,edgecolor='k',lw=.4)
ax.set_xticks(idx); ax.set_xticklabels(labels,rotation=20)
ax.set_ylabel('ACC @10%'); ax.set_ylim(.3,1.0)
ax.set_title('2$\\times$2 block-penalty factorial (paired seeds; D$\\equiv$B: gain comes from removing cross-block penalty)')
ax.legend(ncol=4,fontsize=9); ax.grid(axis='y',alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(F,'fig07_alpha_factorial.png'),dpi=150)
plt.close(fig); print('wrote fig07')

# ---------- Fig.6 compose cumulant + spectral gap ----------
fig,axs=plt.subplots(1,2,figsize=(13,4.6))
for ax,fn,t in zip(axs,['fig3_cumulant_gain.png','fig4_spectral_gap.png'],
                   ['(a) Cumulant gain / third-cumulant diagnosis',
                    '(b) Block spectral-gap validation']):
    ax.imshow(imread(os.path.join(F,fn))); ax.axis('off'); ax.set_title(t,fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(F,'fig06_cumulant_gap.png'),dpi=150)
plt.close(fig); print('wrote fig06')

# ---------- Fig.11 compose three rescue panels ----------
fig,axs=plt.subplots(3,1,figsize=(8.5,15))
for ax,fn,t in zip(axs,['fig_rescue_sensitivity_segment.png',
                        'fig_rescue_sensitivity_glass.png',
                        'fig_rescue_sensitivity_noise.png'],
                   ['Segment','Glass','Ecoli (high noise)']):
    ax.imshow(imread(os.path.join(F,fn))); ax.axis('off'); ax.set_title(t,fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(F,'fig11_rescue.png'),dpi=140)
plt.close(fig); print('wrote fig11')
print('DONE')
