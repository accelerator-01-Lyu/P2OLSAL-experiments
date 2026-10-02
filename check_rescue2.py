import pandas as pd

# segment: check seeds per config and whether params took effect
df = pd.read_csv('results/rescue_tuning_segment.csv')
print('=== segment: rows per (lam,kappa) ===')
print(df.groupby(['lam','kappa']).size())
print()
print('=== segment lam=0 detail ===')
print(df[df.lam==0][['seed','ACC']].to_string(index=False))
print()
print('=== segment lam=0.01 detail (check if identical seeds) ===')
print(df[df.lam==0.01][['kappa','seed','ACC']].head(12).to_string(index=False))

# noise: what is the actual baseline? check eta values and seed count
dn = pd.read_csv('results/rescue_tuning_noise.csv')
print()
print('=== noise: eta values ===', dn.eta.unique())
print('=== noise: datasets ===', dn.dataset.unique())
print('rows per config:', dn.groupby(['lam','kappa']).size().unique())
# baseline config lam=0.1 kappa=1.0 should reproduce ~0.718
base = dn[(dn.lam==0.1)&(dn.kappa==1.0 if 'kappa' in dn else dn.kappa==1)]
print('baseline lam=0.1,kappa=1 mean ACC:', dn[(dn.lam==0.1)].ACC.mean())
print('overall noise mean:', dn.ACC.mean())
