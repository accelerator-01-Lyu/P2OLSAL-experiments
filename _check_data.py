import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import load_dataset

for ds in ['balance','aggregation','compound','letter','shuttle','usps','fashion']:
    X, y, info = load_dataset(ds)
    n = X.shape[0]
    d = X.shape[1]
    c = info['c']
    seeds = 8 if n >= 2500 else 30
    print(f"{ds:12s} n={n:5d} d={d:3d} c={c:2d} seeds={seeds}")
