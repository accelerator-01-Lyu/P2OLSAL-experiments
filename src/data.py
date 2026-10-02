"""Dataset loading from OpenML via sklearn.fetch_openml, cached locally."""
import os, warnings, pickle
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HERE, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

OPENML_SPEC = {
    'iris':    ('iris', 1),
    'wine':    ('wine', 1),
    'seeds':   ('seeds', 1),
    'glass':   ('glass', 1),
    'ecoli':   ('ecoli', 1),
    'segment': ('segment', 1),
    'balance': ('balance-scale', 1),
    'letter':  ('letter', 1),
    'shuttle': ('shuttle', 1),
    'usps':    ('usps', 1),
}


def load_dataset(name):
    """Return (X, y, info) where X is standardized (n,d), y integer labels."""
    cache = os.path.join(DATA_DIR, f'{name}.pkl')
    if os.path.exists(cache):
        with open(cache, 'rb') as f:
            obj = pickle.load(f)
        return obj['X'], obj['y'], obj['info']

    oml_name, version = OPENML_SPEC[name]
    d = fetch_openml(name=oml_name, version=version, as_frame=True, parser='auto')
    df = d.data.copy()
    target = d.target
    drop = [c for c in df.columns if 'name' in str(c).lower()
            or c.lower() in ('sequence_name', 'id')]
    if drop:
        df = df.drop(columns=drop)
    df = df.select_dtypes(include=[np.number])
    X = df.to_numpy(dtype=float)
    y_str = target.astype(str).to_numpy() if hasattr(target, 'astype') else np.asarray(target)
    classes = sorted(set(y_str.tolist()))
    cls2int = {c: i for i, c in enumerate(classes)}
    y = np.array([cls2int[v] for v in y_str], dtype=int)
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    info = {
        'openml_name': oml_name,
        'data_id': d.details.get('id'),
        'url': f'https://www.openml.org/d/{d.details.get("id")}',
        'n': int(Xs.shape[0]),
        'd': int(Xs.shape[1]),
        'c': int(len(classes)),
        'classes': classes,
        'dropped_cols': drop,
        'scaler': 'StandardScaler fit on full data',
    }
    with open(cache, 'wb') as f:
        pickle.dump({'X': Xs, 'y': y, 'info': info}, f)
    return Xs, y, info
