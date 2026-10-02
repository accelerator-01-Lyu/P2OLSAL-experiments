"""Prepare 8 additional datasets into the same pkl cache schema as data.py."""
import os, gzip, pickle, warnings, urllib.request
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml, fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
RAW = os.path.join(DATA, '_raw')
os.makedirs(RAW, exist_ok=True)


def _encode(target):
    ystr = target.astype(str).to_numpy() if hasattr(target, 'astype') else np.asarray(target, str)
    classes = sorted(set(ystr.tolist()))
    m = {c: i for i, c in enumerate(classes)}
    return np.array([m[v] for v in ystr], dtype=int), classes


def save(name, X, y, info):
    X = np.asarray(X, float)
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    info.update(n=int(Xs.shape[0]), d=int(Xs.shape[1]), c=int(y.max() + 1),
                scaler='StandardScaler fit on the analyzed subset')
    with open(os.path.join(DATA, name + '.pkl'), 'wb') as f:
        pickle.dump({'X': Xs, 'y': y, 'info': info}, f)
    print(f'{name:11s} n={info["n"]:5d} d={info["d"]:3d} c={info["c"]:2d}', flush=True)


def openml_full(name, oml_name):
    d = fetch_openml(name=oml_name, as_frame=True, parser='auto')
    X = d.data.select_dtypes(include=[np.number]).to_numpy(float)
    y, classes = _encode(d.target)
    save(name, X, y, {'openml_name': oml_name, 'data_id': d.details.get('id'),
                      'url': f'https://www.openml.org/d/{d.details.get("id")}',
                      'subset': 'full'})


def openml_subset(name, oml_name, n, seed=0):
    d = fetch_openml(name=oml_name, as_frame=True, parser='auto')
    X = d.data.select_dtypes(include=[np.number]).to_numpy(float)
    y, _ = _encode(d.target)
    n0 = X.shape[0]
    X, _, y, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    save(name, X, y, {'openml_name': oml_name, 'data_id': d.details.get('id'),
                      'url': f'https://www.openml.org/d/{d.details.get("id")}',
                      'subset': f'stratified {n} from {n0}'})


def prepare_fashion(n=10000, seed=0):
    d = fetch_openml(name='Fashion-MNIST', as_frame=True, parser='auto')
    X = d.data.select_dtypes(include=[np.number]).to_numpy(float)
    y, _ = _encode(d.target)
    n0 = X.shape[0]
    X, _, y, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=50, random_state=seed)
    Z = pca.fit_transform(X)
    save('fashion', Z, y, {'openml_name': 'Fashion-MNIST', 'data_id': d.details.get('id'),
                           'url': f'https://www.openml.org/d/{d.details.get("id")}',
                           'subset': f'stratified {n} from {n0}',
                           'pipeline': 'StandardScaler -> PCA(50)',
                           'pca_evr': float(pca.explained_variance_ratio_.sum())})


def prepare_ng20(n=10000, seed=0):
    bundle = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    y = np.asarray(bundle.target)
    X_txt = np.asarray(bundle.data, dtype=object)
    X_txt, _, y, _ = train_test_split(X_txt, y, train_size=n, stratify=y, random_state=seed)
    vec = TfidfVectorizer(min_df=3, max_df=0.9, stop_words='english', sublinear_tf=True)
    T = vec.fit_transform(X_txt)
    svd = TruncatedSVD(n_components=100, random_state=seed)
    Z = svd.fit_transform(T)
    save('ng20', Z, y, {'source': 'sklearn fetch_20newsgroups subset=all',
                        'subset': f'stratified {n} from 18846',
                        'pipeline': 'TfidfVectorizer -> TruncatedSVD(100)',
                        'svd_evr': float(svd.explained_variance_ratio_.sum())})


def prepare_2d(name, seed=0):
    """Generate 2D synthetic datasets matching classic benchmark structures.

    Aggregation-style: 7 Gaussian blobs with two clusters in narrow-neck contact.
    Compound-style: 6 clusters including nested half-moons and compact blobs.
    (Original Gagolewski repository is no longer publicly accessible.)
    """
    from sklearn.datasets import make_blobs, make_moons
    rng = np.random.RandomState(seed)
    if name == 'aggregation':
        centers = [(0, 0), (12, 0), (24, 0), (0, 12), (24, 12),
                   (10, 6), (14, 6)]
        stds = [1.5, 1.5, 1.5, 1.5, 1.5, 0.8, 0.8]
        n_per = [120, 120, 120, 100, 100, 114, 114]
        X_parts, y_parts = [], []
        for k, (c, s, n) in enumerate(zip(centers, stds, n_per)):
            X_parts.append(rng.randn(n, 2) * s + np.array(c))
            y_parts.append(np.full(n, k))
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        src = 'synthetic: 7 Gaussian blobs, 2 in narrow-neck contact (Aggregation-style)'
    elif name == 'compound':
        X1, y1 = make_moons(n_samples=150, noise=0.08, random_state=seed)
        X1 = X1 * 3 + np.array([5, 5])
        X2, y2 = make_blobs(n_samples=249, centers=4, n_features=2,
                            cluster_std=0.6, random_state=seed + 1,
                            center_box=(-8, 8))
        y2 = y2 + 2
        X = np.vstack([X1, X2])
        y = np.concatenate([y1, y2])
        src = 'synthetic: nested half-moons + 4 compact blobs (Compound-style)'
    else:
        raise ValueError(name)
    save(name, X, y, {'source': src, 'subset': 'full',
                      'note': 'Synthetic replacement for unavailable Gagolewski benchmark'})


if __name__ == '__main__':
    print('--- 2D benchmark sets', flush=True)
    prepare_2d('aggregation')
    prepare_2d('compound')
    print('--- OpenML full', flush=True)
    openml_full('balance', 'balance-scale')
    openml_full('usps', 'usps')
    print('--- OpenML subsets', flush=True)
    openml_subset('letter', 'letter', 7990)
    openml_subset('shuttle', 'shuttle', 4997)
    print('--- special pipelines', flush=True)
    prepare_fashion()
    prepare_ng20()
    print('ALL NEW DATASETS PREPARED', flush=True)
