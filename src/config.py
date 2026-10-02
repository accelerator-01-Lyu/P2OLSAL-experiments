"""Global configuration for P2-OLSAL experiments."""
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HERE, 'data')
RESULTS_DIR = os.path.join(HERE, 'results')
FIGURES_DIR = os.path.join(HERE, 'figures')

# Original 6 datasets
DATASETS = ['iris', 'wine', 'seeds', 'glass', 'ecoli', 'segment']
# New 7 datasets (ng20 skipped due to data source 403)
NEW_DATASETS = ['balance', 'aggregation', 'compound',
                'letter', 'shuttle', 'usps', 'fashion']
ALL_DATASETS = DATASETS + NEW_DATASETS

BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30]
ETAS = [0.0, 0.10, 0.20, 0.30]
SMALL_SEEDS = list(range(30))
LARGE_SEEDS = list(range(8))
LARGE_THRESHOLD = 2500

# FCM parameters
FCM_M = 2.0
FCM_MAX_ITER = 150
FCM_TOL = 1e-6

# P2-OLSAL parameters (tuned via hyperparameter search, see hyperparam_tuning_report.txt)
ROBUST_KAPPA_BASE = 1.0   # ambiguity set radius base (0.3~1.0 insensitive)
ROBUST_EPS_Q = 0.10       # mixture weight for uniform component
EXPL_RHO = 0.0            # exploration reward weight (0 = off, found harmful)
MC_SAMPLES = 50           # Monte Carlo samples for expected entropy gain
DIRICHLET_ALPHA0 = 50.0   # base concentration for posterior predictive (0.5~50 insensitive)
P2_LAM = 0.1              # redundancy penalty weight (tuned down from 1.0)
P2_ALPHA_CROSS_FRAC = 0.25  # cross-block alpha fraction (tuned down from 0.5)

# Large-dataset guard: subsample redundancy matrix when unlabeled count exceeds this
MAX_REDUNDANCY_N = 4000
REDUNDANCY_SUBSAMPLE_N = 2000
