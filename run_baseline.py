"""Launch baseline sweep."""
import sys
sys.path.insert(0, '.')
from src.sweep_baseline import run

datasets = ['wine', 'seeds', 'glass', 'ecoli', 'segment', 'balance',
            'aggregation', 'compound', 'letter', 'shuttle', 'usps', 'fashion']
run(datasets, workers=4)
