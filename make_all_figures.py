# -*- coding: utf-8 -*-
"""One-click rebuild of EVERY journal figure.

Discovers every build script in journal_figs/ whose filename does NOT start with
'_' (helpers like _make_selection_pool.py are skipped), imports it, and calls its
build(); each script imports the single style source figures/_journal_style.py and
writes 400-dpi PNG + same-name vector PDF into figures/ via js.save_both.

New figure scripts only need to be dropped into journal_figs/ with a build() to
be picked up automatically. Run:  python make_all_figures.py
"""
import os, sys, glob, importlib, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
FIGSRC = os.path.join(HERE, 'figures')
SCRIPTS = os.path.join(HERE, 'journal_figs')
for p in (FIGSRC, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')


def main():
    files = sorted(f for f in glob.glob(os.path.join(SCRIPTS, '*.py'))
                   if not os.path.basename(f).startswith('_'))
    ok, failed = [], []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            mod = importlib.import_module(name)
            if not hasattr(mod, 'build'):
                print('skip (no build):', name)
                continue
            mod.build()
            import matplotlib.pyplot as plt
            plt.close('all')
            ok.append(name)
            print('built:', name)
        except Exception:
            failed.append(name)
            print('FAILED:', name)
            traceback.print_exc()
    print('\n=== %d built, %d failed ===' % (len(ok), len(failed)))
    if failed:
        print('failed scripts:', failed)
        sys.exit(1)


if __name__ == '__main__':
    main()
