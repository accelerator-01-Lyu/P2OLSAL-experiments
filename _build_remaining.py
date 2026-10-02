# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r'D:\P2OLSAL_FSS\experiments\journal_figs')
sys.path.insert(0, r'D:\P2OLSAL_FSS\experiments\figures')
import matplotlib.pyplot as plt
import make_remaining_figs as m
for fn in ['fig_geometry', 'fig_noise', 'fig_adversarial',
           'fig_misfit', 'fig_cumulant', 'fig_rescue']:
    getattr(m, fn)()
    plt.close('all')
    print('built', fn)
print('remaining figures rebuilt at 600 dpi')
