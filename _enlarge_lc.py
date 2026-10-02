# -*- coding: utf-8 -*-
import io
p = r'D:\P2OLSAL_FSS\experiments\journal_figs\learning_curves.py'
s = io.open(p, encoding='utf-8').read()
reps = [
    ('ax.set_title(title, fontsize=9.0,', 'ax.set_title(title, fontsize=9.6,'),
    ('ax.tick_params(labelsize=8.0,', 'ax.tick_params(labelsize=8.7,'),
    ('figsize=(7.2, 6.1)', 'figsize=(7.2, 6.5)'),
    ("lw=1.15, marker=st['marker'], ms=3.4,", "lw=1.2, marker=st['marker'], ms=3.7,"),
    ("marker='o', ms=4.8,", "marker='o', ms=5.1,"),
]
for a, b in reps:
    assert s.count(a) == 1, (a, s.count(a))
    s = s.replace(a, b)
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('learning_curves fonts enlarged')
