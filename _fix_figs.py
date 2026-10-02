# -*- coding: utf-8 -*-
import re
p = r'D:\P2OLSAL_FSS\manuscript\P2OLSAL_FSS_manuscript.md'
lines = open(p, encoding='utf-8').read().split('\n')

# 1) delete the two leftover old figure blocks
drop_markers = ['fig02_advantage_gamma.png', 'fig07_alpha_factorial.png']
out = [ln for ln in lines if not any(m in ln for m in drop_markers)]
text = '\n'.join(out)
text = re.sub(r'\n{3,}', '\n\n', text)

# 2) insert ablation waterfall (Fig. 9) right after the 8.3 heading
wf = ('### 8.3 Ablation studies\n\n'
      '![Fig. 9: Ablation waterfall at 10% budget (13-dataset equal-weighted ACC). '
      'Starting from P2-OLSAL Full (71.5%), removing the robust DV gain ($-3.9$pp), '
      'the redundancy term ($-1.4$pp), and the block-adaptive weighting ($-1.0$pp) '
      'accounts for essentially all performance; the FMIS and exploration removals are '
      'null (the exploration term is already off by default).]'
      '(../experiments/figures/fig_ablation_waterfall.png)\n')
assert text.count('### 8.3 Ablation studies\n') == 1
text = text.replace('### 8.3 Ablation studies\n', wf, 1)

open(p, 'w', encoding='utf-8').write(text)
print('done; old blocks remaining:',
      sum(m in text for m in drop_markers),
      '; waterfall inserted:', 'fig_ablation_waterfall.png' in text)
