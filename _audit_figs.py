# -*- coding: utf-8 -*-
import re, os
p = r'D:\P2OLSAL_FSS\manuscript\P2OLSAL_FSS_manuscript.md'
base = r'D:\P2OLSAL_FSS\manuscript'
text = open(p, encoding='utf-8').read()
lines = text.split('\n')

print('=== image blocks (physical order) ===')
nums = []
for i, ln in enumerate(lines, 1):
    if ln.startswith('![Fig'):
        n = re.search(r'Fig\. (\d+)', ln).group(1)
        path = re.search(r'\]\(([^)]+)\)', ln).group(1)
        exists = os.path.exists(os.path.join(base, path))
        nums.append(int(n))
        print(f'L{i}: Fig {n:>2}  exists={exists}  {os.path.basename(path)}')
print('numbers:', nums, 'continuous 1..N:', nums == list(range(1, len(nums)+1)))

print('\n=== in-text Fig references ===')
for i, ln in enumerate(lines, 1):
    if ln.startswith('!['):
        continue
    refs = re.findall(r'Fig\. ?\d+[a-z]?', ln)
    if refs:
        print(f'L{i}: {refs}')
