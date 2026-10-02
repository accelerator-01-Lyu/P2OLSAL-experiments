"""Fix syntax errors in baselines_fuzzy.py."""
with open('src/baselines_fuzzy.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find all backtick-n occurrences
import re
matches = [(m.start(), content[max(0,m.start()-10):m.end()+10]) for m in re.finditer(r'`n', content)]
print(f'Found {len(matches)} occurrences of `n')
for pos, ctx in matches:
    print(f'  pos {pos}: ...{repr(ctx)}...')

# Remove all backtick-n (they are literal `n that should be newlines)
content = content.replace('`n', '\n')

# Also check for standalone backticks
backticks = content.count('`')
print(f'\nRemaining backticks: {backticks}')

with open('src/baselines_fuzzy.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Verify syntax
import py_compile
try:
    py_compile.compile('src/baselines_fuzzy.py', doraise=True)
    print('\nSyntax OK!')
except py_compile.PyCompileError as e:
    print(f'\nSyntax ERROR: {e}')
