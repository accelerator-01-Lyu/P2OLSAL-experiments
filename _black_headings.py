# -*- coding: utf-8 -*-
from docx import Document
from docx.shared import RGBColor

docx = r'D:\P2OLSAL_FSS\manuscript\P2OLSAL_FSS_manuscript.docx'
doc = Document(docx)
black = RGBColor(0, 0, 0)
changed = []
for s in doc.styles:
    nm = getattr(s, 'name', '') or ''
    if nm == 'Title' or nm.startswith('Heading') or nm.startswith('Subtitle'):
        try:
            s.font.color.rgb = black
            changed.append(nm)
        except Exception as e:
            print('skip', nm, e)
# also force any run currently carrying the heading styles to black explicitly
for p in doc.paragraphs:
    sn = p.style.name if p.style else ''
    if sn == 'Title' or sn.startswith('Heading'):
        for r in p.runs:
            r.font.color.rgb = black
doc.save(docx)
print('styles set black:', changed)
