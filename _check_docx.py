# -*- coding: utf-8 -*-
import zipfile, os
docx = r'D:\P2OLSAL_FSS\manuscript\P2OLSAL_FSS_manuscript.docx'
print('size MB:', round(os.path.getsize(docx)/1e6, 2))
z = zipfile.ZipFile(docx)
media = [n for n in z.namelist() if n.startswith('word/media/')]
print('embedded images:', len(media))
xml = z.read('word/document.xml').decode('utf-8', 'ignore')
print('OMML math regions:', xml.count('<m:oMath'))
for name in ['Jinze Fu', 'Haolin Liu', 'Yutong Zhao', 'Sun Yat-sen', 'Southern University']:
    print(f'  {name!r}:', name in xml)
print('literal "\\newpage" leaked:', 'newpage' in xml)
print('--- appendix coverage ---')
for tag in ['Appendix A', 'Appendix B', 'Appendix C', 'Appendix D', 'Appendix E',
            'Appendix F', 'Appendix G', 'Appendix H', 'Appendix I', 'Appendix J']:
    print(f'  {tag}:', tag in xml)
# page break present?
print('explicit page breaks (w:br page):', xml.count('w:type="page"'))
