"""Extract coloured EXPERIMENTAL markers from Mitchell2023 Figure 4 (CC BY 4.0).
Requires pymupdf, numpy and scipy; never extracts the fitted model curves.
Run at repository root: python examples/data/protein_crystallization/extract_figure4.py
"""
from pathlib import Path
import csv, hashlib
import pymupdf
import numpy as np
from scipy.ndimage import label, find_objects

SOURCE = Path('local_papers/Mitchell2023.pdf')
EXPECTED = '50cb6c6be26a21b83c8fe03a7324d0f4848248352895f69a0e368377d70cfc70'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED, 'Recalibrate a different PDF'
pdf = pymupdf.open(SOURCE)
image = pymupdf.Pixmap(pdf, pdf[7].get_images(full=True)[0][0])
assert (image.width, image.height, image.n) == (1350, 793, 3)
r, g, b = np.frombuffer(image.samples, dtype=np.uint8).reshape(793,1350,3).astype(float).transpose(2,0,1)
# Axis intersections in the embedded image: (111,16),(1244,700).
# Tick readings: t=0..700 min; concentration=20..4 g/L; D50=14..0 micrometres.
rows=[]
for name, mask in [('concentration',(r>160)&(g<105)&(b<105)),
                   ('D50',(g>95)&(r<155)&(b<105)&(g>1.1*r))]:
    labels,_=label(mask)
    for i, region in enumerate(find_objects(labels)):
        if region is None: continue
        yy,xx=region
        area=np.sum(labels[region]==i+1)
        if not 65<=area<=130 or not 95<=xx.start<=1255 or yy.start>710: continue
        x=(xx.start+xx.stop-1)/2; y=(yy.start+yy.stop-1)/2
        t=(x-111)*700/1133
        value=20-(y-16)*16/684 if name=='concentration' else 14-(y-16)*14/684
        # Experiment schedule is recoverable to within one pixel (~0.62 min).
        # Snap to the nearest 5 min; retain graph-read pixel centres for audit.
        rows.append((name,round(t/5)*5,round(value,5),0.1 if name=='concentration' else 0.5,x,y))
rows.sort(key=lambda r:(r[0],r[1]))
assert sum(r[0]=='concentration' for r in rows)==28
assert sum(r[0]=='D50' for r in rows)==4
with (Path(__file__).parent/'figure4_observations.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['observable','time_min','value','assumed_sigma','pixel_x','pixel_y']);w.writerows(rows)
print('Extracted',len(rows),'experimental markers; no synthetic measurements.')
