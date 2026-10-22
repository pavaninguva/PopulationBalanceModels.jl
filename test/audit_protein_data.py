"""Verify retained Figure 4 readings from the local, attributed source PDF.

Requires PyMuPDF and NumPy, not Julia runtime dependencies. This uses an
independent four-neighbour flood fill instead of the extraction recipe's SciPy
component labelling. It never writes observations or redistributes the PDF.
"""
from collections import deque
import csv
import hashlib
from pathlib import Path

import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "50cb6c6be26a21b83c8fe03a7324d0f4848248352895f69a0e368377d70cfc70"


def components(mask):
    visited = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    for y, x in np.argwhere(mask):
        if visited[y, x]:
            continue
        queue = deque([(y, x)])
        visited[y, x] = True
        area, xmin, xmax, ymin, ymax = 0, x, x, y, y
        while queue:
            y, x = queue.popleft()
            area += 1
            xmin, xmax = min(xmin, x), max(xmax, x)
            ymin, ymax = min(ymin, y), max(ymax, y)
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= yy < height and 0 <= xx < width and mask[yy, xx] and not visited[yy, xx]:
                    visited[yy, xx] = True
                    queue.append((yy, xx))
        yield area, xmin, xmax, ymin, ymax


def check():
    source = ROOT / "local_papers" / "Mitchell2023.pdf"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == EXPECTED
    with pymupdf.open(source) as pdf:
        image = pymupdf.Pixmap(pdf, pdf[7].get_images(full=True)[0][0])
        assert (image.width, image.height, image.n) == (1350, 793, 3)
        r, g, b = np.frombuffer(image.samples, dtype=np.uint8).reshape(793, 1350, 3).astype(float).transpose(2, 0, 1)
    rows = []
    masks = (("concentration", (r > 160) & (g < 105) & (b < 105)),
             ("D50", (g > 95) & (r < 155) & (b < 105) & (g > 1.1*r)))
    for name, mask in masks:
        for area, xmin, xmax, ymin, ymax in components(mask):
            if not (65 <= area <= 130 and 95 <= xmin <= 1255 and ymin <= 710):
                continue
            x, y = (xmin+xmax)/2, (ymin+ymax)/2
            time = round(((x-111)*700/1133)/5)*5
            value = 20-(y-16)*16/684 if name == "concentration" else 14-(y-16)*14/684
            rows.append((name, time, round(value, 5), .1 if name == "concentration" else .5, x, y))
    rows.sort(key=lambda row: (row[0], row[1]))
    with (ROOT / "examples/data/protein_crystallization/figure4_observations.csv").open() as file:
        saved = [(row["observable"], *(float(row[key]) for key in
                  ("time_min", "value", "assumed_sigma", "pixel_x", "pixel_y"))) for row in csv.DictReader(file)]
    assert len(rows) == 32
    assert sum(row[0] == "concentration" for row in rows) == 28
    assert rows == saved
    print("PDF hash and image calibration match; all 32 retained experimental marker readings reproduced.")


if __name__ == "__main__":
    check()
