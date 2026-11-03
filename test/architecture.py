"""Architecture rendering contracts; no Julia model evaluations."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from xml.etree import ElementTree

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('pbm_architecture',ROOT/'paper/architecture/generate_pbm_architecture.py')
architecture=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(architecture)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Architecture(unittest.TestCase):
    def setUp(self):
        self.context=architecture.configure(preview=True)
        self.context.__enter__()
        architecture.configure_schematic()
        self.figure=architecture.build_figure(equation_usetex=False)
        self.axes=self.figure.axes[0]
        self.labels={text.get_text():text for text in self.axes.texts}

    def tearDown(self):
        plt.close(self.figure)
        self.context.__exit__(None,None,None)

    def test_routes_and_optional_inputs(self):
        architecture.check_architecture()
        self.assertIn('TermMethods',self.labels)
        self.assertIn('moment_method',self.labels)
        transient=self.labels['PBMProblem'].get_position()[0]
        stationary=self.labels['StationaryPBMProblem'].get_position()[0]
        for label in ('Initial state\nand time span','simulate'):
            self.assertEqual(self.labels[label].get_position()[0],transient)
        self.assertEqual(self.labels['Numerical guess'].get_position()[0],stationary)
        self.assertEqual(self.labels['solve_stationary'].get_position()[0],stationary)
        for label in ('PBMSystem','AxialDispersion'):
            position=self.labels[label].get_position()
            self.assertTrue(any(isinstance(p,FancyBboxPatch) and p.get_linestyle()=='--'
                                and p.get_path().contains_point(position) for p in self.axes.patches))
        for label in ('UncertainInputs','UQProblem: scenario setup'):
            self.assertIn(label,self.labels)

    def test_readable_labels_without_clipping_or_overlap(self):
        self.figure.canvas.draw()
        renderer=self.figure.canvas.get_renderer()
        boxes=[(t.get_text(),t.get_window_extent(renderer)) for t in self.axes.texts]
        for text,box in boxes:
            self.assertGreaterEqual(box.x0,0,text)
            self.assertGreaterEqual(box.y0,0,text)
            self.assertLessEqual(box.x1,self.figure.bbox.width,text)
            self.assertLessEqual(box.y1,self.figure.bbox.height,text)
        for i,(left,a) in enumerate(boxes):
            for right,b in boxes[i+1:]:
                self.assertFalse(a.overlaps(b),(left,right))

    def test_final_exports_and_recorded_review(self):
        directory=ROOT/'paper/figures/00_architecture'
        receipt=json.loads((directory/'section2_architecture_revised.json').read_text())
        png=directory/receipt['png_filename']
        docs=ROOT/'docs/src/assets/architecture'
        self.assertEqual(digest(png),receipt['png_sha256'])
        self.assertEqual(png.read_bytes(),(docs/'section2_architecture_revised.png').read_bytes())
        self.assertFalse(receipt['preview'])
        self.assertEqual(receipt['equation_renderer'],'actual LaTeX')
        self.assertEqual(receipt['font_family'],'DejaVu Sans')
        with Image.open(png) as image:
            self.assertEqual(image.size,(2160,2460))
            self.assertTrue(all(abs(dpi-300)<.01 for dpi in image.info['dpi']))
        ElementTree.parse(docs/'section2_architecture_revised.svg')
        self.assertTrue((docs/'section2_architecture_revised.pdf').read_bytes().rstrip().endswith(b'%%EOF'))
        # A recorded review remains valid as historical provenance after an
        # unrelated source edit. Do not require current-source hash equality.
        for path in architecture.ARCHITECTURE_SOURCES:
            self.assertRegex(receipt['architecture_source_sha256'][str(path)],r'^[0-9a-f]{64}$')

    def test_preview_writes_only_separate_outputs(self):
        docs=ROOT/'docs/src/assets/architecture/section2_architecture_revised.png'
        before=digest(docs)
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([sys.executable,str(ROOT/'paper/architecture/generate_pbm_architecture.py'),
                            '--preview','--output-dir',directory],check=True,
                           env=dict(os.environ,MPLBACKEND='Agg'),capture_output=True,text=True)
            self.assertEqual(set(p.name for p in Path(directory).iterdir()),
                             {'architecture_preview.png','section2_architecture_revised_preview.json'})
        self.assertEqual(digest(docs),before)


if __name__=='__main__':
    unittest.main(verbosity=2)
