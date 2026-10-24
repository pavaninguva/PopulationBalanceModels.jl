"""Independent contracts for the CNT fit, data split and exported bin densities."""
import csv
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "paper/results/04_protein_crystallization/production"


def rows(name):
    with (BASE / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


class ProteinResults(unittest.TestCase):
    def test_provenance_and_original_data(self):
        manifest = json.loads((BASE / "manifest.json").read_text())
        self.assertEqual(manifest["mode"], "production")
        self.assertTrue(manifest["complete"])
        for path, digest in manifest["files"].items():
            self.assertEqual(hashlib.sha256((BASE / path).read_bytes()).hexdigest(), digest, path)
        for path, digest in manifest["source_files"].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest, path)
        data = ROOT / "examples/data/protein_crystallization/tang2018_isothermal"
        origin = json.loads((data / "provenance/integration.json").read_text())
        for path, digest in origin["data_hashes"].items():
            self.assertEqual(hashlib.sha256((data / path).read_bytes()).hexdigest(), digest)

    def test_split_map_and_metrics(self):
        conc = rows("concentration_predictions.csv")
        self.assertEqual(len(conc), 16)
        self.assertEqual(sum(r["split"] == "calibration" for r in conc), 11)
        self.assertEqual(sum(r["split"] == "validation" for r in conc), 4)
        self.assertEqual(float(conc[0]["time_min"]), 0)
        self.assertEqual(float(conc[0]["residual_g_L"]), 0)
        frozen = json.loads((BASE / "frozen_model.json").read_text())
        self.assertFalse(frozen["validation_used"])
        metrics = json.loads((BASE / "validation_metrics.json").read_text())
        self.assertLessEqual(frozen["frozen_at"], metrics["evaluated_at"])
        for split in ("calibration", "validation"):
            residual = [float(r["residual_g_L"]) for r in conc if r["split"] == split]
            self.assertAlmostEqual(np.sqrt(np.mean(np.square(residual))), metrics[f"{split}_concentration_RMSE_g_L"], places=12)
        mode = json.loads((BASE / "map.json").read_text())
        self.assertTrue(mode["converged"])
        self.assertLess(mode["gradient_norm"], .001)
        self.assertAlmostEqual(mode["logposterior"], mode["loglikelihood"] + mode["logprior"], places=12)
        self.assertEqual(len(mode["starts"]), 3)

    def test_conservative_coupling_and_density(self):
        traj = rows("trajectory.csv")
        self.assertLess(max(abs(float(r["material_error_g_L"])) for r in traj), 1e-9)
        self.assertGreaterEqual(min(float(r["minimum_scaled_count"]) for r in traj), 0)
        self.assertEqual(float(traj[0]["number_per_m3"]), 0)
        self.assertLess(max(float(r["upper_bin_fraction"]) for r in traj), 1e-8)
        curves, observed = rows("distribution_curves.csv"), rows("psd_predictions.csv")
        metrics = rows("distribution_metrics.csv")
        for t in (74., 180., 420., 660.):
            r = [a for a in curves if float(a["time_min"]) == t]
            left = np.array([float(a["left_um"]) for a in r])
            right = np.array([float(a["right_um"]) for a in r])
            density = np.array([float(a["conditional_density_per_um"]) for a in r])
            self.assertAlmostEqual(np.dot(right-left, density), 1, places=11)
            cdf = np.r_[0, np.cumsum((right-left)*density)]
            edges = np.r_[left, right[-1]]
            o = [a for a in observed if float(a["time_min"]) == t]
            x = np.array([float(a["size_um"]) for a in o])
            F = np.array([float(a["observed_cdf"]) for a in o])
            predicted = np.array([float(a["predicted_cdf"]) for a in o])
            np.testing.assert_allclose(np.interp(x, edges, cdf), predicted, atol=1e-12)
            # Independent two-point Gauss integration of affine quantile
            # differences on the common probability partition; no midpoint rule.
            support = np.flatnonzero(density > 0)
            lo, hi = support[0], support[-1]+1
            cm = cdf[lo:hi+1]/cdf[hi]
            partition = np.unique(np.r_[cm, F])
            widths = np.diff(partition)
            mid = (partition[:-1]+partition[1:])/2
            u = np.vstack((mid-widths/(2*np.sqrt(3)), mid+widths/(2*np.sqrt(3))))
            qm = np.interp(u, cm, edges[lo:hi+1])
            qo = np.interp(u, F, x)
            exact_w2 = np.sqrt(np.dot(widths, np.mean((qm-qo)**2, axis=0)))
            reported = next(float(a["W2_um"]) for a in metrics if float(a["time_min"]) == t)
            self.assertAlmostEqual(exact_w2, reported, delta=1e-4)

    def test_publication_artifacts(self):
        import pymupdf
        from PIL import Image
        folder=ROOT / "paper/figures/04_protein_crystallization/production"
        names=["protein_crystallization"]
        if (BASE / "sampling_diagnostics.json").exists() and json.loads((BASE / "sampling_diagnostics.json").read_text())["passed"]:
            names += ["protein_posterior","protein_corner"]
        for name in names:
            with Image.open(folder / f"{name}.png") as image:
                self.assertEqual(image.width,2100)
                np.testing.assert_allclose(image.info["dpi"],[300,300],atol=.01)
            with pymupdf.open(folder / f"{name}.pdf") as pdf:
                self.assertEqual(len(pdf),1)
                self.assertAlmostEqual(pdf[0].rect.width,504,places=2)
                fonts=pdf[0].get_fonts(full=True)
                self.assertTrue(fonts)
                for font in fonts:
                    self.assertTrue(pdf.extract_font(font[0])[3],font)
            self.assertTrue((folder / f"{name}.svg").exists())

    def test_posterior_if_requested(self):
        path = BASE / "sampling_diagnostics.json"
        if not path.exists():
            self.assertFalse((BASE / "latent_predictions.csv").exists())
            return
        d = json.loads(path.read_text())
        raw = rows("posterior_draws.csv")
        self.assertEqual(len(raw), d["chains"]*d["samples"])
        self.assertGreaterEqual(min(float(r["gamma"]) for r in raw), 0)
        self.assertTrue(all(0 < float(r["logAJ"]) < 100 and 1 < float(r["g"]) < 3 for r in raw))
        if d["passed"]:
            self.assertLessEqual(max(d["rhat"]), 1.01)
            self.assertGreaterEqual(min(d["ess_bulk"]), 400)
            self.assertGreaterEqual(min(d["ess_tail"]), 400)
            self.assertEqual(d["divergences"], 0)
            self.assertEqual(d["max_depth_hits"], 0)
            self.assertTrue((BASE / "latent_predictions.csv").exists())
        else:
            self.assertFalse((BASE / "latent_predictions.csv").exists())


if __name__ == "__main__":
    unittest.main()
