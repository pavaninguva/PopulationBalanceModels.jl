"""Independent data/physical contracts for the completed Tang batch exports."""
import csv
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
BASE=ROOT/"paper/results/04_protein_crystallization/production"

def rows(name):
    with (BASE/name).open(newline="") as stream:
        return list(csv.DictReader(stream))

class ProteinResults(unittest.TestCase):
    def test_provenance(self):
        manifest=json.loads((BASE/"manifest.json").read_text())
        self.assertEqual(manifest["mode"],"production")
        self.assertTrue(manifest["complete"])
        for path,digest in manifest["files"].items():
            self.assertEqual(hashlib.sha256((BASE/path).read_bytes()).hexdigest(),digest,path)
        for path,digest in manifest["source_files"].items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),digest,path)
        data=ROOT/"examples/data/protein_crystallization/tang2018_isothermal"
        origin=json.loads((data/"provenance/integration.json").read_text())
        for path,digest in origin["data_hashes"].items():
            self.assertEqual(hashlib.sha256((data/path).read_bytes()).hexdigest(),digest,path)

    def test_split_and_loss(self):
        conc=rows("concentration_predictions.csv")
        self.assertEqual(len(conc),16)
        self.assertEqual(sum(r["split"]=="calibration" for r in conc),11)
        self.assertEqual(sum(r["split"]=="validation" for r in conc),4)
        self.assertEqual(float(conc[0]["time_min"]),0.)
        self.assertEqual(float(conc[0]["residual_g_L"]),0.)
        parts=rows("fit_contributions.csv")
        self.assertEqual([p["name"] for p in parts],["concentration","PSD_74","PSD_180","PSD_420"])
        frozen=json.loads((BASE/"frozen_model.json").read_text())
        self.assertFalse(frozen["validation_used"])
        metrics=json.loads((BASE/"validation_metrics.json").read_text())
        self.assertLess(frozen["frozen_at"],metrics["evaluated_at"])
        residual=np.array([float(r["residual_g_L"]) for r in conc if r["split"]=="calibration"])
        # The dense reporting schedule and fit schedule can differ by tiny
        # third-order temporal errors; this is not a second fitted dataset.
        self.assertAlmostEqual(np.mean((residual/.1)**2),float(parts[0]["value"]),delta=.01)

    def test_balance_and_distributions(self):
        mass=rows("material_balance.csv")
        self.assertLess(max(abs(float(r["material_error_g_L"])) for r in mass),1e-9)
        self.assertGreaterEqual(min(float(r["minimum_scaled_count"]) for r in mass),0.)
        curves=rows("distribution_curves.csv")
        observations=rows("psd_predictions.csv")
        recorded=rows("distribution_metrics.csv")
        checks=[]
        for t in (74.,180.,420.,660.):
            r=[r for r in curves if float(r["time_min"])==t]
            left=np.array([float(a["left_um"]) for a in r]);right=np.array([float(a["right_um"]) for a in r])
            density=np.array([float(a["conditional_density_per_um"]) for a in r])
            select=density>0
            self.assertAlmostEqual(np.dot(right-left,density),1.,places=11)
            # Independent inverse CDF of the exact exported cell integrals.
            cdf=np.r_[0.,np.cumsum((right-left)*density)]
            edges=np.r_[left,right[-1]]
            # Collapse flats using first/last support edges rather than
            # inventing probability in empty cells.
            support=np.flatnonzero(select)
            ilo,ihi=support[0],support[-1]+1
            o=[a for a in observations if float(a["time_min"])==t]
            fo=np.array([float(a["observed_cdf"]) for a in o]);xo=np.array([float(a["size_um"]) for a in o])
            cm=cdf[ilo:ihi+1]/cdf[ihi] # normalized exported bin integrals
            partition=np.unique(np.r_[cm,fo])
            widths=np.diff(partition); mid=(partition[:-1]+partition[1:])/2
            # Two-point Gauss integrates the squared affine quantile difference
            # exactly on each common CDF interval, independently of midpoint W₂.
            u=np.vstack((mid-widths/(2*np.sqrt(3)),mid+widths/(2*np.sqrt(3))))
            qm=np.interp(u,cm,edges[ilo:ihi+1])
            qo=np.interp(u,fo,xo)
            w2=np.sqrt(np.dot(widths,np.sum((qm-qo)**2,axis=0)/2))
            reported=float(next(a["W2_um"] for a in recorded if float(a["time_min"])==t))
            self.assertAlmostEqual(w2,reported,delta=.0025)
            checks.append(dict(time_min=t,piecewise_exact_W2_um=w2,midpoint1024_W2_um=reported,
                               quadrature_difference_um=reported-w2))
        settings=json.loads((BASE/"run_settings.json").read_text())
        self.assertFalse(settings["comparison_scales_are_sensor_standard_deviations"])
        output=BASE/"independent_checks"
        output.mkdir(exist_ok=True)
        with (output/"wasserstein.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(checks[0]))
            writer.writeheader();writer.writerows(checks)

if __name__=="__main__":
    unittest.main()
