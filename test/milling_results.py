"""Independent retained-data contracts for milling MC/PCE; no Julia solves."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(folder, name):
    return np.genfromtxt(folder / name, delimiter=",", names=True, encoding="utf-8")


def outputs(folder, name):
    with gzip.open(folder / name, "rt") as stream:
        return np.loadtxt(stream, delimiter=",", skiprows=1)


def check(folder, *, smoke=False, retained=False):
    m = json.loads((folder / "provenance.json").read_text())
    assert m["complete"] and m["mode"] == ("smoke" if smoke else "production")
    for name, digest in m["numerical_files"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest, name
    if not retained:
        for name, digest in m["source_hashes"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    n = m["settings"]["mc_samples"]
    assert n == (64 if smoke else 12000)
    times = load(folder, "times.csv")["time_s"]
    assert np.all(np.diff(times) > 0) and {29.999, 30, 30.001, 31, 200} <= set(times)
    inputs = load(folder, "mc_inputs.csv")
    y = outputs(folder, "mc_outputs.csv.gz")
    assert y.shape == (n, len(times)+1) and np.all(np.isfinite(y))
    np.testing.assert_array_equal(y[:, 0], inputs["sample_id"])
    means = [.44, 8.82e-6, .34, 2.1]
    sds = [.07, 1.01e-6, .08, .12]
    names = ["delta", "alpha", "gamma", "sigma"]
    for name, mean, sd in zip(names, means, sds):
        np.testing.assert_allclose(inputs[name], mean + sd*inputs["z_"+name], rtol=3e-15)
    summary = load(folder, "mc_summary.csv")
    np.testing.assert_allclose(summary["mean_um"], y[:, 1:].mean(axis=0), rtol=2e-14)
    np.testing.assert_allclose(summary["sd_um"], y[:, 1:].std(axis=0, ddof=1), rtol=2e-14)
    np.testing.assert_allclose(summary["q025_um"], np.quantile(y[:, 1:], .025, axis=0), rtol=2e-14)
    np.testing.assert_allclose(summary["q975_um"], np.quantile(y[:, 1:], .975, axis=0), rtol=2e-14)
    np.testing.assert_allclose(summary["mean_mcse_um"], summary["sd_um"]/np.sqrt(n), rtol=2e-15)
    validation = load(folder, "validation_inputs.csv")
    vy = outputs(folder, "validation_outputs.csv.gz")[:, 1:]
    zval = np.array([validation["z_"+k] for k in names]).T
    pdraws = outputs(folder, "pce_propagation_inputs.csv.gz")
    for method, degree, terms, points in (("pce1", 1, 5, 12), ("pce2", 2, 15, 32), ("pce1_nine", 1, 5, 9)):
        training = load(folder, method+"_training_inputs.csv")
        assert len(training) == points
        ztrain = np.array([training["z_"+k] for k in names]).T
        assert not set(map(tuple, ztrain)) & set(map(tuple, zval))
        coefficients = load(folder, method+"_coefficients.csv")
        assert len(coefficients) == terms
        indices = np.array([coefficients["order_"+k] for k in names], dtype=int).T
        assert np.max(indices.sum(axis=1)) == degree
        c = np.array([coefficients["t"+str(k)] for k in range(1, len(times)+1)]).T

        def basis(z):
            # Independent explicit degree <= 2 polynomials, no production recurrence.
            phi = np.ones((len(z), terms))
            for j, multi in enumerate(indices):
                for k, power in enumerate(multi):
                    phi[:, j] *= (1 if power == 0 else z[:, k] if power == 1 else (z[:, k]**2-1)/np.sqrt(2))
            return phi

        ps = load(folder, method+"_summary.csv")
        np.testing.assert_allclose(ps["mean_um"], c[0], rtol=2e-14)
        np.testing.assert_allclose(ps["sd_um"]**2, np.sum(c[1:]**2, axis=0), rtol=2e-14)
        validation_errors = load(folder, method+"_validation.csv")
        np.testing.assert_allclose(validation_errors["rmse_um"], np.sqrt(np.mean((basis(zval)@c-vy)**2, axis=0)), rtol=2e-12)
        cheap = basis(pdraws[:, 1:])@c
        np.testing.assert_allclose(ps["q025_um"], np.quantile(cheap, .025, axis=0), rtol=2e-12)
        np.testing.assert_allclose(ps["q975_um"], np.quantile(cheap, .975, axis=0), rtol=2e-12)
        diag = json.loads((folder / (method+"_diagnostics.json")).read_text())
        assert diag["rank"] == terms and diag["condition"] < 1e8
    records = load(folder, "solve_checks.csv")
    assert len(records) == n + m["settings"]["validation_samples"] + 12 + 32 + 9
    assert np.max(records["maximum_mass_error_kg"]) < 1e-9
    assert np.min(records["minimum_inventory"]) >= -1e-12
    exact = load(folder, "matrix_exponential.csv")
    assert np.max(np.abs(exact["d50_difference_um"])) < 1e-3
    assert m["full_solves"] == len(records)+1
    print(f"Verified {n} MC samples, 53 PCE training solves, independent validation, "
          f"orthonormal moments, retained output quantiles and material balance. "
          f"{'Producer-file integrity only.' if retained else 'Current recorded sources match.'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--retained", action="store_true", help="Verify original numerical assets; do not claim current-source equivalence")
    args = parser.parse_args()
    folder = ROOT / ".cache/milling_uncertainty/smoke" if args.smoke else ROOT / "docs/src/assets/milling_uncertainty/data"
    check(folder, smoke=args.smoke, retained=args.retained)
