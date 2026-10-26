"""Independent checks of the completed synthetic stem-cell case-study exports.

Run with Python 3.11+ after examples/stem_cell_population.jl. No plotting or
Julia simulation is performed here; the analytic cascade is tested in Julia.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import math
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def rows(folder, filename):
    with (folder / filename).open(newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual, expected, *, atol=1e-12, rtol=1e-12):
    assert math.isfinite(actual) and math.isfinite(expected)
    assert abs(actual - expected) <= atol + rtol * abs(expected), (actual, expected)


def check(folder):
    manifest = tomllib.loads((folder / "run.toml").read_text())
    assert manifest["mode"] == "production"
    assert manifest["settings"]["tf"] == 200
    assert manifest["reference_DOI"] == "10.1186/s13287-025-04668-w"
    for filename, digest in manifest["source_hashes"].items():
        if hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest:
            continue
        # The original production Project is retained verbatim. Permit only
        # development/test declarations to differ, never a runtime dependency,
        # extension, package identity or runtime compatibility requirement.
        assert filename == "Project.toml" and "project_snapshot" in manifest, filename
        snapshot = (folder / manifest["project_snapshot"]).read_bytes()
        assert hashlib.sha256(snapshot).hexdigest() == digest
        original, current = tomllib.loads(snapshot.decode()), tomllib.loads((ROOT / filename).read_text())
        for key in set(original) | set(current):
            if key not in ("extras", "targets", "compat"):
                assert original.get(key) == current.get(key), key
        runtime_names = set(original["deps"]) | set(original.get("weakdeps", {})) | {"julia"}
        assert {k: v for k, v in original["compat"].items() if k in runtime_names} == {
            k: v for k, v in current["compat"].items() if k in runtime_names}
    for filename, digest in manifest["files"].items():
        assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == digest, filename

    for name in ("self_renewal", "differentiation"):
        trajectory = rows(folder, f"trajectory_{name}.csv")
        distributions = rows(folder, f"distribution_{name}.csv")
        times = sorted({float(r["time"]) for r in distributions})
        assert times == [2.0 * i for i in range(101)]
        by_time = {time: [] for time in times}
        for row in distributions:
            by_time[float(row["time"])].append(row)
        initial_cargo = float(trajectory[0]["cargo"])
        initial_cells = float(trajectory[0]["stem"])
        close(initial_cells, 1)
        expected_initial_cargo = 0.0
        for record in by_time[0.0]:
            left, right = float(record["left"]), float(record["right"])
            a, b = max(0.0, left), min(100.0, right)
            # Independent normal-bin formula. The moment of a bin-centred
            # representation is not the exact continuum mean on every mesh.
            if b <= a:
                expected = 0.0
            elif a >= 50:
                expected = (math.erfc((a-50)/math.sqrt(0.5)) -
                            math.erfc((b-50)/math.sqrt(0.5))) / 2
            else:
                expected = (math.erfc((50-b)/math.sqrt(0.5)) -
                            math.erfc((50-a)/math.sqrt(0.5))) / 2
            close(float(record["count"]), expected, atol=3e-15)
            expected_initial_cargo += (left + right) / 2 * expected
        close(initial_cargo, expected_initial_cargo)
        for record in trajectory:
            t = float(record["time"])
            cells = by_time[t]
            assert len(cells) == manifest["settings"]["bins"]
            lefts = [float(r["left"]) for r in cells]
            rights = [float(r["right"]) for r in cells]
            assert all(b > a for a, b in zip(lefts, rights))
            assert all(a == b for a, b in zip(rights[:-1], lefts[1:]))
            assert lefts[0] == 0 and rights[-1] == manifest["settings"]["xmax"]
            number = math.fsum(float(r["count"]) for r in cells)
            cargo = math.fsum((a + b) / 2 * float(r["count"])
                              for a, b, r in zip(lefts, rights, cells))
            assert all(float(r["count"]) >= 0 for r in cells)
            probability = math.fsum((b - a) * float(r["normalized_density"])
                                    for a, b, r in zip(lefts, rights, cells))
            close(probability, 1)
            close(number, float(record["stem"]))
            close(cargo, float(record["cargo"]))
            close(number + float(record["differentiated"]) + float(record["lost_number"]),
                  initial_cells + float(record["divisions"]), rtol=2e-11)
            close(cargo + float(record["exit_cargo"]) + float(record["lost_cargo"]),
                  initial_cargo + float(record["synthesized"]), rtol=2e-11)
            assert float(record["upper_loss"]) == 0
            assert float(record["upper_bin_fraction"]) < 1e-8
            assert float(record["q10"]) <= float(record["q50"]) <= float(record["q90"])
            if name == "self_renewal":
                assert float(record["differentiated"]) == float(record["exit_cargo"]) == 0

        events = rows(folder, f"events_{name}.csv")
        for key in ("total", "dividing", "newborn"):
            assert all(float(r[key]) >= 0 for r in events)
            integral = math.fsum((float(r["right"]) - float(r["left"])) * float(r[key])
                                 for r in events)
            # The newborn PDF is normalized by all births, including unresolved
            # daughters. The delivered run is explicitly checked to lose <1e-10.
            close(integral, 1, atol=1e-10)
        if name == "differentiation":
            assert all(float(r["differentiating"]) >= 0 for r in events)
            close(math.fsum((float(r["right"]) - float(r["left"])) * float(r["differentiating"])
                            for r in events), 1)
        for record in events:
            x = float(record["x"])
            close(float(record["R"]), (50 + x*x) / (100 + x*x) - 0.005*x)
            close(float(record["gamma"]), (x / 100)**5)
            kappa = 0 if name == "self_renewal" else 0.5 / (1 + 0.005 * math.exp(0.25*x))
            close(float(record["kappa"]), kappa)
            if name == "self_renewal":
                assert math.isnan(float(record["differentiating"]))
            if record["central"].lower() == "true":
                close(float(record["gamma_recovered"]), (x / 100)**5)
                close(float(record["kappa_recovered"]), kappa)
            else:
                assert math.isnan(float(record["R_recovered"]))

    refinement = rows(folder, "refinement.csv")
    checks = {"coarse_grid", "fine_grid", "time", "daughter_quadrature", "domain"}
    for name in ("self_renewal", "differentiation"):
        subset = [r for r in refinement if r["scenario"] == name]
        assert {r["check"] for r in subset} == checks
        for record in subset:
            for key in ("maximum_CDF_difference", "maximum_relative_number_difference",
                        "maximum_mean_cargo_difference", "seconds"):
                assert math.isfinite(float(record[key])) and float(record[key]) >= 0
    print("Stem-cell exports: provenance, normalization, event/cargo balances and refinement records pass")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "docs/src/assets/stem_cells")
    check(parser.parse_args().data_dir)
