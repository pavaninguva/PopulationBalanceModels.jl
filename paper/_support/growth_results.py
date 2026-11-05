"""One Gaussian benchmark for retained, paper and manual growth comparisons.

This case-specific view bundles small figure tables, not solver histories. It
never solves a PBM and never modifies its numerical producers.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tomllib

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "docs/src/assets/constant_growth"
METHODS = ("upwind", "muscl", "weno3", "weno5_ssprk33", "weno5", "hoc", "inguva2022")
BENCHMARK = dict(profile="gaussian", height=100.0, center=0.2,
                 standard_deviation=0.03, growth=0.1, final_time=4.0,
                 domain=[0.0, 1.0], boundary="zero inflow / open outflow")
METRICS = ("method", "profile", "bins", "courant", "dt", "tf", "rmse", "l1",
           "maximum_error", "minimum_density", "negative_number", "number_drift",
           "seconds", "steps", "bytes", "gc_seconds")
PROFILES = ("method", "courant", "bins", "left", "right", "number", "density",
            "exact_number", "exact_density")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(path):
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf8"))


def csv_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def refinement(data, method):
    c = 1.0 if method == "inguva2022" else float(data["configuration"]["courant"])
    rows = data["metrics"]
    rows = rows[(rows["method"] == method) & (rows["courant"] == c)
                & np.isin(rows["bins"], data["configuration"]["bins"])]
    return np.sort(rows, order="bins")


def profile(data, method, courant):
    rows = data["profiles"]
    return rows[(rows["method"] == method) & (rows["courant"] == courant)]


def observed_orders(data):
    """Measured slopes, screened using the density/number units of each norm.

    This conservative floating-point screen is a display/analysis convention,
    not a solver tolerance, an accuracy certificate or an error correction.
    """
    cfg, benchmark = data["configuration"], data["benchmark"]
    factor = float(cfg.get("roundoff_factor", 1000.0)) * np.finfo(float).eps
    rms_floor = factor * benchmark["height"]
    l1_floor = rms_floor * (benchmark["domain"][1] - benchmark["domain"][0])
    result = []
    for method in METHODS:
        previous = None
        for row in refinement(data, method):
            orders = {}
            for error, floor in (("rmse", rms_floor), ("l1", l1_floor)):
                resolved = previous is not None and min(previous[error], row[error]) > floor
                orders["observed_" + error + "_order"] = (
                    float(np.log(previous[error] / row[error]) / np.log(row["bins"] / previous["bins"]))
                    if resolved else float("nan"))
            result.append(dict(method=method, bins=int(row["bins"]), rmse=float(row["rmse"]),
                               l1=float(row["l1"]), **orders,
                               roundoff_limit_rmse=rms_floor, roundoff_limit_l1=l1_floor))
            previous = row
    return result


def validate(data):
    if data["benchmark"] != BENCHMARK:
        raise ValueError("Expected the declared Gaussian, G=0.1, t=4 and zero-inflow/open-outflow benchmark")
    cfg = data["configuration"]
    metrics, profiles = data["metrics"], data["profiles"]
    if set(metrics["profile"]) != {"gaussian"}:
        raise ValueError("The growth comparison must contain Gaussian runs only")
    if set(metrics["method"]) != set(METHODS):
        raise ValueError("Missing or unexpected growth/time pairing")
    if np.any(metrics["tf"] != BENCHMARK["final_time"]):
        raise ValueError("Mixed final times in the Gaussian comparison")
    for method in METHODS:
        rows = refinement(data, method)
        if list(rows["bins"]) != list(cfg["bins"]):
            raise ValueError(f"Incomplete or duplicated Gaussian refinement: {method}")
        for c in cfg["profile_courants"]:
            if method == "inguva2022" and c != 1.0:
                continue
            values = profile(data, method, c)
            if len(values) != cfg["profile_bins"] or np.any(values["bins"] != cfg["profile_bins"]):
                raise ValueError(f"Incomplete Gaussian profile: {method}, CFL={c}")
    for key in ("seconds", "rmse", "l1", "dt"):
        if not np.all(np.isfinite(metrics[key])) or np.any(metrics[key] < 0):
            raise ValueError(f"Invalid Gaussian metric: {key}")
    return data


def _from_paper(folder):
    from .figures import load_case
    mode=tomllib.loads((Path(folder)/'index.toml').read_text())['mode']
    base, info, runs = load_case(1, check_checkout=False, input_path=folder, mode=mode)
    metrics, profiles, origins = [], [], []
    cfg = info["configuration"]
    for key, (directory, producer) in runs.items():
        metric = csv_rows(directory / "metrics.csv")[0]
        metrics.append(metric)
        if int(metric["bins"]) == cfg["profile_bins"]:
            for row in csv_rows(directory / "profile.csv"):
                profiles.append(dict(method=metric["method"], courant=metric["courant"],
                                     bins=metric["bins"], **row))
        origins.append(dict(run=key, directory=str(directory.relative_to(base)),
                            revision=producer['revision'],
                            provenance_sha256=sha(directory / "provenance.toml"),
                            metrics_sha256=sha(directory / "metrics.csv"),
                            profile_sha256=sha(directory / "profile.csv")))
    first = next(iter(runs.values()))[1]
    provenance = dict(kind="retained Gaussian selection; no new numerical solves",
                      producer_revisions=sorted({entry['revision'] for entry in origins}),
                      index_sha256=sha(base / "index.toml"),
                      source_digest=info["source_digest"], workflow_digest=info["workflow_digest"],
                      environment=first["environment"], environment_digest=first["environment_digest"],
                      timing_protocol=first["timing_protocol"], original_runs=origins)
    return _arrays(metrics, profiles, cfg, provenance)


def _arrays(metrics, profiles, cfg, provenance):
    def array(rows, fields):
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.seek(0)
        return np.atleast_1d(np.genfromtxt(stream, delimiter=",", names=True, dtype=None, encoding="utf8"))
    return validate(dict(benchmark=BENCHMARK.copy(), configuration=cfg,
                         metrics=array(metrics, METRICS), profiles=array(profiles, PROFILES),
                         provenance=provenance))


def load_results(folder=None):
    """Read the bundled selection or an explicitly supplied new calculation."""
    folder = Path(folder) if folder is not None else DEFAULT
    if (folder / "index.toml").is_file():
        return _from_paper(folder)
    selection = folder / "gaussian_selection.json"
    source = folder / "gaussian_source.toml"
    if selection.is_file():
        info = json.loads(selection.read_text())
    elif source.is_file():
        info = tomllib.loads(source.read_text())
    else:
        raise FileNotFoundError(f"Missing Gaussian figure selection in {folder}; supply --input from a complete Gaussian calculation")
    for name, digest in info["files"].items():
        if sha(folder / name) != digest:
            raise ValueError(f"Changed Gaussian figure input: {folder / name}")
    result = validate(dict(benchmark=info["benchmark"], configuration=info["configuration"],
                           metrics=table(folder / "gaussian_metrics.csv"),
                           profiles=table(folder / "gaussian_profiles.csv"), provenance=info["provenance"]))
    result["selection_files"] = {name: sha(folder / name) for name in
                                  ("gaussian_metrics.csv", "gaussian_profiles.csv", selection.name if selection.is_file() else source.name)}
    return result


def write_selection(data, folder):
    """New derived plotting tables; original producer files remain untouched."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    def write(name, rows, fields):
        with (folder / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    def rows(array):
        return [{name: row[name].item() for name in array.dtype.names} for row in array]
    write("gaussian_metrics.csv", rows(data["metrics"]), METRICS)
    write("gaussian_profiles.csv", rows(data["profiles"]), PROFILES)
    orders = observed_orders(data)
    write("gaussian_orders.csv", orders, list(orders[0]))
    files = {name: sha(folder / name) for name in
             ("gaussian_metrics.csv", "gaussian_profiles.csv", "gaussian_orders.csv")}
    info = dict(schema=1, benchmark=data["benchmark"], configuration=data["configuration"],
                provenance=data["provenance"], files=files)
    (folder / "gaussian_selection.json").write_text(json.dumps(info, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="A complete Gaussian calculation directory")
    parser.add_argument("--output", type=Path, required=True, help="A separate derived figure-selection directory")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Keep the derived selection separate from original numerical producers")
    write_selection(load_results(args.input), args.output)


if __name__ == "__main__":
    # Also allow the source file to be invoked directly from a checkout.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "_support"
    main()
