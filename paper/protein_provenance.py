"""Read-only validation of separately recorded retained-target compatibility.

Original manifests remain authoritative for producer identity and file integrity.
A receipt is valid only for the exact checkout/configuration it actually checked.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "paper/results/04_protein_crystallization/analysis"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numerical_sources():
    paths = [ROOT / "examples/isothermal_protein_crystallization.jl"]
    paths += [p for d in ("src", "ext") for p in (ROOT / d).rglob("*.jl")]
    data = ROOT / "examples/data/protein_crystallization/tang2018_isothermal"
    paths += [data / f for f in ("concentration.csv", "psd_cdf.csv", "psd_summary.csv")]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def verify_compatibility(folder, manifest):
    path = ANALYSIS / folder.name / "compatibility.json"
    if not path.exists():
        raise ValueError("Changed target sources: run run.jl --compatibility-only first; "
                         "do not replace the original producer manifest")
    receipt = json.loads(path.read_text())
    if not receipt["passed"] or receipt["model"] != folder.name:
        raise ValueError("Unsuccessful or mismatched compatibility evidence")
    if receipt["original_target_hash"] != manifest["target_hash"]:
        raise ValueError("Compatibility evidence belongs to another target")
    review = receipt["source_review"]
    if review["original_sources"] != manifest["source_files"]:
        raise ValueError("Compatibility evidence changed the original producer sources")
    if review["current_sources"] != numerical_sources():
        raise ValueError("Sources changed since the compatibility evaluation; re-evaluate")
    uncertainty_review = review.get("uncertainty_review_sha256")
    if uncertainty_review is not None:
        path = ROOT / "paper/case_studies/04_protein_crystallization/uncertainty_compatibility.json"
        if sha(path) != uncertainty_review:
            raise ValueError("Uncertainty compatibility review changed; re-evaluate")
    for name, digest in receipt["inputs"].items():
        if sha(folder / name) != digest:
            raise ValueError(f"Changed compatibility input: {name}")
    for name, digest in receipt["projects"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Environment changed since compatibility evaluation: {name}")
    checks = receipt["checks"]
    if not checks or checks[0]["kind"] != "MAP" or len(checks) < 9:
        raise ValueError("Incomplete numerical compatibility evidence")
    for check in checks:
        if abs(check["logdensity_error"]) >= 1e-6 or check["gradient_error"] >= 1e-3:
            raise ValueError("Posterior/gradient compatibility failed")
    return receipt
