"""Regenerate manuscript figures only from validated paper-mode numerical exports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from plotting_style import configure,save_png,METHODS
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
CASES = ["growth","moments","precipitation"]
ORIGINAL_CASES = ["01_growth_time_methods","02_aggregation_fragmentation","03_tubular_precipitation"]
MODE = "production"
PREVIEW = False
RETAINED = False
INPUTS = {}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(files):
    content = "".join(f"{p.relative_to(ROOT).as_posix()}\0{sha(p)}\n" for p in sorted(files))
    return hashlib.sha256(content.encode()).hexdigest()

def source_digest():
    return digest([p for d in ("src","ext","examples") for p in (ROOT/d).rglob("*.jl")]+[ROOT/"Project.toml"])

def rows(path):
    with open(path,newline="") as stream:
        return list(csv.DictReader(stream))

def table(path):
    return np.genfromtxt(path,delimiter=",",names=True,encoding="utf-8",dtype=None)

def relative_difference(value,reference):
    return value/reference-1 if reference != 0 else np.nan

def density_l1(edges,density,reference_edges,reference):
    """Normalized L1 integral of two PWC densities on their common partition."""
    if not np.allclose([edges[0],edges[-1]],[reference_edges[0],reference_edges[-1]],rtol=1e-12,atol=0):
        raise ValueError("L1 comparison requires identical physical domains")
    union = np.unique(np.r_[edges,reference_edges])
    middle = (union[:-1]+union[1:])/2
    left = np.clip(np.searchsorted(edges,middle,side="right")-1,0,len(density)-1)
    right = np.clip(np.searchsorted(reference_edges,middle,side="right")-1,0,len(reference)-1)
    normalization = np.dot(np.diff(reference_edges),reference)
    if normalization <= 0:
        raise ValueError("Reference inventory must be positive")
    return np.dot(np.diff(union),np.abs(density[left]-reference[right]))/normalization

def analysis_table(number,base,name,data):
    """Plot-derived summaries stay separate from immutable solver exports."""
    folder = base/"analysis"
    folder.mkdir(exist_ok=True)
    path = folder/name
    with open(path,"w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(data[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)
    INPUTS[number][str(path.relative_to(ROOT))] = sha(path)

def load_case(number, *, check_checkout=None):
    """Validate recorded inputs; current producer equivalence is required by default.

    The independent checker can require current-producer equality. Figure-only
    entry points validate saved payloads and retain the original producer.
    """
    if check_checkout is None:
        check_checkout = not RETAINED
    name = CASES[number-1]
    base = ROOT/".cache/paper/checks"/name if MODE == "smoke" else PAPER/name/"data"
    index_path = base/"index.toml"
    info = tomllib.loads(index_path.read_text())
    if info["mode"] != MODE or info["case"] not in (name, ORIGINAL_CASES[number-1]):
        raise ValueError("Incorrect result mode or case")
    if check_checkout and info["source_digest"] != source_digest():
        raise ValueError("Numerical source differs from the production provenance; regenerate results")
    support = [PAPER/"_support/common.jl",PAPER/name/"run.jl",PAPER/name/"config.toml"]
    if number == 3:
        support.append(PAPER/"_support/precipitation_solver.jl")
    if check_checkout and info["workflow_digest"] != digest(support):
        raise ValueError("Driver/configuration differs from the retained results")
    if number == 1:
        cfg=info['configuration']
        if cfg.get('profile_family')!='gaussian':
            raise ValueError('The active growth workflow requires fresh Gaussian results; run paper/growth/run.jl')
        expected = {f"smooth_{m}_N{n}" for m in METHODS for n in cfg["bins"]}
        for method in METHODS:
            courant=1.0 if method=='inguva2022' else cfg['courant']
            expected |= {gaussian_key(method,n,courant) for n in cfg['bins']}
            expected |= {gaussian_key(method,cfg['profile_bins'],c) for c in cfg['profile_courants']
                         if method!='inguva2022' or c==1.0}
    elif number == 2:
        expected = {m+f"_dt{r}" for m in MOMENT_METHODS for r in (1,2)} | {"qmom4_dt1"}
    else:
        cfg = info["configuration"]
        if not {"size_bins", "axial_cells"} <= cfg.keys():
            raise ValueError("The active tubular workflow requires the independent logarithmic-grid study; run paper/precipitation/run.jl")
        expected = {"reference", "time_refinement", "solver_verification"}
        expected |= {f"size_Nx{n}" for n in cfg["size_bins"]}
        expected |= {f"axial_Nz{n}" for n in cfg["axial_cells"]}
    missing = expected-set(info["runs"])
    if missing:
        raise ValueError(f"Incomplete paper dataset: missing {sorted(missing)}")
    info = dict(info, checkout_verified=bool(check_checkout), retained_redraw=not check_checkout)
    runs = {}
    compute_environments = set()
    for key,relative in info["runs"].items():
        folder = base/relative
        provenance = tomllib.loads((folder/"provenance.toml").read_text())
        if not provenance["complete"] or provenance["mode"] != MODE:
            raise ValueError(f"Incomplete or smoke input: {folder}")
        if provenance["source_digest"] != info["source_digest"] or provenance["workflow_digest"] != info["workflow_digest"]:
            raise ValueError("Mixed source states in one figure")
        if number == 3 and key != "solver_verification":
            settings = provenance["settings"]
            required_mesh = (info["configuration"]["reference_nx"],
                             info["configuration"]["reference_nz"])
            mesh = (settings["nx"], settings["nz"])
            expected_mesh = ((int(key.removeprefix("size_Nx")), required_mesh[1])
                             if key.startswith("size_Nx") else
                             (required_mesh[0], int(key.removeprefix("axial_Nz")))
                             if key.startswith("axial_Nz") else required_mesh)
            if mesh != expected_mesh:
                raise ValueError(f"{key} changes the wrong refinement coordinate: {mesh}, expected {expected_mesh}")
            if (settings["size_reconstruction"] != "geometry-aware WENO5" or
                    settings["axial_reconstruction"] != "MUSCL" or
                    settings["mesh"] != "logarithmic sizes; uniform axial"):
                raise ValueError(f"{key} is not the declared logarithmic WENO5 / axial MUSCL calculation")
        env = provenance["environment"]
        compute_environments.add((provenance["environment_digest"],env["julia"],env["cpu"],
                                  env["julia_threads"],env["blas_threads"]))
        for item in provenance["files"]:
            if sha(folder/item["path"]) != item["sha256"]:
                raise ValueError(f"Changed numerical data: {folder/item['path']}")
        runs[key] = (folder,provenance)
    if len(compute_environments) != 1:
        raise ValueError("Mixed compute environments: archive these results and rerun before comparing timings")
    for item in info.get("derived_files",[]):
        if sha(base/item["path"]) != item["sha256"]:
            raise ValueError("Changed derived numerical table")
    INPUTS[number] = {str(index_path.relative_to(ROOT)):sha(index_path)}
    for folder,_ in runs.values():
        path = folder/"provenance.toml"
        INPUTS[number][str(path.relative_to(ROOT))] = sha(path)
    if number == 3:
        for name in ("cadet_reference.csv","zhang_figure9c_digitized.csv","provenance.json"):
            path = PAPER/"precipitation/data/literature"/name
            INPUTS[number][str(path.relative_to(ROOT))] = sha(path)
    return base,info,runs

def gaussian_key(method,bins,courant):
    tag=str(float(courant)).replace('.', 'p')
    return f'gaussian_{method}_C{tag}_N{bins}'

def export(fig,number,name,info):
    folder = ROOT/".cache/paper/figures"/CASES[number-1] if MODE == "smoke" else PAPER/CASES[number-1]/"figures"
    folder.mkdir(parents=True,exist_ok=True)
    stem = name+("_PREVIEW" if PREVIEW else "")
    image=save_png(fig,folder/(stem+".png"))
    provenance = dict(mode=MODE,renderer="Matplotlib + external LaTeX" if not PREVIEW else "fallback preview",
        source_digest=info["source_digest"],workflow_digest=info["workflow_digest"],
        checkout_verified=info.get("checkout_verified",True),retained_redraw=info.get("retained_redraw",False),
        inputs=INPUTS[number],plotter_sha256=sha(PAPER/CASES[number-1]/"plot.py"),
        shared_plotting_sha256=sha(__file__),style_sha256=sha(PAPER/"plotting_style.py"),
        requirements_sha256=sha(PAPER/"requirements.txt"),
        python=sys.version,matplotlib=__import__("matplotlib").__version__,numpy=np.__version__,
        pillow=__import__("PIL").__version__,pymupdf=__import__("pymupdf").VersionBind,
        latex_version=(subprocess.check_output(["latex","--version"],text=True).splitlines()[0]
                       if not PREVIEW else "not used; fallback preview"),
        display_conventions=info.get("display_conventions",{}),
        inches=list(fig.get_size_inches()),**image,
        files={".png":sha(folder/(stem+".png"))})
    (folder/(stem+".json")).write_text(json.dumps(provenance,indent=2)+"\n")
    plt.close(fig)
    print("Rendered",folder/stem,flush=True)


MOMENT_METHODS = ["qmom3","adaptive3","eqmom3","dqmom3","lognormal"]
MOMENT_LABELS = ["QMOM(3)","Rank-adaptive QMOM(3)","Gamma EQMOM(3)","DQMOM(3)","Assumed lognormal"]

def run_plot(number, renderer, argv=None):
    """Common CLI; retained redraws validate original payloads, never relabel producers."""
    global MODE, PREVIEW, RETAINED
    parser=argparse.ArgumentParser(description=renderer.__doc__)
    parser.add_argument('--retained',action='store_true',help='Redraw recorded production exports; do not assert current producer equality')
    parser.add_argument('--smoke',action='store_true',help='Ignored small-check output, never a production figure')
    parser.add_argument('--preview',action='store_true')
    args=parser.parse_args(argv)
    MODE='smoke' if args.smoke else 'production'
    PREVIEW=args.preview; RETAINED=True
    if PREVIEW and MODE!='smoke': parser.error('--preview requires --smoke')
    if RETAINED: print('Redrawing retained exports; original producer provenance is preserved.',flush=True)
    configure(PREVIEW)
    renderer()
