"""Regenerate manuscript figures only from validated paper-mode numerical exports."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from plotting_style import configure,panel,method_style,DOUBLE,METHODS
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from PIL import Image
import pymupdf

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
CASES = ["01_growth_time_methods","02_aggregation_fragmentation","03_tubular_precipitation"]
MODE = "production"
PREVIEW = False
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
        writer = csv.DictWriter(stream,fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    INPUTS[number][str(path.relative_to(PAPER))] = sha(path)

def load_case(number, *, check_checkout=True):
    """Validate recorded inputs; current producer equivalence is required by default.

    The independent checker may explicitly inspect retained historical data
    without requiring current source equality. Plot generation keeps the default.
    """
    name = CASES[number-1]
    base = PAPER/"results"/name/MODE
    index_path = base/"index.toml"
    info = tomllib.loads(index_path.read_text())
    if info["mode"] != MODE or info["case"] != name:
        raise ValueError("Incorrect result mode or case")
    if check_checkout and info["source_digest"] != source_digest():
        raise ValueError("Numerical source differs from the production provenance; regenerate results")
    support = [PAPER/"common.jl",PAPER/"case_studies"/name/"run.jl",PAPER/"case_studies"/name/"config.toml"]
    if number == 3:
        support.append(PAPER/"precipitation_solver.jl")
    if check_checkout and info["workflow_digest"] != digest(support):
        raise ValueError("Driver/configuration differs from the retained results")
    if number == 1:
        expected = {f"front_{m}_N{info['configuration']['profile_bins']}" for m in METHODS}
        expected |= {f"smooth_{m}_N{n}" for m in METHODS for n in info["configuration"]["smooth_bins"]}
    elif number == 2:
        expected = {m+f"_dt{r}" for m in MOMENT_METHODS for r in (1,2)} | {"qmom4_dt1"}
    else:
        expected = {"reference","time_refinement","size_refinement","axial_refinement","paper_field","solver_verification"}
        expected |= {f"outlet_Nz{n}" for n in info["configuration"]["outlet_nz"]}
    missing = expected-set(info["runs"])
    if missing:
        raise ValueError(f"Incomplete paper dataset: missing {sorted(missing)}")
    runs = {}
    compute_environments = set()
    for key,relative in info["runs"].items():
        folder = base/relative
        provenance = tomllib.loads((folder/"provenance.toml").read_text())
        if not provenance["complete"] or provenance["mode"] != MODE:
            raise ValueError(f"Incomplete or smoke input: {folder}")
        if provenance["source_digest"] != info["source_digest"] or provenance["workflow_digest"] != info["workflow_digest"]:
            raise ValueError("Mixed source states in one figure")
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
    INPUTS[number] = {str(index_path.relative_to(PAPER)):sha(index_path)}
    for folder,_ in runs.values():
        path = folder/"provenance.toml"
        INPUTS[number][str(path.relative_to(PAPER))] = sha(path)
    if number == 3:
        for name in ("cadet_reference.csv","zhang_figure9c_digitized.csv","provenance.json"):
            path = PAPER/"literature"/name
            INPUTS[number][str(path.relative_to(PAPER))] = sha(path)
    return base,info,runs

def export(fig,number,name,info):
    folder = PAPER/"figures"/CASES[number-1]/MODE
    folder.mkdir(parents=True,exist_ok=True)
    stem = name+("_PREVIEW" if PREVIEW else "")
    fig.savefig(folder/(stem+".pdf"),dpi=300,
                metadata={"CreationDate":None,"ModDate":None,
                          "Creator":"PopulationBalanceModels.jl paper workflow"})
    fig.savefig(folder/(stem+".png"),dpi=300)
    image = Image.open(folder/(stem+".png"))
    if any(abs(v-300)>.1 for v in image.info["dpi"]):
        raise ValueError("PNG lacks 300-dpi metadata")
    document = pymupdf.open(folder/(stem+".pdf"))
    font_records = []
    for page in document:
        for font in page.get_fonts(full=True):
            xref,ext,kind,basefont,*_ = font
            data = document.extract_font(xref)[3]
            if not data:
                raise ValueError(f"Unembedded PDF font: {basefont}")
            font_records.append(dict(name=basefont,type=kind,embedded_bytes=len(data)))
    document.close()
    provenance = dict(mode=MODE,renderer="Matplotlib + external LaTeX" if not PREVIEW else "fallback preview",
        source_digest=info["source_digest"],workflow_digest=info["workflow_digest"],
        inputs=INPUTS[number],plotter_sha256=sha(__file__),style_sha256=sha(PAPER/"plotting_style.py"),
        requirements_sha256=sha(PAPER/"requirements.txt"),
        python=sys.version,matplotlib=__import__("matplotlib").__version__,numpy=np.__version__,
        pillow=__import__("PIL").__version__,pymupdf=pymupdf.VersionBind,
        latex_version=(subprocess.check_output(["latex","--version"],text=True).splitlines()[0]
                       if not PREVIEW else "not used; fallback preview"),
        display_conventions=info.get("display_conventions",{}),
        inches=list(fig.get_size_inches()),png_pixels=image.size,png_dpi=image.info["dpi"],fonts=font_records,
        files={suffix:sha(folder/(stem+suffix)) for suffix in (".pdf",".png")})
    (folder/(stem+".json")).write_text(json.dumps(provenance,indent=2)+"\n")
    plt.close(fig)
    print("Rendered",folder/stem,flush=True)

def growth():
    base,info,runs = load_case(1)
    summaries = []
    for name,(folder,record) in runs.items():
        if name.startswith(("front_","smooth_")):
            summary = rows(folder/"metrics.csv")[0]
            summary["dataset"] = name
            summary["api_time_method"] = record["settings"]["api_time_method"]
            summaries.append(summary)
    analysis_table(1,base,"growth_comparison.csv",summaries)
    n = info["configuration"]["profile_bins"]
    if n != 100 or info["configuration"]["final_time"] != 4.0:
        raise ValueError("The manuscript profiles require 100 bins at t=4 s")
    fig,axes = plt.subplots(1,2,figsize=(DOUBLE,3.35))
    fig.subplots_adjust(left=.09,right=.99,bottom=.26,top=.93,wspace=.3)
    exact = table(runs[f"front_upwind_N{n}"][0]/"profile.csv")
    for ax in axes:
        ax.stairs(exact["exact_density"],np.r_[exact["left"],exact["right"][-1]],color=".45",lw=1.6,label="Analytical bin average")
    for method in METHODS:
        data = table(runs[f"front_{method}_N{n}"][0]/"profile.csv")
        x = (data["left"]+data["right"])/2
        for ax in axes:
            style = method_style(method,markers=False)
            ax.stairs(data["density"],np.r_[data["left"],data["right"][-1]],**style)
            marker = method_style(method)["marker"]
            ax.plot(x[::5],data["density"][::5],ls="none",marker=marker,ms=3,
                    color=style["color"],mfc="white",mew=.8)
    axes[0].set(xlim=(0,1),xlabel=r"Size coordinate $a$",ylabel=r"Bin-average number density $\bar f$")
    axes[1].set(xlim=(.32,.53),xlabel=r"Size coordinate $a$")
    for letter,ax in zip("ab",axes): panel(ax,letter)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=3)
    export(fig,1,"growth_profiles_N100",info)
    order = table(base/"observed_orders.csv")
    fig,axes = plt.subplots(2,2,figsize=(DOUBLE,5.3))
    fig.subplots_adjust(left=.1,right=.98,bottom=.16,top=.93,wspace=.32,hspace=.35)
    for method in METHODS:
        d = order[order["method"]==method]
        for col,(error,observed) in enumerate((("l1","observed_l1_order"),("rmse","observed_rmse_order"))):
            axes[0,col].loglog(d["bins"],np.where(d[error]>0,d[error],np.nan),**method_style(method))
            axes[1,col].semilogx(d["bins"],d[observed],**method_style(method))
    for col,norm in enumerate((r"$L^1$ error",r"RMSE")):
        axes[0,col].set(xlabel="Size bins",ylabel=norm)
        axes[1,col].set(xlabel="Size bins",ylabel=r"$p_{\mathrm{obs}}$")
    for letter,ax in zip("abcd",axes.flat): panel(ax,letter)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="lower center",ncol=3)
    export(fig,1,"growth_convergence",info)
    fig,axes = plt.subplots(1,2,figsize=(DOUBLE,3.3))
    fig.subplots_adjust(left=.1,right=.98,bottom=.25,top=.93,wspace=.32)
    for method in METHODS:
        points = []
        for bins in info["configuration"]["smooth_bins"]:
            data = rows(runs[f"smooth_{method}_N{bins}"][0]/"metrics.csv")[0]
            points.append((bins,float(data["seconds"]),float(data["rmse"])))
        points = np.array(points)
        axes[0].loglog(points[:,0],points[:,1],**method_style(method))
        axes[1].loglog(points[:,1],points[:,2],**method_style(method))
    axes[0].set(xlabel="Size bins",ylabel=r"Integration wall time (s)")
    axes[1].set(xlabel=r"Integration wall time (s)",ylabel="RMSE")
    for letter,ax in zip("ab",axes): panel(ax,letter)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=3)
    export(fig,1,"growth_work_precision",info)

MOMENT_METHODS = ["qmom3","adaptive3","eqmom3","dqmom3","lognormal"]
MOMENT_LABELS = ["QMOM(3)","Rank-adaptive QMOM(3)","Gamma EQMOM(3)","DQMOM(3)","Assumed lognormal"]

def moments():
    base,info,runs = load_case(2)
    resolution_relative = 1000*np.finfo(float).eps
    info["display_conventions"] = dict(
        omitted_relative_error_at_or_below=resolution_relative,
        error_unit="percent",raw_values="unchanged in numerical tables; no display floor")
    summaries = []
    refinement = []
    for method in MOMENT_METHODS+["qmom4"]:
        a = table(runs[method+"_dt1"][0]/"moments.csv")
        b = table(runs[method+"_dt2"][0]/"moments.csv") if method != "qmom4" else None
        summary = rows(runs[method+"_dt1"][0]/"summary.csv")[0]
        summary["comparison_role"] = "primary" if b is not None else "node_refinement"
        options = runs[method+"_dt1"][1]["settings"]["options"]
        summary["primary_nodes"] = options.get("primary_nodes",options.get("nodes",0))
        summary["secondary_nodes_per_kernel"] = options.get("secondary_nodes",0)
        summary["inversion_rtol"] = options.get("rtol",np.nan)
        summary["max_mu0_error_percent"] = float(a[a["order"]==0]["absolute_relative_percent"].max())
        summary["max_mu3_error_percent"] = float(a[a["order"]==3]["absolute_relative_percent"].max())
        summary["state_layout"] = ("weights and weighted abscissas" if method == "dqmom3"
                                   else "raw moments")
        summary["material_evaluation"] = ("assumed-lognormal reconstruction" if method == "lognormal"
                                          else "sum of weights times abscissas cubed" if method == "dqmom3"
                                          else "evolved raw third moment")
        for k in range(4):
            coarse = a[a["order"]==k]
            if b is not None:
                fine = b[b["order"]==k]
                assert np.array_equal(coarse["time_s"],fine["time_s"])
                difference = np.max(np.abs(coarse["moment"]-fine["moment"])/fine["exact"])
                summary[f"time_halving_mu{k}_percent"] = 100*difference
            else:
                summary[f"time_halving_mu{k}_percent"] = np.nan
            signed = coarse["signed_difference"]
            summary[f"signed_crossings_mu{k}"] = int(np.sum(signed[:-1]*signed[1:]<0))
        summaries.append(summary)
    three = table(runs["qmom3_dt1"][0]/"moments.csv")
    four = table(runs["qmom4_dt1"][0]/"moments.csv")
    for k in range(4):
        a,b = three[three["order"]==k],four[four["order"]==k]
        refinement.append(dict(order=k,maximum_relative_difference=float(np.max(np.abs(a["moment"]-b["moment"])/b["exact"])),
            qmom3_maximum_error_percent=float(a["absolute_relative_percent"].max()),qmom4_maximum_error_percent=float(b["absolute_relative_percent"].max())))
    for name,data in (("moment_summary.csv",summaries),("node_refinement.csv",refinement)):
        analysis_table(2,base,name,data)
    fig,axes = plt.subplots(2,2,figsize=(DOUBLE,5.3))
    fig.subplots_adjust(left=.1,right=.98,bottom=.16,top=.93,wspace=.32,hspace=.35)
    colors = ["#0072B2","#D55E00","#009E73","#CC79A7","#6A3D9A"]
    markers = ["o","x","^","s","D"]
    for j,(method,label) in enumerate(zip(MOMENT_METHODS,MOMENT_LABELS)):
        data = table(runs[method+"_dt1"][0]/"moments.csv")
        style = dict(color=colors[j],ls=["-","--","-.",":","--"][j],lw=1.1,
                     marker=markers[j],ms=3.5,mfc="white",markevery=(j*2,20),label=label)
        for k,ax in ((0,axes[0,0]),(1,axes[0,1]),(2,axes[1,0]),(3,axes[1,1])):
            d = data[data["order"]==k]
            if k == 0:
                ax.plot(d["time_s"],d["moment"]/d["moment"][0],**style)
                if j == 0:
                    ax.plot(d["time_s"],d["exact"]/d["exact"][0],color=".35",lw=1.7,label="Analytical")
            elif k == 3:
                ax.plot(d["time_s"],d["moment"]/.001,**style)
            else:
                values = d["absolute_relative_percent"]
                ax.semilogy(d["time_s"],np.where(values>100*resolution_relative,values,np.nan),**style)
    axes[0,0].set_ylabel(r"$\mu_0(t)/\mu_0(0)$")
    axes[0,1].set_ylabel(r"Absolute relative error in $\mu_1$ (\%)")
    axes[1,0].set_ylabel(r"Absolute relative error in $\mu_2$ (\%)")
    axes[1,1].axhline(1,color=".5",lw=.7)
    axes[1,1].set_ylabel(r"Material ratio $\mu_3/V_0$")
    for letter,ax in zip("abcd",axes.flat):
        panel(ax,letter);ax.set_xlabel(r"Time (s)")
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="lower center",ncol=3)
    export(fig,2,"moment_methods",info)

def precipitation():
    base,info,runs = load_case(3)
    runtime = []
    for name,(_,provenance) in runs.items():
        setting,summary = provenance["settings"],provenance["summary"]
        if summary.get("role") == "verification":
            continue
        runtime.append(dict(configuration=name,nx=setting["nx"],nz=setting["nz"],
            seconds=summary["seconds"],setup_seconds=summary["setup_seconds"],steps=summary["steps"],
            completed_time=summary["completed_time"],cpu=provenance["environment"]["cpu"],
            julia=provenance["environment"]["julia"],julia_threads=provenance["environment"]["julia_threads"],
            blas_threads=provenance["environment"]["blas_threads"]))
    analysis_table(3,base,"runtimes.csv",runtime)
    folder,p = runs["reference"]
    settings = p["settings"]
    if settings["times"] != [5.0,40.0,80.0,200.0]:
        raise ValueError("The field figure requires t=5,40,80,200 s in that order")
    x = table(folder/"size_edges.csv")["size_m"]
    z = table(folder/"axial_edges.csv")["z_m"]
    if not np.allclose(x[[0,-1]],[1e-6,900e-6],rtol=1e-12,atol=0):
        raise ValueError("The field figure must retain the full 1--900 micrometre size domain")
    nx,nz = settings["nx"],settings["nz"]
    fields,reference_numbers = [],[]
    for i in range(1,5):
        with gzip.open(folder/f"population_{i}.csv.gz","rt") as stream:
            numbers = np.loadtxt(stream,skiprows=1).reshape((nx,nz),order="F")
        reference_numbers.append(numbers)
        fields.append(numbers/np.diff(x)[:,None])
    # Independent observables complement the distribution-norm refinement
    # table. No normalization is imposed on the physical populations.
    reference_account = table(folder/"accounting.csv")
    color_minimum = 1e9
    absolute_scale = nx*settings["population_absolute_tolerance"]*settings["population_scaling"]
    checked_refinement = []
    for row in rows(base/"refinement.csv"):
        i = settings["times"].index(float(row["time_s"]))
        reference_number = reference_account["outlet_number_per_m3"][i]
        above = reference_number > absolute_scale
        row.update(reference_outlet_number_per_m3=reference_number,
            nominal_outlet_absolute_tolerance_scale_per_m3=absolute_scale,
            outlet_reference_above_nominal_absolute_scale=above,
            interpretable_outlet_L1=float(row["normalized_outlet_L1"]) if above else np.nan)
        checked_refinement.append(row)
    analysis_table(3,base,"checked_refinement.csv",checked_refinement)
    observables = []
    reference_centers = (x[:-1]+x[1:])/2
    for name,(run_path,record) in runs.items():
        if record["summary"].get("role") == "verification":
            continue
        edges = table(run_path/"size_edges.csv")["size_m"]
        axial_edges = table(run_path/"axial_edges.csv")["z_m"]
        centers = (edges[:-1]+edges[1:])/2
        account = table(run_path/"accounting.csv")
        for i,t in enumerate(record["settings"]["times"]):
            with gzip.open(run_path/f"population_{i+1}.csv.gz","rt") as stream:
                numbers = np.loadtxt(stream,skiprows=1).reshape(
                    (record["settings"]["nx"],record["settings"]["nz"]),order="F")
            material = centers[:,None]**3*numbers*np.diff(axial_edges)[None,:]
            reference = reference_numbers[i]
            denominator = float(np.sum(reference_centers**3*reference[:,-1]))
            a,b = account[i],reference_account[i]
            observables.append(dict(configuration=name,time_s=t,
                nx=record["settings"]["nx"],nz=record["settings"]["nz"],
                relative_inventory_number_difference=relative_difference(a["number_per_m2"],b["number_per_m2"]),
                relative_inventory_M3_difference=relative_difference(a["integrated_M3_m"],b["integrated_M3_m"]),
                relative_outlet_number_difference=relative_difference(a["outlet_number_per_m3"],b["outlet_number_per_m3"]),
                relative_outlet_M3_difference=relative_difference(float(np.sum(centers**3*numbers[:,-1])),denominator),
                outlet_solute_difference_kg_m3=a["outlet_solute_kg_m3"]-b["outlet_solute_kg_m3"],
                upper_bin_number_fraction=a["upper_bin_fraction"],
                upper_bin_material_fraction=float(material[-1].sum()/material.sum()),
                negative_number_per_m2=a["negative_number_per_m2"],
                stored_zero_cells=int(np.count_nonzero(numbers==0)),
                negative_cells=int(np.count_nonzero(numbers<0)),
                positive_cells_below_color_minimum=int(np.count_nonzero((numbers>0)&(numbers/np.diff(edges)[:,None]<color_minimum))),
                material_rate_defect_kg_m2_s=a["material_rate_defect_kg_m2_s"]))
    analysis_table(3,base,"refinement_observables.csv",observables)
    maximum = max(np.max(v) for v in fields)
    norm = LogNorm(vmin=color_minimum,vmax=10**np.ceil(np.log10(maximum)))
    fig,axes = plt.subplots(2,2,figsize=(DOUBLE,4.8))
    fig.subplots_adjust(left=.09,right=.84,bottom=.11,top=.93,wspace=.3,hspace=.3)
    for letter,ax,t,data in zip("abcd",axes.flat,settings["times"],fields):
        mesh = ax.pcolormesh(z,x*1e6,np.ma.masked_less_equal(data,0),norm=norm,cmap="viridis",shading="flat",rasterized=True)
        ax.set(yscale="log",ylim=(1,900),xlabel=r"Axial position $z$ (m)",ylabel=r"Size $x$ ($\mu$m)")
        panel(ax,letter)
        ax.text(.97,.96,rf"$t={t:g}\,\mathrm{{s}}$",transform=ax.transAxes,ha="right",va="top",color="black",
                bbox=dict(facecolor="white",edgecolor="none",alpha=.8,pad=1))
    coloraxis = fig.add_axes([.88,.16,.025,.74])
    fig.colorbar(mesh,cax=coloraxis,label=r"Number density $n$ (m$^{-4}$)",extend="min")
    export(fig,3,"precipitation_fields",info)
    fig,axes = plt.subplots(1,2,figsize=(DOUBLE,3.55))
    fig.subplots_adjust(left=.09,right=.98,bottom=.29,top=.93,wspace=.3)
    colors = {1:"#8C564B",2:"#9467BD",3:"#D62728",13:"#2CA02C",50:"#FF7F0E",200:"#1F77B4"}
    for i,nz_ in enumerate(info["configuration"]["outlet_nz"]):
        data = table(runs[f"outlet_Nz{nz_}"][0]/"outlet.csv")
        edges = np.r_[data["left_m"],data["right_m"][-1]]*1e6
        for ax,quantity in zip(axes,(data["number_density_m4"]/1e15,data["volume_density_per_m"]*1e-6)):
            ax.stairs(quantity,edges,color=colors.get(nz_,"#17BECF"),lw=1.0,label=rf"Package $N_z={nz_}$")
    data = table(folder/"outlet.csv")
    edges = np.r_[data["left_m"],data["right_m"][-1]]*1e6
    for ax,quantity in zip(axes,(data["number_density_m4"]/1e15,data["volume_density_per_m"]*1e-6)):
        ax.stairs(quantity,edges,color="k",lw=1.2,ls="--",label=rf"Package reference ({nx}$\times${nz})")
    if MODE == "production":
        lit = table(PAPER/"literature"/"cadet_reference.csv")
        literature_edges = np.r_[lit["left_m"],lit["right_m"][-1]]
        centers = (lit["left_m"]+lit["right_m"])*.5e6
        for ax,quantity in zip(axes,(lit["number_density_m4"]/1e15,lit["volume_density_per_m"]*1e-6)):
            ax.plot(centers,quantity,ls="none",marker="s",markevery=10,ms=3,mfc="none",color=".4",label="Public CADET verification")
        digitized = table(PAPER/"literature"/"zhang_figure9c_digitized.csv")
        axes[0].errorbar(digitized["size_m"]*1e6,digitized["number_density_m4"]/1e15,
            xerr=digitized["size_uncertainty_m"]*1e6,yerr=digitized["density_uncertainty_m4"]/1e15,
            ls="none",marker="o",ms=3,mfc="white",color="k",elinewidth=.5,label="Zhang Fig. 9(c), digitized")
        comparisons = []
        for name in ("outlet_Nz200","reference"):
            values = table(runs[name][0]/"outlet.csv")
            edges_ = np.r_[values["left_m"],values["right_m"][-1]]
            comparisons.append(dict(configuration=name,
                reference="Public CADET size Upwind / axial WENO23; not published WENO35",
                number_normalized_L1=density_l1(edges_,values["number_density_m4"],literature_edges,lit["number_density_m4"]),
                normalized_volume_L1=density_l1(edges_,values["volume_density_per_m"],literature_edges,lit["volume_density_per_m"])))
        analysis_table(3,base,"public_cadet_comparison.csv",comparisons)
        extracted_comparison = []
        for name in ("outlet_Nz200","reference"):
            values = table(runs[name][0]/"outlet.csv")
            edges_ = np.r_[values["left_m"],values["right_m"][-1]]
            for point in digitized:
                index_ = np.clip(np.searchsorted(edges_,point["size_m"],side="right")-1,0,len(values)-1)
                cells = ((edges_[:-1]<=point["size_m"]+point["size_uncertainty_m"]) &
                         (edges_[1:]>=point["size_m"]-point["size_uncertainty_m"]))
                low,high = np.min(values["number_density_m4"][cells]),np.max(values["number_density_m4"][cells])
                extracted_comparison.append(dict(configuration=name,size_m=point["size_m"],literature_density_m4=point["number_density_m4"],
                    density_uncertainty_m4=point["density_uncertainty_m4"],size_uncertainty_m=point["size_uncertainty_m"],
                    package_PWC_density_m4=values["number_density_m4"][index_],
                    relative_difference=relative_difference(values["number_density_m4"][index_],point["number_density_m4"]),
                    package_range_over_x_uncertainty_min=low,package_range_over_x_uncertainty_max=high,
                    extraction_intervals_overlap=bool(high>=point["number_density_m4"]-point["density_uncertainty_m4"] and
                        low<=point["number_density_m4"]+point["density_uncertainty_m4"])))
        analysis_table(3,base,"digitized_point_comparison.csv",extracted_comparison)
    axes[0].set_ylabel(r"$n/(10^{15}\,\mathrm{m}^{-4})$")
    axes[1].set_ylabel(r"Normalized volume density ($\mu$m$^{-1}$)")
    for letter,ax in zip("ab",axes):
        panel(ax,letter);ax.set(xscale="log",xlim=(1,900),xlabel=r"Particle size $x$ ($\mu$m)")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=3)
    export(fig,3,"precipitation_outlet",info)

def main():
    global MODE,PREVIEW
    parser = argparse.ArgumentParser()
    parser.add_argument("--case",choices=["all","1","2","3","4"],default="all")
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--preview",action="store_true")
    parser.add_argument("--data-only",action="store_true",help="Case 4 observations only; no fit needed")
    parser.add_argument("--figures-only",action="store_true",help="Case 4: regenerate figures without rewriting result tables")
    parser.add_argument("--laplace-comparison",action="store_true",help="Case 4: plot separate Laplace/MCMC analysis; read-only production inputs")
    parser.add_argument("--gauss-newton-comparison",action="store_true",help="Case 4: sensitivity-based covariance/MCMC comparison")
    args = parser.parse_args()
    MODE = "smoke" if args.smoke else "production"
    PREVIEW = args.preview
    if PREVIEW and MODE != "smoke":
        parser.error("fallback previews cannot be labelled production figures")
    if args.figures_only and args.case != "4":
        parser.error("--figures-only requires --case 4")
    if args.laplace_comparison or args.gauss_newton_comparison:
        if args.laplace_comparison and args.gauss_newton_comparison:
            parser.error("Select one comparison per invocation")
        if args.case != "4" or args.smoke or args.preview or args.data_only:
            parser.error("--laplace-comparison requires production --case 4")
        from plot_protein_laplace import main as compare
        compare(['--approximation','gauss_newton' if args.gauss_newton_comparison else 'laplace'])
        return
    configure(PREVIEW,require_dvipng=args.case != "4")
    for i,function in enumerate((growth,moments,precipitation),1):
        if args.case in ("all",str(i)):
            function()
    if args.case in ("all","4"):
        if args.smoke or args.preview:
            print("Protein smoke mode verifies the forward solve; no calibrated paper figure is produced.")
        else:
            from plot_protein_isothermal import plot
            plot(data_only=args.data_only,figures_only=args.figures_only)

if __name__ == "__main__":
    main()
