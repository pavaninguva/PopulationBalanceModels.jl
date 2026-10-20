"""Plot the three physical case studies from exported Julia CSV files.

python docs/benchmarks/plot_case_studies.py transport|cooling|lnp [--data-dir PATH]
Requires NumPy and Matplotlib. No simulation is run by this script.
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSETS=Path(__file__).resolve().parents[1]/"src"/"assets"
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
plt.rcParams["svg.hashsalt"]="PopulationBalanceModels-case-studies"

def save_figure(fig,folder,name,vector=True):
    fig.savefig(folder/f"{name}.png",dpi=180)
    if vector:
        path=folder/f"{name}.svg"
        fig.savefig(path,dpi=180,metadata={"Date":None})
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    plt.close(fig)

def load(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run the corresponding Julia case first; no substitute data are plotted.")
    return np.genfromtxt(path,delimiter=",",names=True)

def field(path):
    d=load(path);x=np.unique(d["a1"]);y=np.unique(d["a2"])
    z=d["density"].reshape(len(y),len(x))
    if "lower_a1" in d.dtype.names:
        # Nonuniform physical cells must use their actual edges, rather than
        # edges inferred from neighbouring centres on the transformed mesh.
        x=np.unique(np.concatenate((d["lower_a1"],d["upper_a1"])))
        y=np.unique(np.concatenate((d["lower_a2"],d["upper_a2"])))
    return x,y,z

def transport(folder):
    names=(("reference","Analytical bin averages"),("upwind","Upwind + SSPRK3"),
           ("weno5","WENO5 + SSPRK3"),("mapped_shift","Exact shifts on transformed mesh"))
    fig,axes=plt.subplots(2,2,figsize=(10,8),layout="constrained")
    for ax,(name,title) in zip(axes.flat,names):
        x,y,z=field(folder/f"{name}_100.csv")
        im=ax.pcolormesh(x,y,z,cmap="viridis",shading="auto",vmin=0,vmax=40,rasterized=True)
        ax.set(xlabel="$a_1$",ylabel="$a_2$",title=title,aspect="equal",xlim=(0,2),ylim=(0,2))
        fig.colorbar(im,ax=ax,label="Number density")
    fig.suptitle("Growth in two particle dimensions: coordinate-dependent velocities, t = 1")
    save_figure(fig,folder,"solutions")
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout="constrained")
    for ax,name in zip(axes,("upwind","weno5","mapped_shift")):
        x,y,z=field(folder/f"{name}_100.csv")
        ref="mapped_reference" if name=="mapped_shift" else "reference"
        _,_,truth=field(folder/f"{ref}_100.csv");error=z-truth;bound=max(np.max(abs(error)),1e-15)
        im=ax.pcolormesh(x,y,error,cmap="RdBu_r",shading="auto",vmin=-bound,vmax=bound)
        ax.set(xlabel="$a_1$",ylabel="$a_2$",title=name,aspect="equal")
        fig.colorbar(im,ax=ax,label="Numerical − analytical density")
    save_figure(fig,folder,"errors",vector=False)

def cooling(folder):
    # Provenance is part of the exported result, rather than a figure label
    # hard-coded to a previously substituted initial distribution.
    provenance=(folder/"inputs.txt").read_text()
    data=np.genfromtxt(folder/"moments.csv",delimiter=",",names=True,dtype=None,encoding="utf-8")
    ref=load(folder/"reference.csv")
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout="constrained")
    for method in np.unique(data["method"]):
        subset=data[data["method"]==method];bins=max(subset["bins"]);subset=subset[subset["bins"]==bins]
        axes[0,0].plot(subset["time"],subset["concentration"],label=f"{method}, {bins} bins")
        for k in (0,3):
            error=abs(subset[f"mu{k}"]-ref[f"mu{k}"])/np.maximum(abs(ref[f"mu{k}"]),1e-30)
            axes[0,1].semilogy(subset["time"],np.maximum(error,1e-16),label=f"{method}, moment {k}")
        axes[1,1].plot(subset["time"],subset["material_drift"],label=method)
    axes[0,0].plot(ref["time"],ref["concentration"],"k--",label="Independent moment ODE")
    for path in sorted(folder.glob("*_psd.csv")):
        d=load(path);axes[1,0].plot(d["length"],d["density"],label=path.stem.replace("_psd",""))
    axes[0,0].set(xlabel="Time [s]",ylabel="Concentration [g/g water]")
    axes[0,1].set(xlabel="Time [s]",ylabel="Relative moment error")
    axes[1,0].set(xlabel="Crystal length [µm]",ylabel="Number density",title="Final distribution")
    axes[1,1].set(xlabel="Time [s]",ylabel="Material drift [g/g water]")
    for ax in axes.flat:ax.legend(fontsize=7)
    fig.suptitle("Batch cooling crystallization (input provenance in inputs.txt)")
    if "demonstration" in provenance.lower() or "synthetic" in provenance.lower():
        fig.suptitle("Batch cooling crystallization: demonstration initial data")
    save_figure(fig,folder,"coupled_solution")

def lnp(folder):
    history=load(folder/"history.csv");psd=load(folder/"psd.csv")
    if "intensity_peak_scaled" not in psd.dtype.names:
        raise ValueError("Re-export with the current Figure 4 driver; peak scaling must be explicit.")
    fig,axes=plt.subplots(1,5,figsize=(13,3.4),layout="constrained")
    for ax,minutes in zip(axes,(1,40,100,170,4320)):
        target=60.0*minutes;mask=np.isclose(psd["time"],target,rtol=1e-10,atol=1e-9)
        if not np.any(mask):raise ValueError(f"No snapshot at {minutes} min. Use figure4_saveat().")
        d=psd[mask];ax.plot(d["diameter_m"]*1e9,d["intensity_peak_scaled"],"k-")
        ax.set(xlabel="Diameter [nm]",title=f"t = {minutes} min",xlim=(0,500),ylim=(0,1.05))
    axes[0].set_ylabel("Peak-scaled intensity-weighted density")
    fig.suptitle("Lipid nanoparticle formation: calculated intensity distributions")
    save_figure(fig,folder,"figure4a_model")
    fig,ax=plt.subplots(figsize=(7,4),layout="constrained")
    valid=(history["time"]>0)&np.isfinite(history["intensity_mean_m"])
    ax.semilogx(history["time"][valid]/60,history["intensity_mean_m"][valid]*1e9,"k-")
    ax.set(xlabel="Time [min]",ylabel="Intensity-weighted mean diameter [nm]",title="Lipid nanoparticle formation and incubation")
    save_figure(fig,folder,"figure4b_model")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case",choices=("transport","cooling","lnp"))
    parser.add_argument("--data-dir",type=Path)
    args=parser.parse_args()
    folder=args.data_dir or ASSETS/{"transport":"inguva_case2","cooling":"gunawan_example2","lnp":"shin_case"}[args.case]
    {"transport":transport,"cooling":cooling,"lnp":lnp}[args.case](folder)
