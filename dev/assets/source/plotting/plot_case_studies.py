"""Render case studies and kinetic uncertainty from exported Julia CSV files.

python docs/benchmarks/plot_case_studies.py CASE [--data-dir PATH] [--usetex]
Requires NumPy and Matplotlib. No simulation is run by this script.
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_style import configure

ASSETS=Path(__file__).resolve().parents[1]/"src"/"assets"
configure()

def save_figure(fig,folder,name,vector=True):
    folder.mkdir(parents=True,exist_ok=True)
    for axis in fig.axes:
        axis.grid(False,which="both")
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
           ("weno5","WENO5 + SSPRK3"),("mapped_shift","Inguva and Braatz (2023)"))
    fig,axes=plt.subplots(2,2,figsize=(10,8),layout="constrained")
    for ax,(name,title) in zip(axes.flat,names):
        x,y,z=field(folder/f"{name}_100.csv")
        im=ax.pcolormesh(x,y,z,cmap="viridis",shading="auto",vmin=0,vmax=40,rasterized=True)
        ax.set(xlabel="$a_1$",ylabel="$a_2$",title=title,aspect="equal",xlim=(0,2),ylim=(0,2))
        fig.colorbar(im,ax=ax,label=r"$f(a_1,a_2,t)$")
    fig.suptitle(r"Coordinate-dependent growth, $t=1$")
    save_figure(fig,folder,"solutions")
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout="constrained")
    for ax,(name,title) in zip(axes,(("upwind","Upwind"),("weno5","WENO5"),("mapped_shift","Inguva and Braatz (2023)"))):
        x,y,z=field(folder/f"{name}_100.csv")
        ref="mapped_reference" if name=="mapped_shift" else "reference"
        _,_,truth=field(folder/f"{ref}_100.csv");error=z-truth;bound=max(np.max(abs(error)),1e-15)
        im=ax.pcolormesh(x,y,error,cmap="RdBu_r",shading="auto",vmin=-bound,vmax=bound)
        ax.set(xlabel="$a_1$",ylabel="$a_2$",title=title,aspect="equal")
        fig.colorbar(im,ax=ax,label=r"$f_{\mathrm{num}}-f_{\mathrm{exact}}$")
    save_figure(fig,folder,"errors",vector=False)
    rows=np.genfromtxt(folder/"errors.csv",delimiter=",",names=True,dtype=None,encoding="utf-8")
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout="constrained")
    for name,label in (("upwind","Upwind"),("weno5","WENO5"),("mapped_shift","Inguva and Braatz (2023)")):
        data=rows[rows["method"]==name]; bins=data["nx"]*data["ny"]
        axes[0].loglog(bins,data["RMS_area"],"o-",label=label)
        axes[1].loglog(bins,np.maximum(data["negative_number"],1e-20),"o-",label=label)
    axes[0].set(xlabel=r"$N_{\mathrm{bins}}$",ylabel=r"$\mathrm{RMS}_{A}(f_{\mathrm{num}}-f_{\mathrm{exact}})$")
    axes[1].set(xlabel=r"$N_{\mathrm{bins}}$",ylabel=r"$-\int\min(f,0)\,\mathrm{d}a_1\,\mathrm{d}a_2$")
    axes[0].legend(fontsize=8)
    axes[1].text(.04,.08,r"Zero shown at $10^{-20}$",transform=axes[1].transAxes,fontsize=8)
    save_figure(fig,folder,"convergence")

def cooling(folder,output=None):
    output=folder if output is None else output
    # Provenance is part of the exported result, rather than a figure label
    # hard-coded to a previously substituted initial distribution.
    provenance=(folder/"inputs.txt").read_text()
    data=np.genfromtxt(folder/"moments.csv",delimiter=",",names=True,dtype=None,encoding="utf-8")
    ref=load(folder/"reference.csv")
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout="constrained")
    for method in np.unique(data["method"]):
        label="MUSCL" if method=="muscl" else "Upwind"
        subset=data[data["method"]==method];bins=max(subset["bins"]);subset=subset[subset["bins"]==bins]
        axes[0,0].plot(subset["time"],subset["concentration"],label=f"{label}, {bins} bins")
        for k in (0,3):
            error=abs(subset[f"mu{k}"]-ref[f"mu{k}"])/np.maximum(abs(ref[f"mu{k}"]),1e-30)
            axes[0,1].semilogy(subset["time"],np.maximum(error,1e-16),label=rf"{label}, $k={k}$")
        axes[1,1].plot(subset["time"],subset["material_drift"],label=label)
    axes[0,0].plot(ref["time"],ref["concentration"],"k--",label="Independent moment ODE")
    for path in sorted(folder.glob("*_psd.csv")):
        d=load(path);label="MUSCL" if path.stem.startswith("muscl") else "Upwind"
        axes[1,0].plot(d["length"],d["density"],label=f"{label}, {len(d)} bins")
    axes[0,0].set(xlabel=r"$t\;[\mathrm{s}]$",ylabel=r"$c\;[\mathrm{g}/\mathrm{g\ water}]$")
    axes[0,1].set(xlabel=r"$t\;[\mathrm{s}]$",ylabel=r"$|\mu_k-\mu_{k,\mathrm{ref}}|/|\mu_{k,\mathrm{ref}}|$")
    axes[1,0].set(xlabel=r"$L\;[\mu\mathrm{m}]$",ylabel=r"$f(L,t_{\mathrm{f}})$",title="Final distribution")
    axes[1,1].set(xlabel=r"$t\;[\mathrm{s}]$",ylabel=r"$\Delta(c+\rho_c\mu_3)\;[\mathrm{g}/\mathrm{g\ water}]$")
    for ax in axes.flat:ax.legend(fontsize=7)
    fig.suptitle("Batch cooling crystallization (input provenance in inputs.txt)")
    if "demonstration" in provenance.lower() or "synthetic" in provenance.lower():
        fig.suptitle("Batch cooling crystallization: demonstration initial data")
    save_figure(fig,output,"coupled_solution")

def lnp(folder):
    history=load(folder/"history.csv");psd=load(folder/"psd.csv")
    if "intensity_peak_scaled" not in psd.dtype.names:
        raise ValueError("Re-export with the current Figure 4 driver; peak scaling must be explicit.")
    fig,axes=plt.subplots(1,5,figsize=(13,3.4),layout="constrained")
    for ax,minutes in zip(axes,(1,40,100,170,4320)):
        target=60.0*minutes;mask=np.isclose(psd["time"],target,rtol=1e-10,atol=1e-9)
        if not np.any(mask):raise ValueError(f"No snapshot at {minutes} min. Use figure4_saveat().")
        d=psd[mask];ax.plot(d["diameter_m"]*1e9,d["intensity_peak_scaled"],"k-")
        ax.set(xlabel=r"$L\;[\mathrm{nm}]$",title=rf"$t={minutes}\;\mathrm{{min}}$",xlim=(0,500),ylim=(0,1.05))
    axes[0].set_ylabel("Peak-scaled intensity-weighted density")
    fig.suptitle("Lipid nanoparticle formation: calculated intensity distributions")
    save_figure(fig,folder,"figure4a_model")
    fig,ax=plt.subplots(figsize=(7,4),layout="constrained")
    valid=(history["time"]>0)&np.isfinite(history["intensity_mean_m"])
    ax.semilogx(history["time"][valid]/60,history["intensity_mean_m"][valid]*1e9,"k-")
    ax.set(xlabel=r"$t\;[\mathrm{min}]$",ylabel=r"$\overline{L}\;[\mathrm{nm}]$",title="Lipid nanoparticle formation and incubation")
    save_figure(fig,folder,"figure4b_model")


def illustrations():
    diagnostics=Path(__file__).resolve().parent/"diagnostics"
    cooling(diagnostics/"cooling_benchmark",ASSETS.parents[2]/".cache/legacy_docs_assets/illustrations/cooling")
    folder=diagnostics/"lnp_demo";history=load(folder/"history.csv");psd=load(folder/"psd.csv")
    fig,axes=plt.subplots(2,2,figsize=(9,6.5),layout="constrained")
    axes[0,0].plot(history["time"]*1e3,history["number"])
    axes[0,0].set(xlabel=r"$t\;[\mathrm{ms}]$",ylabel=r"$M_0\;[\mathrm{m}^{-3}]$")
    axes[0,1].plot(history["time"]*1e3,history["free_lipid"])
    axes[0,1].set(xlabel=r"$t\;[\mathrm{ms}]$",ylabel=r"$c\;[\mathrm{kg}\,\mathrm{m}^{-3}]$")
    for t in (0.005,0.01,0.02):
        data=psd[np.isclose(psd["time"],t,atol=1e-12,rtol=0)]
        axes[1,0].semilogx(data["diameter_m"]*1e9,data["intensity_density_per_m"],label=rf"${t*1e3:g}\;\mathrm{{ms}}$")
    axes[1,0].set(xlabel=r"$L\;[\mathrm{nm}]$",ylabel=r"$I(L)/\!\int I\,\mathrm{d}L\;[\mathrm{m}^{-1}]$")
    axes[1,0].legend()
    material=history["material_element_lipid"]
    axes[1,1].plot(history["time"]*1e3,(material-material[0])/material[0])
    axes[1,1].set(xlabel=r"$t\;[\mathrm{ms}]$",ylabel=r"$[m_{\mathrm{lipid}}(t)-m_{\mathrm{lipid}}(0)]/m_{\mathrm{lipid}}(0)$")
    fig.suptitle("Nanoparticle formation: synthetic material and mixing inputs")
    save_figure(fig,ASSETS.parents[2]/".cache/legacy_docs_assets/illustrations/lnp","demonstration")


def temperature_moments(folder):
    data=np.genfromtxt(folder/"history.csv",delimiter=",",names=True,dtype=None,encoding="utf-8")
    ref=data[data["method"]=="moments"]
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout="constrained")
    axes[0,0].plot(ref["time"],ref["supersaturation"],label="Native moments")
    temp=axes[0,0].twinx();temp.plot(ref["time"],ref["temperature_K"]-273.15,"--",color="0.5")
    temp.set_ylabel(r"$T\;[{}^\circ\mathrm{C}]$")
    axes[0,0].set(xlabel=r"$t\;[\mathrm{min}]$",ylabel=r"$S=c/c_s(T)$")
    axes[0,1].plot(ref["time"],ref["L30"],label=r"$\overline{L}_{30}=(\mu_3/\mu_0)^{1/3}$")
    axes[0,1].plot(ref["time"],ref["L43"],label=r"$\overline{L}_{43}=\mu_4/\mu_3$")
    axes[0,1].set(xlabel=r"$t\;[\mathrm{min}]$",ylabel=r"$\overline{L}\;[\mu\mathrm{m}]$")
    axes[1,0].plot(ref["time"],ref["mu0"],label="Native moments")
    axes[1,0].set(xlabel=r"$t\;[\mathrm{min}]$",ylabel=r"$\mu_0\;[\mathrm{kg}_{\mathrm{solvent}}^{-1}]$")
    for bins in sorted(set(data[data["method"]=="muscl"]["bins"])):
        d=data[(data["method"]=="muscl")&(data["bins"]==bins)]
        axes[0,0].plot(d["time"],d["supersaturation"],":",label=f"MUSCL, {bins} bins")
        axes[1,0].plot(d["time"],d["mu0"],":",label=f"MUSCL, {bins} bins")
        error=abs(d["mu3"]-ref["mu3"])/max(abs(ref["mu3"]))
        axes[1,1].semilogy(d["time"],np.maximum(error,1e-16),label=f"{bins} bins")
    axes[1,1].set(xlabel=r"$t\;[\mathrm{min}]$",ylabel=r"$|\mu_{3,\mathrm{FV}}-\mu_{3,\mathrm{MOM}}|/\max_t|\mu_{3,\mathrm{MOM}}|$")
    for axis in axes.flat:axis.legend(fontsize=8)
    fig.suptitle("Unseeded crystallization: reconstructed monotonic cooling input")
    save_figure(fig,folder,"moment_comparison")


def moments(folder):
    data=load(folder/'aggregation_breakage.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for k in range(6):
        q=data[(data['nodes']==3)&(data['order']==k)]
        initial=q['exact'][0]
        axes[0].plot(q['time'],q['exact']/initial,color=f'C{k}',lw=1.1)
        axes[0].plot(q['time'][::5],q['moment'][::5]/initial,'o',color=f'C{k}',ms=3,mfc='none',label=rf'$\mu_{k}$')
        if k!=3:axes[1].plot(q['time'],q['relative_error'],color=f'C{k}',label=rf'$\mu_{k}$')
    axes[0].set(xlabel=r'$t\ (\mathrm{s})$',ylabel=r'$\mu_k(t)/\mu_k(0)$',title='Lines: analytical; markers: three-node QMOM')
    axes[1].set(xlabel=r'$t\ (\mathrm{s})$',ylabel='Relative moment error',title='Three-node closure error')
    for ax in axes:ax.legend(fontsize=8)
    save_figure(fig,folder,'aggregation_breakage')

    data=np.genfromtxt(folder/'pulse.csv',delimiter=',',names=True,dtype=None,encoding='utf-8');ref=load(folder/'sectional_reference.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax,process,title in zip(axes,['growth_nucleation','all'],['Growth and nucleation','Growth, nucleation and collisions']):
        for k in range(5):
            q=data[(data['processes']==process)&(data['nodes']==3)&(data['order']==k)]
            initial=q['moment'][0]
            if process=='growth_nucleation':ax.semilogy(q['time'],q['reference']/initial,color=f'C{k}',lw=1.1)
            else:
                rr=ref[(ref['bins']==max(ref['bins']))&(ref['order']==k)]
                ax.semilogy(rr['time'],rr['moment']/initial,color=f'C{k}',lw=1.1)
            ax.semilogy(q['time'][::3],q['moment'][::3]/initial,'o',color=f'C{k}',ms=3,mfc='none',label=rf'$\mu_{k}$')
        ax.set(xlabel=r'$t\ (\mathrm{s})$',ylabel=r'$\mu_k(t)/\mu_k(0)$',title=title);ax.legend(fontsize=8)
    save_figure(fig,folder,'nucleation_pulse')

    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for n in np.unique(ref['bins']):
        for k in (0,3,4):
            q=data[(data['processes']=='all')&(data['nodes']==3)&(data['order']==k)]
            r=ref[(ref['bins']==n)&(ref['order']==k)]
            relative=abs(r['moment']/q['moment']-1)
            axes[0].semilogy(q['time'][1:],np.maximum(relative[1:],1e-16),color=f'C{k}',ls='-' if n==max(ref['bins']) else '--',label=rf'$\mu_{k}$, {int(n)} bins')
    for n in np.unique(data['nodes']):
        q=data[(data['processes']=='growth_nucleation')&(data['nodes']==n)&(data['order']==4)]
        axes[1].semilogy(q['time'][1:],np.maximum(q['relative_difference'][1:],1e-16),label=rf'Growth/source, $\mu_4$, {int(n)} nodes')
    axes[0].set(xlabel=r'$t\ (\mathrm{s})$',ylabel='Relative difference from three-node QMOM',title='Sectional-reference refinement')
    axes[1].set(xlabel=r'$t\ (\mathrm{s})$',ylabel='Relative analytical error',title='Growth/source integration check')
    for ax in axes:ax.legend(fontsize=7)
    save_figure(fig,folder,'pulse_verification')


def estimation(folder):
    data=load(folder/"observations.csv");curve=load(folder/"curve.csv")
    region=load(folder/"region.csv");estimate=load(folder/"estimates.csv");profile=load(folder/"profile.csv")
    fig,axes=plt.subplots(2,2,figsize=(9,7),layout="constrained")
    axes[0,0].errorbar(data["time"],data["observed"],yerr=0.03,fmt="o",ms=3,label=r"Synthetic data $\pm\sigma$")
    axes[0,0].plot(curve["time"],curve["fitted"],label="Fitted model")
    axes[0,0].plot(curve["time"],curve["analytical_truth"],"--",color="0.4",label="Generating model")
    axes[0,0].set(xlabel=r"$t$",ylabel=r"$M_0(t)$");axes[0,0].legend(fontsize=8)
    axes[0,1].axhline(0,color="0.6",lw=.8)
    axes[0,1].plot(data["time"],data["whitened_residual"],"o",ms=4)
    axes[0,1].set(xlabel=r"$t$",ylabel=r"$r_j=(\widehat y_j-y_j)/\sigma_j$")
    axes[1,0].plot(region["J"],region["k"],label=r"Mapped $95\%$ joint region")
    axes[1,0].plot(estimate["J"],estimate["k"],"o",label="Estimate")
    axes[1,0].plot(estimate["true_J"],estimate["true_k"],"x",ms=7,label="Generating parameters")
    axes[1,0].set(xlabel=r"$J$",ylabel=r"$k$");axes[1,0].legend(fontsize=8)
    valid=profile["converged"]==1
    axes[1,1].plot(np.exp(profile["logJ"][valid]),profile["twice_delta_nll"][valid])
    axes[1,1].axhline(3.841458820694124,color="0.4",ls="--",label=r"$\chi^2_{1,0.95}$")
    axes[1,1].set(xlabel=r"$J$",ylabel=r"$2[\ell_{\mathrm{prof}}(J)-\ell(\widehat{\boldsymbol{z}})]$")
    axes[1,1].legend(fontsize=8)
    fig.suptitle("Kinetic estimation with known measurement errors")
    save_figure(fig,folder,"fit_and_uncertainty")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case",choices=("transport","cooling","lnp","illustrations","moments","temperature_moments","estimation"))
    parser.add_argument("--usetex",action="store_true",help="Render with an installed LaTeX distribution instead of Matplotlib MathText.")
    parser.add_argument("--data-dir",type=Path)
    args=parser.parse_args()
    configure(usetex=args.usetex)
    if args.case=="illustrations":
        illustrations()
    else:
        folder=args.data_dir or ASSETS/{"transport":"inguva_case2","cooling":"gunawan_example2","lnp":"shin_case","moments":"qamar_moments","temperature_moments":"moment_crystallization","estimation":"parameter_estimation"}[args.case]
        {"transport":transport,"cooling":cooling,"lnp":lnp,"moments":moments,"temperature_moments":temperature_moments,"estimation":estimation}[args.case](folder)
