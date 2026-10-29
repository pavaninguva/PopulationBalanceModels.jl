"""Regenerate this case's paper figures from its own numerical exports."""
from pathlib import Path
import sys
import gzip
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import NullLocator, LogLocator, LogFormatterSciNotation, MaxNLocator, ScalarFormatter
PAPER=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PAPER))
from _support import figures
from _support.figures import (load_case,rows,table,export,analysis_table,
    density_l1,relative_difference)
from plotting_style import panel, DOUBLE

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
    runtime.sort(key=lambda r:(0 if r["configuration"].startswith("size_") else
        1 if r["configuration"].startswith("axial_") else
        2 if r["configuration"]=="reference" else 3,
        r["nx"] if r["configuration"].startswith("size_") else r["nz"]))
    analysis_table(3,base,"runtimes.csv",runtime)
    folder,p = runs["reference"]
    settings = p["settings"]
    if settings["times"] != [5.0,40.0,80.0,200.0]:
        raise ValueError("The field figure requires t=5,40,80,200 s in that order")
    x = table(folder/"size_edges.csv")["size_m"]
    z = table(folder/"axial_edges.csv")["z_m"]
    if not np.allclose(x[[0,-1]],[1e-6,900e-6],rtol=1e-12,atol=0):
        raise ValueError("The saved reference must span the full 1--900 micrometre size domain")
    nx,nz = settings["nx"],settings["nz"]
    info["display_conventions"] = dict(
        fields="actual logarithmic size/uniform axial edges; linear 1--100 micrometre size-axis zoom; physical bin density; no interpolation",
        field_size_axis="linear",
        field_display_size_um=[1,100],
        simulated_size_um=[1,900],
        errors="normalized PWC L1 on common partitions; one common numerical reference",
        outlet="stairs of bin averages; center-cubed volume weights; digitized number-density points only",
        literature_ordinate="published ordinates unchanged, interpreted as SI m^-4; the printed per-micrometre label would violate the feed material scale",
        color="shared linear scale from zero to the maximum over all four full reference fields; magma; negative tails use the lower endpoint colour",
        field_color_scale="linear",
        field_color_map="magma",
        historical_tail_count_threshold_m4=1e9)
    fields,reference_numbers = [],[]
    for i in range(1,5):
        with gzip.open(folder/f"population_{i}.csv.gz","rt") as stream:
            numbers = np.loadtxt(stream,skiprows=1).reshape((nx,nz),order="F")
        reference_numbers.append(numbers)
        fields.append(numbers/np.diff(x)[:,None])
    # Independent observables complement the distribution-norm refinement
    # table. No normalization is imposed on the physical populations.
    reference_account = table(folder/"accounting.csv")
    # Preserve the established tail-count column and its threshold in existing
    # exports. It is no longer the lower bound of the linear colour scale.
    historical_tail_count_threshold = 1e9
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
                positive_cells_below_color_minimum=int(np.count_nonzero((numbers>0)&(numbers/np.diff(edges)[:,None]<historical_tail_count_threshold))),
                material_rate_defect_kg_m2_s=a["material_rate_defect_kg_m2_s"]))
    analysis_table(3,base,"refinement_observables.csv",observables)
    maximum = max(np.max(v) for v in fields)
    norm = Normalize(vmin=0,vmax=maximum)
    info["display_conventions"]["field_color_bounds_m4"] = [0.,float(maximum)]
    fig,axes = plt.subplots(2,2,figsize=(DOUBLE,4.8))
    fig.subplots_adjust(left=.09,right=.84,bottom=.11,top=.93,wspace=.3,hspace=.3)
    for letter,ax,t,data in zip("abcd",axes.flat,settings["times"],fields):
        mesh = ax.pcolormesh(z,x*1e6,data,norm=norm,cmap="magma",shading="flat",rasterized=True)
        ax.set(yscale="linear",ylim=(1,100),xlabel=r"Axial position $z$ (m)",ylabel=r"Size $x$ ($\mu$m)")
        panel(ax,letter)
        ax.text(.97,.96,rf"$t={t:g}\,\mathrm{{s}}$",transform=ax.transAxes,ha="right",va="top",color="black",
                bbox=dict(facecolor="white",edgecolor="none",alpha=.8,pad=1))
    coloraxis = fig.add_axes([.88,.16,.025,.74])
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0,0))
    fig.colorbar(mesh,cax=coloraxis,label=r"Number density $n$ (m$^{-4}$)",format=formatter)
    export(fig,3,"precipitation_fields",info)
    fig,axes = plt.subplots(1,2,figsize=(DOUBLE,3.55))
    fig.subplots_adjust(left=.09,right=.98,bottom=.18,top=.93,wspace=.3)
    colors = ["#0072B2", "#D55E00", "#009E73", "#6A3D9A"]
    for color,nz_ in zip(colors,info["configuration"]["axial_cells"]):
        data = table(runs[f"axial_Nz{nz_}"][0]/"outlet.csv")
        edges = np.r_[data["left_m"],data["right_m"][-1]]*1e6
        for ax,quantity in zip(axes,(data["number_density_m4"]/1e15,data["volume_density_per_m"]*1e-6)):
            ax.stairs(quantity,edges,color=color,lw=1.0,label=rf"$N_z={nz_}$")
    data = table(folder/"outlet.csv")
    edges = np.r_[data["left_m"],data["right_m"][-1]]*1e6
    for ax,quantity in zip(axes,(data["number_density_m4"]/1e15,data["volume_density_per_m"]*1e-6)):
        ax.stairs(quantity,edges,color="k",lw=1.2,ls="--",label="Reference solution")
    if figures.MODE == "production":
        lit = table(PAPER/"precipitation/data/literature"/"cadet_reference.csv")
        literature_edges = np.r_[lit["left_m"],lit["right_m"][-1]]
        digitized = table(PAPER/"precipitation/data/literature"/"zhang_figure9c_digitized.csv")
        axes[0].errorbar(digitized["size_m"]*1e6,digitized["number_density_m4"]/1e15,
            xerr=digitized["size_uncertainty_m"]*1e6,yerr=digitized["density_uncertainty_m4"]/1e15,
            ls="none",marker="o",ms=3,mfc="white",color="k",elinewidth=.5,label="Reference data")
        comparisons = []
        for name in ("size_Nx200","reference"):
            values = table(runs[name][0]/"outlet.csv")
            edges_ = np.r_[values["left_m"],values["right_m"][-1]]
            comparisons.append(dict(configuration=name,
                reference="Public CADET size Upwind / axial WENO23; not published WENO35",
                number_normalized_L1=density_l1(edges_,values["number_density_m4"],literature_edges,lit["number_density_m4"]),
                normalized_volume_L1=density_l1(edges_,values["volume_density_per_m"],literature_edges,lit["volume_density_per_m"])))
        analysis_table(3,base,"public_cadet_comparison.csv",comparisons)
        extracted_comparison = []
        for name in ("size_Nx200","reference"):
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
    axes[1].legend(*axes[0].get_legend_handles_labels(),loc="upper left",fontsize=7.5,
                   handlelength=2.0,borderaxespad=.4)
    export(fig,3,"precipitation_outlet",info)

    # Changing one coordinate at a time prevents fine-axial/coarse-size runs
    # from being mixed into a misleading convergence or work-precision curve.
    comparisons = rows(base/"refinement.csv")
    for filename,work in (("precipitation_refinement",False),("precipitation_work_precision",True)):
        fig,axes = plt.subplots(1,2,figsize=(DOUBLE,3.65))
        fig.subplots_adjust(left=.14,right=.98,bottom=.25,top=.92,wspace=.45)
        for letter,ax,direction,variable in zip("ab",axes,("size","axial"),("nx","nz")):
            data = sorted((r for r in comparisons if r["refinement"]==direction and float(r["time_s"])==200.),
                key=lambda r:int(r[variable]))
            horizontal = [runs[r["comparison"]][1]["summary"]["seconds"] if work else int(r[variable]) for r in data]
            for quantity,color,marker,label in (("normalized_field_L1",colors[0],"s","Whole field"),
                    ("interpretable_outlet_L1",colors[1],"o","Outlet")):
                values = np.array([float(r[quantity]) for r in data])
                ax.loglog(horizontal,np.where(values>0,values,np.nan),color=color,marker=marker,
                    mfc="white",lw=1.2,ms=4,label=label)
            ax.set(xlabel="Integration wall time (s)" if work else
                (r"Size bins $N_x$" if direction=="size" else r"Axial cells $N_z$"),
                ylabel=r"Normalized $L^1$ difference")
            if not work:
                ax.set_xticks(horizontal,labels=[str(n) for n in horizontal])
                ax.xaxis.set_minor_locator(NullLocator())
            lower,upper = ax.get_ylim()
            if upper/lower < 3:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.yaxis.set_major_formatter(ScalarFormatter())
            else:
                ax.yaxis.set_major_locator(LogLocator(subs=(1,2,5),numticks=7))
                ax.yaxis.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=False,minor_thresholds=(np.inf,np.inf)))
            ax.yaxis.set_minor_locator(NullLocator())
            panel(ax,letter)
            ax.text(.96,.95,rf"$N_z={nz}$" if direction=="size" else rf"$N_x={nx}$",
                transform=ax.transAxes,ha="right",va="top",fontsize=8)
        fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=2)
        export(fig,3,filename,info)

    if figures.MODE=="production":
        destination = PAPER.parent/"docs/src/assets/continuous_precipitation_log"
        destination.mkdir(parents=True,exist_ok=True)
        for name in ("precipitation_fields","precipitation_outlet","precipitation_refinement","precipitation_work_precision"):
            for suffix in (".png",".json"):
                shutil.copyfile(PAPER/"precipitation/figures"/(name+suffix),destination/(name+suffix))
        for name in ("refinement.csv","analysis/runtimes.csv","analysis/refinement_observables.csv",
                "analysis/digitized_point_comparison.csv","literature/zhang_figure9c_digitized.csv",
                "literature/provenance.json"):
            shutil.copyfile(base/name,destination/Path(name).name)
        for name in ("accounting.csv","outlet.csv","provenance.toml"):
            shutil.copyfile(folder/name,destination/("reference_"+name))

if __name__=="__main__":
    figures.run_plot(3,precipitation)
