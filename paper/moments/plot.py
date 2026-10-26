"""Regenerate this case's paper figures from its own numerical exports."""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
PAPER=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PAPER))
from _support import figures
from _support.figures import (load_case,rows,table,export,analysis_table,
    MOMENT_METHODS,MOMENT_LABELS)
from plotting_style import panel, DOUBLE

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

if __name__=="__main__":
    figures.run_plot(2,moments)
