"""Regenerate this case's paper figures from its own numerical exports."""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
PAPER=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PAPER))
from _support import figures
from _support.figures import load_case, rows, table, export, analysis_table
from plotting_style import panel,method_style,DOUBLE,METHODS

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

if __name__=="__main__":
    figures.run_plot(1,growth)
