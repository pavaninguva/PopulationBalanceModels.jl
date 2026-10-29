"""Draw the two milling paper figures; never run simulations or change numerical data."""
from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PAPER))
from _support import figures
from _support.figures import load_case,table,export
from plotting_style import configure,panel,DOUBLE

COLORS = {"mc":"#24679a","pce1":"#bd6b28","pce2":"#26816b"}
LABELS = {"mc":"MC","pce1":"PCE, degree 1","pce2":"PCE, degree 2"}


def milling(input_path=None):
    base,info,producers = load_case(2,check_checkout=False,input_path=input_path)
    forward,reference = table(base/"forward.csv"),table(base/"reference.csv")
    cfg = info["configuration"]
    if set(forward["bins"]) != {cfg["forward_bins"]}:
        raise ValueError("The forward figure must contain only the selected grid")
    info["display_conventions"] = dict(
        forward="One conservative fixed-pivot/CVODE-BDF scheme; 100 power-grid pivots; two constant speeds",
        forward_median="Instantaneous outlet volume median; linear diameter interpolation of the pivot-volume CDF, as in the UQ example",
        reference="Barrasso (2013) Figure 7 simulation markers; original graph readings, not experimental data",
        uq="Original 12,000-draw MC, 12-point PCE1 and 32-point PCE2 summaries; full Gaussian input law",
        bands="Pointwise 2.5/97.5% output quantiles and approximate mean +/-1.96 output SD, not confidence intervals on the mean",
        timing="No new simulations or timings in this redraw; original numerical producers are retained")

    fig,axes = plt.subplots(1,2,figsize=(DOUBLE,3.1))
    fig.subplots_adjust(left=.10,right=.985,top=.92,bottom=.19,wspace=.30)
    speeds = {3350:"#24679a",4923:"#26816b"}
    for rpm,color in speeds.items():
        data = forward[forward["rpm"] == rpm]
        if len(data) == 0 or np.any(np.diff(data["time_s"]) <= 0):
            raise ValueError(f"Missing or unordered {rpm} RPM forward data")
        for ax,field,quantity in zip(axes,(cfg["forward_statistic"],"holdup_g"),("d50_um","holdup_g")):
            ax.plot(data["time_s"],data[field],color=color,lw=1.2,label=f"{rpm} RPM")
            points = reference[(reference["rpm"] == rpm)&(reference["quantity"] == quantity)]
            ax.plot(points["time_s"],points["value"],ls="none",marker="o",
                    ms=3.2,mfc="white",mec=color,mew=.7)
    axes[0].set(ylim=(0,1000),ylabel=r"Outlet $d_{50}$ ($\mu$m)")
    axes[1].set(ylim=(0,250),ylabel="Mill hold-up (g)")
    for letter,ax in zip("ab",axes):
        ax.set(xlim=(0,180),xlabel=r"Time, $t$ (s)")
        panel(ax,letter)
    handles = [Line2D([],[],color=color,lw=1.2,label=f"{rpm} RPM") for rpm,color in speeds.items()]
    handles.append(Line2D([],[],color=".2",ls="none",marker="o",mfc="white",ms=3.2,label="Reference data"))
    axes[1].legend(handles=handles,loc="lower right",fontsize=7.5)
    export(fig,2,"milling_forward",info)

    # The same three-panel presentation and numerical summaries as the manual:
    # common scales, stored quantiles, nominal curve and source-convention band.
    nominal = table(base/"nominal.csv")
    summaries = {name:table(base/f"{name}_summary.csv") for name in COLORS}
    for data in summaries.values():
        if not np.array_equal(data["time_s"],nominal["time_s"]):
            raise ValueError("UQ summaries do not share the nominal output times")
    low = min(0.,*(float(np.min(d[key])) for d in summaries.values() for key in ("gaussian_low_um","q025_um")))
    high = max(float(np.max(d[key])) for d in summaries.values() for key in ("gaussian_high_um","q975_um"))
    limits = (low-.03*(high-low),high+.05*(high-low))
    fig,axes = plt.subplots(1,3,figsize=(DOUBLE,2.85),sharex=True,sharey=True)
    fig.subplots_adjust(left=.10,right=.985,top=.91,bottom=.30,wspace=.12)
    for letter,ax,name in zip("abc",axes,COLORS):
        data,color = summaries[name],COLORS[name]
        t = data["time_s"]
        ax.fill_between(t,data["q025_um"],data["q975_um"],color=color,alpha=.20,lw=0)
        ax.plot(t,data["gaussian_low_um"],color=color,ls=":",lw=.75,alpha=.75)
        ax.plot(t,data["gaussian_high_um"],color=color,ls=":",lw=.75,alpha=.75)
        ax.plot(t,data["mean_um"],color=color,lw=1.25)
        ax.plot(nominal["time_s"],nominal["d50_um"],color="#424242",ls="--",lw=.8)
        ax.axvline(30,color=".5",ls=(0,(3,3)),lw=.6,zorder=0)
        ax.set(xlim=(0,200),ylim=limits,xlabel=r"Time, $t$ (s)",xticks=[0,100,200])
        ax.tick_params(direction="out")
        ax.legend(handles=[Line2D([],[],color=color,label=LABELS[name])],loc="upper right",frameon=False)
        panel(ax,letter)
    axes[0].set_ylabel(r"$d_{50}$ ($\mu$m)")
    handles = [Line2D([],[],color="black",lw=1.25,label="Propagated mean"),
        Line2D([],[],color=".3",ls="--",lw=.8,label="Inputs at their means"),
        Patch(facecolor=".55",alpha=.25,label=r"Pointwise 95\% output quantiles"),
        Line2D([],[],color=".4",ls=":",lw=.8,label=r"Mean $\pm1.96$ SD")]
    fig.legend(handles=handles,loc="lower center",ncol=2,frameon=False,
               bbox_to_anchor=(.52,.005),columnspacing=1.5,handlelength=2.4)
    export(fig,2,"milling_uq",info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,help="A staged milling paper data directory")
    parser.add_argument("--retained",action="store_true",help="Retained redraw is the default; original producers are preserved")
    args = parser.parse_args()
    print("Redrawing retained production curves; no simulations or numerical-table updates.",flush=True)
    configure()
    milling(args.input)
