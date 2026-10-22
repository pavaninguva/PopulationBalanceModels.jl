"""Plot measured compact-growth errors and work, retaining signed results."""
from pathlib import Path
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_style import configure
from plot_case_studies import save_figure


def run(usetex=False):
    configure(usetex=usetex)
    folder = Path(__file__).resolve().parents[1] / "src/assets/compact_fv"
    data = np.genfromtxt(folder / "results.csv", delimiter=",", names=True,
                         dtype=None, encoding="utf-8")
    orders = []
    for scenario in dict.fromkeys(data["scenario"]):
        for method in dict.fromkeys(data[data["scenario"] == scenario]["method"]):
            rows = data[(data["scenario"] == scenario) & (data["method"] == method)]
            temporal = scenario.startswith("temporal")
            rows = np.sort(rows, order="dt" if temporal else "nx")
            if temporal:
                rows = rows[::-1]
            for coarse, fine in zip(rows[:-1], rows[1:]):
                ratio = coarse["dt"] / fine["dt"] if temporal else fine["nx"] / coarse["nx"]
                slope = np.log(coarse["L1"] / fine["L1"]) / np.log(ratio)
                status = "roundoff" if min(coarse["L1"], fine["L1"]) < 1e-12 else "measured"
                orders.append((scenario, method, "dt" if temporal else "nx",
                               coarse["dt"] if temporal else coarse["nx"],
                               fine["dt"] if temporal else fine["nx"], slope, status))
    with (folder / "orders.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("scenario", "method", "axis", "coarse", "fine", "L1_order", "status"))
        writer.writerows(orders)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), layout="constrained")
    for scenario, label in (("affine", "Affine growth"), ("exponential", "Exponential growth"),
                            ("physical_inflow", "Prescribed inflow"),
                            ("upper_escape", "Upper-boundary escape")):
        rows = data[data["scenario"] == scenario]
        axes[0, 0].loglog(rows["nx"], rows["L1"], "o-", label=label)
    for scenario, label in (("rectangular_translation", "Constant growth"),
                            ("rectangular_variable", "Affine / exponential growth")):
        rows = data[data["scenario"] == scenario]
        axes[0, 1].loglog(rows["nx"], rows["L1"], "o-", label=label)
    for scenario, label in (("temporal_constant", "1D constant"),
                            ("temporal_prescribed", "1D prescribed in time"),
                            ("temporal_rectangular", "2D constant")):
        rows = data[data["scenario"] == scenario]
        axes[1, 0].loglog(rows["dt"], rows["L1"], "o-", label=label)
    rows = data[data["scenario"] == "work_precision"]
    for method, label in (("HOCFiniteVolume", "Compact / midpoint"),
                          ("WENO3_RK4", "WENO3 / RK4"),
                          ("WENO5_RK4", "WENO5 / RK4"),
                          ("Inguva2022", "Inguva2022 (CFL=1)")):
        r = rows[rows["method"] == method]
        axes[1, 1].loglog(r["seconds"], r["L1"], "o-", label=label)
    axes[0, 0].set(xlabel=r"Size bins $N_x$", ylabel=r"1D integrated $L^1$ error")
    axes[0, 1].set(xlabel=r"Size bins $N_x$ ($N_y=3N_x/4$)", ylabel=r"2D integrated $L^1$ error")
    axes[1, 0].set(xlabel=r"Time step $\Delta t$", ylabel=r"Time error against matrix exponential")
    axes[1, 1].set(xlabel="Warmed solve time (s)", ylabel=r"Integrated $L^1$ error")
    for ax in axes.flat:
        ax.legend(fontsize=8)
    save_figure(fig, folder, "verification")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.7), layout="constrained")
    rows = data[data["scenario"] == "rectangular_work"]
    for method, label in (("HOCFiniteVolume", "Conservative compact / midpoint"),
                          ("HOCADI4", "Periodic compact / fourth-order time"),
                          ("WENO5_RK4", "WENO5 / RK4")):
        r = rows[rows["method"] == method]
        ax.loglog(r["seconds"], r["L1"], "o-", label=label)
    ax.set(xlabel="Warmed solve time (s)", ylabel=r"2D integrated $L^1$ error")
    ax.legend(fontsize=9)
    save_figure(fig, folder, "work_precision_2d")
    plt.close(fig)

    field = np.genfromtxt(folder / "variable_2d.csv", delimiter=",", names=True)
    x, y = np.unique(field["x"]), np.unique(field["y"])
    numerical = field["density"].reshape(len(y), len(x))
    exact = field["reference_density"].reshape(len(y), len(x))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), layout="constrained")
    for ax, value, title in zip(axes, (exact, numerical, numerical-exact),
                               ("Characteristic reference", "Conservative compact", "Density difference")):
        difference = title == "Density difference"
        bound = np.max(np.abs(value)) if difference else max(np.max(exact), np.max(numerical))
        artist = ax.pcolormesh(x, y, value, shading="nearest",
                              cmap="RdBu_r" if difference else "viridis",
                              vmin=-bound if difference else min(0., np.min(numerical)), vmax=bound)
        ax.set(xlabel=r"Size $x$", ylabel=r"Size $y$", title=title)
        fig.colorbar(artist, ax=ax, label=r"Bin-average density")
    save_figure(fig, folder, "variable_2d")
    plt.close(fig)

    births = np.genfromtxt(folder / "boundary_births.csv", delimiter=",", names=True)
    fig, ax = plt.subplots(figsize=(6, 3.3), layout="constrained")
    ax.plot(births["x"], births["reference_density"], "k--", label="Characteristic bin averages")
    ax.plot(births["x"], births["density"], label="Compact, signed values retained")
    ax.axhline(0, color="0.7", linewidth=.6)
    ax.set(xlim=(0, .5), xlabel=r"Size $x$", ylabel=r"Bin-average density")
    ax.legend(fontsize=9)
    save_figure(fig, folder, "boundary_births")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--usetex", action="store_true")
    run(parser.parse_args().usetex)
