"""Render the native stem-cell example's validated exports; never run a fit."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import tomllib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from plot_style import configure

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "docs/src/assets/stem_cells"
COLORS = {"self_renewal": "#0072B2", "differentiation": "#D55E00"}
LABELS = {"self_renewal": "Self-renewal", "differentiation": "Differentiation"}


def read(folder, name):
    return np.atleast_1d(np.genfromtxt(folder / name, delimiter=",", names=True,
                                     dtype=None, encoding="utf8"))


def validate(folder):
    settings = tomllib.loads((folder / "run.toml").read_text())
    if settings["mode"] != "production" or settings["settings"]["tf"] != 200:
        raise ValueError("Final figures require the complete 0–200 production study")
    for file, digest in settings["source_hashes"].items():
        if hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == digest:
            continue
        if file == "Project.toml" and "project_snapshot" in settings:
            snapshot = (folder / settings["project_snapshot"]).read_bytes()
            if hashlib.sha256(snapshot).hexdigest() != digest:
                raise ValueError("Production Project snapshot hash mismatch")
            def runtime_contract(project):
                dependencies = set(project.get("deps", {})) | set(project.get("weakdeps", {})) | {"julia"}
                return ({k: v for k, v in project.items() if k not in ("extras", "targets", "compat")},
                        {k: v for k, v in project.get("compat", {}).items() if k in dependencies})
            if runtime_contract(tomllib.loads(snapshot.decode())) == runtime_contract(tomllib.loads((ROOT / file).read_text())):
                continue  # Test-only dependencies do not change the forward calculation.
        raise ValueError(f"Changed numerical source {file}; rerun the example")
    for file, digest in settings["files"].items():
        if hashlib.sha256((folder / file).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Changed numerical data {file}")
    return settings


def save(fig, folder, name):
    for extension in ("png", "svg", "pdf"):
        path = folder / f"{name}.{extension}"
        fig.savefig(path, dpi=300)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def panel(ax, letter):
    ax.text(-.12, 1.04, f"({letter})", transform=ax.transAxes, fontsize=12)


def dynamics(folder):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    fig.subplots_adjust(left=.1, right=.88, bottom=.09, top=.96, wspace=.3, hspace=.36)
    fields = {name: read(folder, f"distribution_{name}.csv") for name in COLORS}
    vmax = max(np.max(data["normalized_density"]) for data in fields.values())
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    cmap.set_under("#eeeeee")
    for letter, ax in zip("abcd", axes.flat):
        panel(ax, letter)
    for ax, (name, data) in zip(axes[0], fields.items()):
        times = np.unique(data["time"])
        first = data[data["time"] == times[0]]
        edges = np.r_[first["left"], first["right"][-1]]
        time_edges = np.r_[times[0], (times[:-1] + times[1:]) / 2, times[-1]]
        density = data["normalized_density"].reshape(len(times), len(first))
        if np.min(density) < 0:
            raise ValueError("Negative populations cannot be masked for display")
        image = ax.pcolormesh(edges, time_edges, np.ma.masked_equal(density, 0),
                              shading="flat", cmap=cmap, norm=LogNorm(1e-5, vmax), rasterized=True)
        ax.set(xlim=(0, edges[-1]), ylim=(0, times[-1]),
               xlabel="Marker cargo (a.u.)", ylabel="Model time")
        ax.text(.98, .96, LABELS[name], ha="right", va="top", transform=ax.transAxes,
                fontsize=9, bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
    colorbar = fig.colorbar(image, cax=fig.add_axes([.91, .58, .018, .37]), extend="min")
    colorbar.set_label(r"Normalized density (a.u.$^{-1}$)")
    for name, color in COLORS.items():
        series = read(folder, f"trajectory_{name}.csv")
        axes[1, 0].plot(series["time"], series["stem"], color=color,
                        label=f"Pluripotent: {LABELS[name].lower()}")
        if name == "differentiation":
            # Zero accumulated exits at t=0 have no logarithm; no floor is added.
            active = series["differentiated"] > 0
            axes[1, 0].plot(series["time"][active], series["differentiated"][active],
                            "--", color="k", label="Accumulated differentiated exits")
        axes[1, 1].fill_between(series["time"], series["q10"], series["q90"], color=color, alpha=.12)
        axes[1, 1].plot(series["time"], series["mean_cargo"], color=color, label=LABELS[name])
    axes[1, 0].set(xlabel="Model time", ylabel=r"Cell concentration / $C_{p,0}$", yscale="log", xlim=(0, 200))
    axes[1, 1].set(xlabel="Model time", ylabel="Pluripotent-cell cargo (a.u.)", xlim=(0, 200))
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    save(fig, folder, "stem_cell_dynamics")


def physiology(folder):
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.8))
    fig.subplots_adjust(left=.07, right=.99, bottom=.12, top=.94, wspace=.4, hspace=.43)
    for letter, ax in zip("abcdefgh", axes.flat):
        panel(ax, letter)
    labels = {"total": ("Total", "#0072B2"), "dividing": ("Division events", "#D55E00"),
              "newborn": ("Newborn daughters", "#009E73"), "differentiating": ("Differentiation events", "#CC79A7")}
    for row, name in enumerate(COLORS):
        data = read(folder, f"events_{name}.csv")
        edges = np.r_[data["left"], data["right"][-1]]
        for key, (label, color) in labels.items():
            if key == "differentiating" and name == "self_renewal":
                continue
            axes[row, 0].stairs(data[key], edges, label=label, color=color, lw=1.2)
        axes[row, 0].set(xlabel="Marker cargo (a.u.)", ylabel=r"Density (a.u.$^{-1}$)", xlim=(0, 100))
        axes[row, 0].legend(fontsize=7)
        axes[row, 0].text(.98, .04, LABELS[name], transform=axes[row, 0].transAxes,
                          ha="right", va="bottom", fontsize=8)
        for column, key, ylabel in ((1, "R", r"$R(x)$ (a.u. / model time)"),
                                     (2, "gamma", r"$\Gamma(x)$ (model time$^{-1}$)"),
                                     (3, "kappa", r"$\kappa(x)$ (model time$^{-1}$)")):
            ax = axes[row, column]
            shown = data["x"] <= 100
            ax.plot(data["x"][shown], data[key][shown], "k-", lw=1.2, label="Specified function")
            if not (key == "kappa" and name == "self_renewal"):
                ax.plot(data["x"][shown], data[key + "_recovered"][shown], color="#D55E00", ls="--", lw=1.2,
                        label="Balanced-shape estimate" if key == "R" else "Event-PDF identity")
            ax.set(xlabel="Marker cargo (a.u.)", ylabel=ylabel, xlim=(0, 100))
        if row == 0:
            axes[row, 1].legend(fontsize=7, loc="lower center")
            axes[row, 2].legend(fontsize=7, loc="upper left")
    # Common limits within each physical quantity; never clip a reconstruction.
    for column in range(4):
        lo = min(ax.get_ylim()[0] for ax in axes[:, column])
        hi = max(ax.get_ylim()[1] for ax in axes[:, column])
        for ax in axes[:, column]:
            ax.set_ylim(lo, hi)
    save(fig, folder, "stem_cell_physiology")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT)
    parser.add_argument("--usetex", action="store_true")
    args = parser.parse_args()
    configure(args.usetex)
    validate(args.data_dir)
    dynamics(args.data_dir)
    physiology(args.data_dir)
    names = [f"{stem}.{extension}" for stem in ("stem_cell_dynamics", "stem_cell_physiology")
             for extension in ("png", "svg", "pdf")]
    metadata = dict(renderer="LaTeX" if args.usetex else "MathText preview",
                    python=platform.python_version(), numpy=np.__version__,
                    matplotlib=matplotlib.__version__, png_dpi=300,
                    plotter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    input_manifest_sha256=hashlib.sha256((args.data_dir / "run.toml").read_bytes()).hexdigest(),
                    files={name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
                           for name in names})
    (args.data_dir / "figures.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.data_dir)


if __name__ == "__main__":
    main()
