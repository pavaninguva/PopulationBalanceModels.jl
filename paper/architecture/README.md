# Package architecture schematic

From the repository root, with `paper/requirements.txt` installed:

```sh
python paper/architecture/generate_pbm_architecture.py --usetex
```

This runs no Julia calculations. It checks dependency and extension roles against
`Project.toml`, then exports `section2_architecture_revised.{pdf,svg,png}` and a
revision/hash receipt under `paper/figures/00_architecture/`. The PNG is rendered
from the LaTeX vector PDF at 300 dpi. Final assets are also copied into
`docs/src/assets/architecture/`. Matplotlib, NumPy, PyMuPDF and LaTeX use the
existing paper environment. `--preview` explicitly produces differently named
mathtext previews; they do not replace final assets.

The drawing adapts the supplied two-panel Matplotlib generator. Its architecture
is reviewed against the current public exports, source layout and extension
declarations; each generated receipt records the revision and working-tree state. It is
7.2 inches wide and 8.4 inches high, with 7–8 pt body text. The taller layout
keeps the two problem branches and dependency roles legible. The distribution
glyph is schematic, not benchmark data. No artwork from other packages is used.

Transient initial conditions/time spans and stationary numerical guesses remain
distinct. Both solution branches feed observation operators and the shared
estimation workflow. Stationary inference uses checked equilibria and implicit
sensitivities for smooth isolated roots; AD-incompatible closures need an explicit
finite-difference choice. Each numerical route retains its compatibility
restrictions. DQMOM evolves weights and weighted
abscissas, unlike the raw-moment closures.
