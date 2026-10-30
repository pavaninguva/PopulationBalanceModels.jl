# Package architecture schematic

From the repository root, with `paper/requirements.txt` installed:

```sh
python paper/architecture/generate_pbm_architecture.py
```

This runs no Julia calculations. It checks dependency and extension roles against
`Project.toml`, then exports `section2_architecture_revised.png` at 300 dpi and a
revision/hash receipt under `paper/figures/00_architecture/`. DejaVu Sans lettering
keeps the softer style; the population-balance equation alone uses external
LaTeX with Computer Modern. Other mathematical labels use STIX mathtext.
The artwork is rasterized from an in-memory vector image; no PDF/SVG
derivatives are saved. The PNG is also copied into
`docs/src/assets/architecture/`. Matplotlib, NumPy and PyMuPDF use the existing
paper environment; LaTeX is required for the equation. `--usetex` renders all
lettering through LaTeX. `--preview` uses mathtext throughout and produces a
differently named copy without replacing final assets.

The drawing adapts the supplied two-panel Matplotlib generator. Its architecture
is reviewed against the current public exports, source layout and extension
declarations; each generated receipt records the revision and working-tree state. It is
7.2 inches wide and 8.8 inches high, with 7–8 pt body text. The layout
separates the two problem branches, forward UQ and dependency roles. The distribution
glyph is schematic, not benchmark data. No artwork from other packages is used.

Transient initial conditions/time spans and stationary numerical guesses remain
distinct. A solid path leads directly from compilation to either problem type.
Dashed side inputs identify two optional additions with equal visual weight:
`PBMSystem` for population/bulk coupling, and `AxialDispersion` for prescribed
reactor transport. The axial domain can be selected during compilation for a
single population; `axial_problem` lifts an eligible coupled local problem.
The additions can be combined where their compatibility contracts permit it.
Both solution branches feed observation operators and the shared
estimation workflow. Stationary inference uses checked equilibria and implicit
sensitivities for smooth isolated roots; AD-incompatible closures need an explicit
finite-difference choice. Each numerical route retains its compatibility
restrictions. DQMOM evolves weights and weighted
abscissas, unlike the raw-moment closures.

The forward-UQ row uses `UncertainInputs` and `UQProblem` to vary parameters,
initial states or prescribed conditions through a scenario builder. Monte Carlo
and non-intrusive PCE reuse eligible transient/stationary solves and observation
operators. Output summaries and independent PCE validation are separate from
parameter fitting. MC accepts joint distributions or retained posterior draws.
The built-in PCE bases cover independent normal, uniform and beta reference
marginals, with a declared map to physical inputs. Regression, quadrature
projection and validation/refinement reuse the same forward interface.
These propagation methods need no additional backend beyond
the package's core dependencies. The receipt also hashes the reviewed UQ sources.
