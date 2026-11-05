# Package architecture schematic

From the repository root, with `paper/requirements.txt` installed:

```sh
python paper/architecture/generate_pbm_architecture.py --paper-only
```

This runs no Julia calculations. It checks dependency and extension roles against
`Project.toml`, then exports `architecture.png` at 300 dpi and a
revision/hash receipt under `paper/figures/00_architecture/`. DejaVu Sans lettering
keeps the softer style; the population-balance equation alone uses external
LaTeX with Computer Modern. Other mathematical labels use STIX mathtext.
The artwork is rasterized from an in-memory vector image; no PDF/SVG
derivatives are saved under `paper/`. Paper panels use lowercase Roman numerals.
Omit `--paper-only` to also refresh
`docs/src/assets/architecture/section2_architecture_revised.png` and its PDF/SVG
downloads, retaining alphabetic labels for the manual. The receipt retains
its `section2_architecture_revised.json` name and
records the paper PNG filename. Matplotlib, NumPy and PyMuPDF use the existing
paper environment; LaTeX is required for the equation. `--usetex` renders all
lettering through LaTeX. `--preview` uses mathtext throughout and produces a
differently named copy without replacing final assets.

The focused rendering checks need the same Python environment:

```sh
MPLBACKEND=Agg python test/architecture.py
```

They check label spacing, branch/optional-input structure, export dimensions,
paper/docs label scopes and preview isolation without running simulations.

The drawing adapts the supplied two-panel Matplotlib generator. Its architecture
is reviewed against the current public exports, source layout and extension
declarations; each generated receipt records the revision and working-tree state. It is
7.2 inches wide and 8.2 inches high, with 7–8 pt body text. The layout
separates the two problem branches, forward UQ and dependency roles. The distribution
glyph is schematic, not benchmark data. No artwork from other packages is used.

The direct-discretization route selects `Grid1D`/`Grid2D` and `TermMethods`;
the moment route selects a `MomentSet` basis and `moment_method` family/resolution.
`compile` checks compatibility and prepares fixed numerical structure. Supported
moment contributions use tracked moments analytically and obtain only missing
moments/integrals from the selected reconstruction. A selected family still
controls its state layout and admissibility, even for analytically closed
contributions. DQMOM advances weights and weighted abscissas, not raw moments.

Transient initial conditions/time spans and stationary numerical guesses remain
distinct. `PBMProblem` automatically converts eligible physical initial densities
or probability laws into the selected state through `initialize`; explicit
numerical arrays keep their meaning. `ContinuousDaughters` supplies an expected
daughter-number law inside `Breakage`, not a new independent physical term.
Normalization, physical support, conserved properties and unresolved tails remain
explicit choices; see the forward-execution guide and the case studies.

The problem branches show only their essential inputs and solve calls. The
caption and guide explain execution details: `prepare_run` binds parameter-dependent
projections and solve-local storage automatically, separate from compilation. State/time-
dependent rates still use the current integration inputs. `ForwardSolver` selects
eligible native, external or sensitivity execution; the simple `method`/`dt`
form is normalized internally and remains supported. Optional `StepControl`
provides native residual-RK stability/admissibility control, not integration-error
control or a universal positivity guarantee. External completion/saved-state checks
do not establish native internal-stage validity. `OperatingHistory` optionally
segments prescribed fixed transitions with continuous state transfer and
right-hand observations. Complete updates and derivative backends retain their
compatibility restrictions. Stationary root solving does not inherit these
transient execution controls or operating-history semantics.

A solid path leads directly from compilation to either problem type.
Dashed side inputs identify two optional additions with equal visual weight:
`PBMSystem` for population/bulk coupling, and `AxialDispersion` for prescribed
reactor transport. The axial domain can be selected during compilation for a
single population; `axial_problem` lifts an eligible coupled local problem.
The additions can be combined where their compatibility contracts permit it.
Both solution branches feed observation operators and the shared
estimation workflow. Stationary inference uses checked equilibria and implicit
sensitivities for smooth isolated roots; AD-incompatible closures need an explicit
finite-difference choice. Each numerical route retains its compatibility
restrictions. The drawing is a workflow, not a claim of universal combinations.

The forward-UQ row uses `UncertainInputs` and `UQProblem` to vary parameters,
initial states or prescribed conditions through a scenario builder. Monte Carlo
and non-intrusive PCE reuse eligible transient/stationary solves and observation
operators. Output summaries and independent PCE validation are separate from
parameter fitting. MC accepts joint distributions or retained posterior draws.
The built-in PCE bases cover independent normal, uniform and beta reference
marginals, with a declared map to physical inputs. Regression, quadrature
projection and validation/refinement reuse the same forward interface.
These propagation methods need no additional backend beyond
the package's core dependencies. The receipt also hashes the reviewed UQ sources and the preparation, execution,
moment, initialization and stationary source files checked for this revision.
These hashes record the review; they do not require refreshing an otherwise
accurate figure after every unrelated source edit.
