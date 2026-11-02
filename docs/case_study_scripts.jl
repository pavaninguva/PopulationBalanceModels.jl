"""Assemble readable tutorial scripts from the maintained examples, without running them."""
module CaseStudyScripts

const ROOT = normpath(joinpath(@__DIR__, ".."))
const CASES = (:constant_growth, :crystallization, :moment_crystallization,
    :continuous_precipitation, :protein_crystallization, :stem_cells,
    :nanoparticles, :transport, :estimating_growth, :milling_uncertainty)

function sections(path)
    result = Dict{String,String}()
    active = nothing
    content = String[]
    for line in eachline(path)
        start = match(r"^# tutorial:start ([a-zA-Z0-9_]+)$", line)
        stop = match(r"^# tutorial:end ([a-zA-Z0-9_]+)$", line)
        if start !== nothing
            active === nothing || error("Nested tutorial section in $path")
            active = only(start.captures)
            haskey(result, active) && error("Duplicate tutorial section $active in $path")
            empty!(content)
        elseif stop !== nothing
            active == only(stop.captures) || error("Unmatched tutorial section in $path")
            result[active] = strip(join(content, '\n')) * "\n"
            active = nothing
        elseif active !== nothing
            push!(content, line)
        end
    end
    active === nothing || error("Unclosed tutorial section $active in $path")
    result
end

function source(relative, section; root=ROOT)
    pieces = sections(joinpath(root, relative))
    haskey(pieces, section) || error("Missing tutorial section $section in $relative")
    "# From $relative ($section).\n" * pieces[section]
end

# Select complete top-level definitions, preserving their original formatting.
# This is source presentation only: no callback is classified or evaluated here.
function definition_name(ex)
    ex isa Expr || return nothing
    ex.head === :macrocall && return definition_name(last(ex.args))
    ex.head === :const && return definition_name(only(ex.args))
    ex.head in (:function, :(=)) || return nothing
    lhs = first(ex.args)
    lhs isa Symbol && return lhs
    lhs isa Expr && lhs.head === :where && (lhs = first(lhs.args))
    lhs isa Expr && lhs.head === :call && first(lhs.args) isa Symbol ? first(lhs.args) : nothing
end

function definitions(relative, section, names; root=ROOT)
    text = sections(joinpath(root, relative))[section]
    # Two legacy recipes mark their entire module. Extract its declarations,
    # rather than embedding a second module and all its numerical machinery.
    lines = split(text, '\n')
    if startswith(first(lines), "module ")
        lastline = findlast(line -> !isempty(strip(line)), lines)
        startswith(strip(lines[lastline]), "end") || error("Unclosed source module")
        text = join(lines[2:lastline-1], '\n')
    end
    selected = Dict{Symbol,String}()
    position = firstindex(text)
    while position <= lastindex(text)
        ex, next = Meta.parse(text, position; raise=true)
        ex === nothing && break
        name = definition_name(ex)
        if name in names
            snippet = strip(text[position:prevind(text,next)])
            if ex isa Expr && ex.head === :macrocall
                # Omit long reference docstrings from the compact walkthrough.
                snippet = replace(snippet, r"(?s)^\s*\"\"\".*?\"\"\"\s*" => ""; count=1)
            end
            selected[name] = snippet * "\n"
        end
        position = next
    end
    missing = setdiff(collect(names), collect(keys(selected)))
    isempty(missing) || error("Missing tutorial definitions $missing in $relative ($section)")
    "# Physical definitions from $relative.\n" * join((selected[name] for name in names), "\n")
end

function script(name::Symbol; root=ROOT)
    name in CASES || throw(ArgumentError("Unknown case study: $name"))
    take(path, section) = source("examples/" * path, section; root)
    pick(path, section, names) = definitions("examples/" * path, section, names; root)
    imports = "using PopulationBalanceModels\n"
    pieces = String[]
    body = if name === :constant_growth
        imports *= "using Distributions: Normal\n"
        push!(pieces,take("constant_growth_methods.jl","velocity"))
        push!(pieces,pick("constant_growth_methods.jl","gaussian",
            (:gaussian_height,:gaussian_center,:gaussian_width,:gaussian_initial)))
        raw"""
function tutorial_problem(; bins=200)
    grid = Grid1D(range(0.0, 1.0; length=bins+1); coordinate=:length)
    model = PBMModel(growth=Growth(AffineGrowth(growth, 0.0)))
    boundary = SizeBC(OpenBoundary(), OpenBoundary())
    operator = compile(model, grid;
        methods=TermMethods(growth=MUSCL()), bc=boundary)
    initial = gaussian_initial() # Number concentration, with physical [0,1] support.
    problem = PBMProblem(operator, initial, (0.0, 4.0))
    reference = vec(initialize(operator, gaussian_initial(4.0))) # Exact translation.
    (; grid, model, boundary, initial, problem, reference)
end

function main(; bins=200)
    case = tutorial_problem(; bins)
    dt = 0.5 * first(case.grid.widths) / growth
    solver = ForwardSolver(SSPRK33(); dt)
    solution = simulate(case.problem; solver, saveat=[4.0])
    error = (vec(last(solution.u)) .- case.reference) ./ case.grid.widths
    @show sum(case.grid.widths .* abs.(error)) # Integrated L1 density error.
    solution
end
"""
    elseif name === :crystallization
        push!(pieces, pick("papers/gunawan2004_case2.jl", "physics",
            (:parameters, :temperature, :solubility, :benchmark_initial_density,
             :supersaturation, :growth_scale, :growth_rate, :nucleation_rate)))
        push!(pieces,pick("papers/gunawan2004_case2.jl","assembly",(:assemble_problem,:density_problem)))
        raw"""
function tutorial_problem()
    p = parameters() # µm, seconds, g solute/g water; rho is g/µm³.
    edges = sort!(unique(vcat(collect(0.0:2.0:1600.0), 180.5, 210.5)))
    grid = Grid1D(edges; coordinate=:length)
    # The compact support breakpoints are grid edges; the density is not normalized.
    density_problem(;grid,initial_density=benchmark_initial_density,support=(180.5,210.5),
        c0=solubility(temperature(0.0)),tspan=(0.0,10.0),p)
end

function main(; times=[0.0,5.0,10.0])
    problem = tutorial_problem()
    solver = ForwardSolver(SSPRK33(); dt=0.05)
    solution = simulate(problem; solver, saveat=times)
    @show bulk(problem, last(solution.u), :c)
    @show observe(PopulationMoment(:crystals, 3), solution)
    solution
end
"""
    elseif name === :moment_crystallization
        imports *= "using SpecialFunctions: gamma\n"
        push!(pieces, take("qamar_moment_benchmarks.jl", "geometry"))
        raw"""
function tutorial_problem()
    basis = MomentSet(0:5; coordinate=:length) # Three-node QMOM.
    model = PBMModel(
        aggregation=Aggregation(ConstantKernel(1.0); coordinate_map=volume_map),
        breakage=Breakage((x,p,t,ctx) -> x^3, daughter_law; coordinate_map=volume_map),
    )
    operator = compile(model, basis; moment_method=QMOM())
    # Exponential volume distribution: N0=1, volume V0=0.001, volume=x^3.
    initial = [0.001^(k/3) * gamma(1+k/3) for k in basis.orders]
    PBMProblem(operator, initial, (0.0, 50.0))
end

function main(; times=0.0:1.0:50.0)
    problem = tutorial_problem()
    solver = ForwardSolver(SSPRK33(); dt=0.02)
    solution = simulate(problem; solver, saveat=times)
    @show observe(MomentObservable(0), solution)
    @show observe(MomentObservable(3), solution) # Conserved material moment.
    solution
end
"""
    elseif name === :continuous_precipitation
        push!(pieces, take("continuous_precipitation.jl", "model"))
        push!(pieces,take("continuous_precipitation.jl","bound"))
        raw"""
function tutorial_problem(; nx=50, nz=50)
    # Log size bins on [1,900] µm, uniform physical reactor cells.
    # Size and axial reconstruction are deliberately selected separately.
    build(; nx, nz, mesh=:log, scheme=WENO5(), axial_scheme=MUSCL(), tf=5.0)
end

function main(; nx=50, nz=50, times=[0.0,5.0])
    problem = tutorial_problem(; nx, nz)
    # A physical transport/depletion bound, with package-owned trial/stage checks.
    # This controls admissibility, not integration error; WENO has no limiter.
    bound = (pb,u,t) -> precipitation_step_bound(pb,u,t; fraction=0.3)
    solver = ForwardSolver(RK4(); dt=0.25, negative_tolerance=1e-8,
        control=StepControl(; cfl=nothing, bound, min_dt=1e-12))
    solution = simulate(problem; solver, saveat=times)
    @show extrema(bulk(problem, last(solution.u), :c))
    # Restore the population scaling when reporting particles/m³.
    outlet = CellObservable(PopulationMoment(:crystals, 0), nz)
    @show 1e12 .* observe(outlet, solution)
    solution
end
"""
    elseif name === :protein_crystallization
        imports *= "using Optim\nusing Distributions: Uniform, Exponential\n" *
            "# Import CSV readers, the checked interval solver and calibration starts.\n" *
            "include(joinpath(pkgdir(PopulationBalanceModels), \"examples\", \"isothermal_protein_crystallization.jl\"))\n" *
            "using .IsothermalProteinCrystallization: load_data, select_concentration, select_psds,\n" *
            "    model_name, stage_concentration, cfl_solve, fit_model\n"
        push!(pieces, take("isothermal_protein_crystallization.jl", "constants"))
        push!(pieces, pick("isothermal_protein_crystallization.jl", "model",
            (:parameters, :barrier, :growth_rate, :nucleation_rate, :nucleation)))
        push!(pieces, pick("isothermal_protein_crystallization.jl", "priors", (:parameter_specs,)))
        raw"""
function build(; p=parameters(), model=:cnt, bins=200, Lmax=50.0, tf=660.0)
    model = model_name(model)
    model === :cnt && (p=merge(p, (grd_strength=zero(p.grd_strength),)))
    0 < p.cstar < C0 || throw(ArgumentError("require 0 < solubility < initial concentration"))
    p.grd_strength >= 0 || throw(ArgumentError("diffusivity must be nonnegative"))
    grid = Grid1D(range(0.0, Lmax; length=bins+1); coordinate=:length)
    terms = (
        growth=Growth((L,p,t,ctx) -> growth_rate(p, stage_concentration(p,ctx))),
        primary=Nucleation(nucleation, 0.0),
    )
    if model === :cnt_grd
        terms = merge(terms, (dispersion=SizeDiffusion((L,p,t,ctx) -> p.grd_strength),))
    end
    methods = TermMethods(growth=MUSCL(), by_term=(primary=BoundaryInjection(:left),))
    operator = compile(PBMModel(; terms...), grid; methods, bc=SizeBC())
    exchange = ConservativeExchange(:crystals, :c, (L,p,t,ctx) -> p.rho*p.kv*L^3)
    system = PBMSystem((crystals=operator,); exchanges=(exchange,), nonnegative_bulk=(:c,))
    PBMProblem(system, (crystals=zeros(bins),), (0.0, tf); bulk0=(c=C0,), p)
end

function tutorial_problem(; model=:cnt, bins=200)
    model = model_name(model)
    data = load_data()
    concentration = select_concentration(data, :calibration)
    distributions = select_psds(data, :calibration)
    observed = [r.concentration_g_L for r in concentration]
    measurements = Measurement[Measurement([r.model_time_min for r in concentration], observed,
        BulkObservable(:c); likelihood=GaussianLikelihood(sigma=reshape(0.05 .* observed,1,:)),
        source=:concentration)]
    for rows in distributions
        selected = rows[2:2:end-1] # Same 22 interior CDF ordinates; endpoints excluded.
        x = [r.size_um for r in selected]
        F = reshape([r.conditional_cdf for r in selected], :, 1)
        covariance = SIGMA_F^2 .* [exp(-abs(a-b)/CORRELATION_LENGTH) for a in x,b in x]
        push!(measurements, Measurement([first(rows).time_min], F,
            PopulationCDF(:crystals, x; window=WINDOW);
            likelihood=GaussianLikelihood(; covariance),
            source=Symbol("CDF_", Int(first(rows).time_min))))
    end
    p = parameters(grd_strength=model === :cnt ? 0.0 : D0_PRIOR_MEAN)
    problem = build(; p, model, bins, tf=420.0)
    # This specialized bucketed schedule preserves the study's derivative contract.
    # StepControl rejects AD through state-dependent step decisions.
    solver = cfl_solve
    experiment = InferenceExperiment(:batch, problem, measurements...; solver)
    fixed = model === :cnt ? (cstar=p.cstar, gamma=FIXED_GAMMA, grd_strength=0.0) :
        (cstar=p.cstar, gamma=FIXED_GAMMA)
    EstimationProblem(parameter_specs(; p, model), (experiment,); fixed)
end

function main(; model=:cnt)
    estimation = tutorial_problem(; model)
    fitted = fit_model(estimation) # Three calibration-only starts; no MCMC.
    fitted.converged || error("MAP failed; inspect the fit diagnostics")
    theta = physical_parameters(estimation, fitted.minimizer, :batch)
    # Freeze calibration before predicting the 660-min held-out distribution.
    p = merge(parameters(), theta)
    solver = only(estimation.experiments).solver
    solution = solver(build(; p, model), [0.0, 74.0, 180.0, 420.0, 660.0])
    @show theta [bulk(solution.problem,u,:c) for u in solution.u]
    @show observe(PopulationQuantiles(:crystals, [0.1,0.5,0.9]; window=WINDOW), solution)
    (; estimation, fitted, solution)
end
"""
    elseif name === :stem_cells
        imports *= "using Distributions: Normal, Beta, pdf\n"
        push!(pieces, pick("stem_cell_population.jl", "model",
            (:cargo_rate, :division_rate, :differentiation_rate)))
        push!(pieces,take("stem_cell_population.jl","inputs"))
        push!(pieces,pick("stem_cell_population.jl","assembly",(:assemble_problem,:continuous_problem)))
        raw"""
function tutorial_problem(; differentiation=false, bins=600)
    # A short transient in synthetic model time, not the balanced-growth study.
    continuous_problem(; bins, tf=20.0, differentiation)
end

function main(; bins=600, differentiation=true, times=[0.0,10.0,20.0])
    problem = tutorial_problem(; bins, differentiation)
    # Autonomous, state-independent rates: this bound applies throughout the run.
    dt = min(0.05, suggest_dt(problem.operator, problem.u0, problem.p, 0.0; cfl=0.35))
    solver = ForwardSolver(SSPRK33(); dt, negative_tolerance=0.0)
    solution = simulate(problem; solver, saveat=times)
    @show observe(PopulationMoment(:stem, 0), solution)
    @show bulk(problem, last(solution.u), :differentiated_cells)
    solution
end
"""
    elseif name === :nanoparticles
        imports *= "# DLVO quadrature and mechanism accounting are unchanged numerical helpers.\n" *
            "include(joinpath(pkgdir(PopulationBalanceModels), \"examples\", \"papers\", \"cej2025_lnp.jl\"))\n" *
            "using .CEJ2025LNP: kB, mechanism_rates\nconst material = CEJ2025LNP\n"
        push!(pieces, pick("papers/cej2025_lnp.jl", "model",
            (:critical_length, :nucleation_rate, :brownian_kernel, :aqueous_fraction, :build)))
        push!(pieces, replace(pick("shin_lnp_reproduction.jl", "initial_conditions", (:figure4_problem,)),
            "CEJ2025LNP.build" => "build"))
        push!(pieces, pick("shin_lnp_long_time.jl", "reconstruction",
            (:reconstructed_solubility, :make_problem)))
        raw"""
function tutorial_problem(; bins=80)
    # Explicit reconstruction assumptions are in make_problem above.
    # This startup check does not reproduce the historical full-horizon figures.
    remake(make_problem(; bins, FRR=3.0); tspan=(0.0, 1e-5))
end

function main(; bins=80, times=[0.0,1e-5])
    problem = tutorial_problem(; bins)
    solver = ForwardSolver(SSPRK33(); dt=1e-8)
    solution = simulate(problem; solver, saveat=times)
    @show bulk(problem, last(solution.u), :lipid)
    solution
end
"""
    elseif name === :transport
        push!(pieces, pick("inguva_braatz_case2.jl", "model",
            (:FINAL_TIME, :NORMAL, :parameters, :growth_rate, :initial_density,
             :foot, :axis_integrals, :reference, :problem)))
        raw"""
function tutorial_problem(; bins=50)
    # Explicit joint number density on [0,2]²; no inferred independence/normalization.
    initial = InitialDensity((xy,p,t) -> initial_density(xy[1],xy[2],p);
        support=((0.0,2.0),(0.0,2.0)))
    problem(bins; scheme=WENO5(epsilon=1e-12), p=parameters(), initial)
end

function main(; bins=50, times=[1.0])
    pb = tutorial_problem(; bins)
    dt = suggest_dt(pb.operator, pb.u0, pb.p, 0.0; cfl=0.4)
    # Keep signed undershoots in this diagnostic comparison; no clipping.
    solver = ForwardSolver(SSPRK33(); dt, check_nonnegative=false)
    solution = simulate(pb; solver, saveat=times)
    @show observe(MomentObservable((0,0)), solution)
    solution
end
"""
    elseif name === :estimating_growth
        imports *= "using Optim, Random\n" *
            "using Distributions: Normal, LogNormal, Uniform, pdf, cdf, ccdf, quantile, truncated\n"
        push!(pieces, pick("estimate_size_dependent_growth.jl", "inference",
            (:TRUTH, :TIMES, :SIZES, :CENTER, :WIDTH, :INITIAL_NUMBER, :UNIFORM_BOUNDS, :LAW,
             :initial_density, :initial_profile, :shift, :exact_density, :initial_integral, :exact_counts,
             :synthetic_measurements, :analytical_solver)))
        raw"""
function tutorial_problem(; prior=:lognormal)
    data = synthetic_measurements(seed=2022)
    grid = characteristic_grid(LAW, (0.0,1.4); dt=0.005, p=TRUTH)
    model = PBMModel(growth=Growth(LAW))
    operator = compile(model, grid; evolution=Inguva2022(), p=TRUTH,
        bc=SizeBC(OpenBoundary(),OpenBoundary()))
    base = PBMProblem(operator, initial_profile(), (0.0,1.0); p=TRUTH)
    observable = FunctionObservable((N,op,p,t) -> exact_density.(data.sizes,t,Ref(p)))
    measurement = Measurement(data.times, data.values, observable;
        likelihood=TruncatedGaussianLikelihood(sigma=data.noise), source=:density_measurements)
    # Closed-form forward values isolate estimation from transport error.
    solver = analytical_solver # Deliberate exact forward reference, not an ODE wrapper.
    experiment = InferenceExperiment(:growth, base, measurement; solver)
    distributions = prior === :lognormal ?
        (LogNormal(log(0.4),0.4), LogNormal(log(0.25),0.6)) :
        (Uniform(UNIFORM_BOUNDS[1]...), Uniform(UNIFORM_BOUNDS[2]...))
    parameters = [ParameterSpec(name; initial, prior=d, transform=LogTransform())
        for (name,initial,d) in zip((:g0,:g1),(0.40,0.23),distributions)]
    EstimationProblem(parameters, (experiment,))
end

function main()
    problem = tutorial_problem()
    fitted = fit(problem; nstarts=3, seed=22, maxiters=500)
    fitted.converged || error("MAP failed; inspect the fit diagnostics")
    @show fitted.parameters.shared fitted.convention
    @show posterior_components(problem, fitted.minimizer)
    predicted = predictions(problem, fitted.minimizer) # Explicit forward work.
    (; estimation=problem, fitted, predicted)
end
"""
    else # milling_uncertainty
        imports *= "using LinearAlgebra\nimport Distributions\n"
        imports *= "# Prepared daughter projection, outlet-CDF integration and checked BDF adapter.\n" *
            "include(joinpath(pkgdir(PopulationBalanceModels), \"examples\", \"support\", \"milling_model.jl\"))\n" *
            "using .MillingModel: MillingHazard, MillingFixedPivot, daughter_projection\n" *
            "include(joinpath(pkgdir(PopulationBalanceModels), \"examples\", \"support\", \"milling_solver.jl\"))\n"
        push!(pieces, pick("support/milling_model.jl", "model",
            (:PARAMETER_NAMES, :MEAN, :SD, :parameters, :admissible, :speed, :diameter,
             :normal_interval, :grid, :retention, :net_production, :outlet, :build)))
        raw"""
physical(z) = NamedTuple{PARAMETER_NAMES}(Tuple(
    getproperty(MEAN,k) + getproperty(SD,k)*getproperty(z,k) for k in PARAMETER_NAMES))

function tutorial_problem(; bins=40)
    times = [1.0,5.0,29.9,30.0,31.0,60.0,200.0]
    inputs = UncertainInputs((delta=Distributions.Normal(), alpha=Distributions.Normal(),
        gamma=Distributions.Normal(), sigma=Distributions.Normal());
        map=physical, valid=admissible, coordinates=:standard_normal,
        metadata=(joint_law=:independent_gaussian,))
    base = build(; bins)
    # sigma changes the daughter projection: rebuild it for each realization.
    setup = (theta,base) -> build(theta; bins)
    # Instantaneous OUTLET volume median, not the resident number median.
    d50 = SystemObservable((state,op,p,t) -> MillingModel.outlet_median(p,state.populations.mill,t))
    # Retain the constrained material-coordinate reduction; generic external
    # ForwardSolver has no reduced-state or CVODE-constraint declaration.
    solver = MillingSolver.solve
    UQProblem(base, inputs; setup, solver, times,
        outputs=(d50=d50,), output_coordinates=(d50=(units=:micrometre,
            weighting=:instantaneous_outlet_volume, interpolation=:linear_diameter_CDF),))
end

function main(; bins=40, mc_samples=16, validation_samples=8, training_samples=12,
              surrogate_draws=10_000)
    uq = tutorial_problem(; bins)
    solver = uq.solver
    nominal = solver(uq.base, uq.times) # Same callable as the uncertainty calculation.
    @show observe(uq.outputs.d50, nominal; index=findfirst(==(31.0),uq.times))
    # Demonstration budgets, not the retained 12,000-realization production run.
    mc = propagate(uq, MonteCarlo(mc_samples; seed=2018255); max_solves=mc_samples)
    validation = propagate(uq, MonteCarlo(validation_samples; seed=2018256); max_solves=validation_samples)
    box = hcat(fill(-2.0,4), fill(2.0,4)) # Training design, not a truncated input law.
    design = pce_design(uq.inputs; n=training_samples, seed=2018257, bounds=box)
    pce = fit_pce(uq, PolynomialChaos(1); design, validation, max_solves=training_samples)
    @show uq_summary(mc; covariance=false).mean.d50
    @show pce.validation.rmse pce.validation.passed
    if pce.validation.passed
        cheap = evaluate_pce(pce, MonteCarlo(surrogate_draws; seed=2018259))
        @show uq_summary(cheap; covariance=false).quantiles.d50
    else
        @warn "Refine the PCE degree/design before using its uncertainty summaries"
    end
    (; nominal, mc, pce)
end
"""
    end
    header = "# Minimal $(replace(string(name),'_'=>' ')) tutorial.\n" *
        "# Physical definitions and one calculation; numerical helpers are linked on the case page.\n" *
        "# No benchmark exports or retained files are written. See the case-study page.\n\n"
    entry = "\nif abspath(PROGRAM_FILE) == @__FILE__\n    main()\nend\n"
    header * imports * "\n" * join(pieces, "\n") * "\n" * body * "\n" * plotting_code(name) * entry
end

# These functions operate on existing numerical results. Importing PythonPlot is
# an explicit optional step, never a dependency of main() or tutorial inclusion.
function plotting_code(name)
    if name===:constant_growth
        raw"""
# Optional: import PythonPlot; plot_results(solution). No solve/export occurs here.
plot_results(solution) = plot_distribution(solution; times=[last(solution.t)],
    xlabel="Size coordinate", ylabel="Number density")
"""
    elseif name===:crystallization
        raw"""
# Optional after import PythonPlot; reuse the returned solution.
function plot_results(solution)
    distribution = plot_distribution(solution; population=:crystals,
        xlabel="Length (micrometres)", ylabel="Number density")
    concentration = plot_observable(solution, BulkObservable(:c);
        xlabel="Time (s)", ylabel="Solute mass / water mass")
    (; distribution, concentration)
end
"""
    elseif name===:moment_crystallization
        raw"""
# Optional after import PythonPlot. Moment states do not imply a unique density.
plot_results(solution) = plot_observable(solution, MomentObservable(0);
    xlabel="Time (s)", ylabel="Total number")
"""
    elseif name===:continuous_precipitation
        raw"""
# Optional after import PythonPlot. Actual nonuniform size edges are retained.
function plot_results(solution)
    field = plot_field(solution; population=:crystals, orientation=:axial_x,
        xlabel="Axial position (m)", ylabel="Size (m)", colorbar_label="Scaled number density")
    outlet = plot_distribution(solution; population=:crystals,
        cell=length(solution.problem.operator.domain.grid), times=[last(solution.t)],
        xlabel="Size (m)", ylabel="Scaled number density")
    (; field, outlet)
end
"""
    elseif name===:protein_crystallization
        raw"""
# Optional after import PythonPlot. These MAP plots require no MCMC or bands.
function plot_results(result)
    concentration = plot_observable(result.solution, BulkObservable(:c);
        xlabel="Time (min)", ylabel="Concentration (g/L)")
    parameters = plot_parameters(result.estimation, result.fitted; coordinates=:physical)
    (; concentration, parameters)
end
"""
    elseif name===:stem_cells
        raw"""
# Optional after import PythonPlot; synthetic model time is not hours.
function plot_results(solution)
    distribution = plot_distribution(solution; population=:stem,
        xlabel="Cargo (arbitrary units)", ylabel="Cell number density")
    number = plot_observable(solution, PopulationMoment(:stem,0);
        xlabel="Model time", ylabel="Stem-cell concentration")
    (; distribution, number)
end
"""
    elseif name===:nanoparticles
        raw"""
# Optional after import PythonPlot; the short startup only, not a full-horizon claim.
plot_results(solution) = plot_observable(solution, BulkObservable(:lipid);
    xlabel="Time (s)", ylabel="Dissolved lipid concentration")
"""
    elseif name===:transport
        raw"""
# Optional after import PythonPlot. No implicit tensor marginal is selected.
plot_results(solution) = plot_field(solution; orientation=:internal_x,
    xlabel="First internal coordinate", ylabel="Second internal coordinate",
    colorbar_label="Joint number density")
"""
    elseif name===:estimating_growth
        raw"""
# Optional after import PythonPlot; predictions were computed explicitly in main.
function plot_results(result)
    fit_figure = plot_fit(result.estimation, result.predicted; abscissa=SIZES,
        xlabel="Size coordinate", ylabel="Number density")
    parameters = plot_parameters(result.estimation, result.fitted; coordinates=:physical)
    (; fit_figure, parameters)
end
"""
    else
        raw"""
# Optional after import PythonPlot. MC output quantiles are not errors on the mean.
function plot_results(result)
    mc = plot_uq(result.mc; output=:d50,
        xlabel="Time (s)", ylabel="Outlet volume median (micrometres)")
    # PCE analytical moments do not define quantiles; label one output SD explicitly.
    pce = plot_uq(result.pce; output=:d50, band=:sd,
        xlabel="Time (s)", ylabel="Outlet volume median (micrometres)")
    (; mc, pce)
end
"""
    end
end

function write_scripts(destination; root=ROOT)
    mkpath(destination)
    for name in CASES
        write(joinpath(destination, string(name)*".jl"), script(name; root))
    end
    destination
end

end
