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

function script(name::Symbol; root=ROOT)
    name in CASES || throw(ArgumentError("Unknown case study: $name"))
    take(path, section) = source("examples/" * path, section; root)
    imports = "using PopulationBalanceModels\n"
    pieces = String[]
    body = if name === :constant_growth
        imports *= "using Distributions: Normal, cdf, ccdf\n"
        append!(pieces, [take("constant_growth_methods.jl", section)
            for section in ("velocity", "gaussian", "pairings")])
        raw"""
function tutorial_problem(; bins=200)
    grid = Grid1D(range(0.0, 1.0; length=bins+1); coordinate=:length)
    model = PBMModel(growth=Growth(AffineGrowth(growth, 0.0)))
    boundary = SizeBC(OpenBoundary(), OpenBoundary())
    (; grid, model, boundary, initial=gaussian_counts(grid, 0.0),
       reference=gaussian_counts(grid, 4.0))
end

function main()
    case = tutorial_problem()
    for (name, time_name, scheme, time_method) in combinations()
        operator = scheme === nothing ?
            compile(case.model, case.grid; evolution=time_method, bc=case.boundary) :
            compile(case.model, case.grid;
                methods=TermMethods(growth=scheme), bc=case.boundary)
        problem = PBMProblem(operator, case.initial, (0.0, 4.0))
        CFL = time_method isa Inguva2022 ? 1.0 : 0.5
        dt = CFL * first(case.grid.widths) / growth
        # Signed diagnostic comparison: expose undershoots rather than clipping.
        solution = simulate(problem; method=time_method, dt,
            saveat=[4.0], check_nonnegative=false)
        error = (vec(last(solution.u)) .- case.reference) ./ case.grid.widths
        L1 = sum(case.grid.widths .* abs.(error))
        RMSE = sqrt(sum(case.grid.widths .* error.^2) / sum(case.grid.widths))
        @show name time_name CFL L1 RMSE minimum(density(case.grid, last(solution.u)))
    end
end
"""
    elseif name === :crystallization
        append!(pieces, [take("papers/gunawan2004_case2.jl", section)
            for section in ("physics", "assembly")])
        raw"""
function tutorial_problem()
    p = parameters() # µm, seconds, g solute/g water; rho is g/µm³.
    edges = sort!(unique(vcat(collect(0.0:2.0:1600.0), 180.5, 210.5)))
    grid = Grid1D(edges; coordinate=:length)
    c0 = solubility(temperature(0.0))
    # Explicit nonnegative benchmark IC, not the negative printed polynomial.
    build(; grid, initial_density=benchmark_initial_density, c0, p,
        scheme=MUSCL(), tspan=(0.0, 10.0))
end

function main()
    problem = tutorial_problem()
    solution = simulate(problem; method=SSPRK33(), dt=0.05, saveat=[0.0, 5.0, 10.0])
    grid = problem.operator.system.populations.crystals.grid
    total(u) = bulk(problem, u, :c) +
        problem.p.rho * moment(grid, population(problem, u, :crystals), 3)
    @show bulk(problem, last(solution.u), :c)
    @show maximum(abs(total(u)-total(problem.u0)) for u in solution.u)
    solution
end
"""
    elseif name === :moment_crystallization
        imports *= "using SpecialFunctions: gamma\n"
        append!(pieces, [take("qamar_moment_benchmarks.jl", section)
            for section in ("geometry", "problem4")])
        raw"""
function tutorial_problem(; method=QMOM())
    # `nodes` chooses the basis when a family omits its node count.
    # Each family's required orders and DQMOM initial-state conversion are above.
    problem4(; nodes=3, method, tf=50.0)
end

function reported_moment(op, u, k)
    k in op.grid.orders && return moment_value(op.closure, op.grid, u, k)
    q = quadrature(op.grid, u, op.closure)
    reconstructed_moment(op.closure, op.grid, u, q, k)
end

function main(; method=QMOM())
    problem = tutorial_problem(; method)
    @show moment_plan(problem.operator)
    solution = simulate(problem; method=SSPRK33(), dt=0.02, saveat=0.0:1.0:50.0)
    op = problem.operator
    # Lognormal stores only M0:M2; its M3 is reconstructed, not conserved state.
    M(k,u) = reported_moment(op,u,k)
    error = [abs(M(1,u)-exact4(t)[2])/abs(exact4(t)[2])
        for (t,u) in zip(solution.t, solution.u)]
    material_kind = 3 in op.grid.orders ? "tracked" : "reconstructed"
    @show maximum(error) material_kind M(3,problem.u0)/0.001
    @show maximum(abs(M(3,u)/0.001-1) for u in solution.u)
    solution
end
"""
    elseif name === :continuous_precipitation
        append!(pieces, [take("continuous_precipitation.jl", section)
            for section in ("model", "controller")])
        raw"""
function tutorial_problem(; nx=50, nz=50)
    # Log size bins on [1,900] µm, uniform physical reactor cells.
    # Size and axial reconstruction are deliberately selected separately.
    build(; nx, nz, mesh=:log, scheme=WENO5(), axial_scheme=MUSCL(), tf=5.0)
end

function main()
    problem = tutorial_problem()
    # The controller above bounds the next explicit RK4 step; it is not an
    # adaptive-error solver or an unconditional positivity guarantee for WENO.
    solution = advance(problem, [0.0, 5.0]; method=RK4())
    grid = first(problem.operator.local_operators).system.populations.crystals.grid
    physical_density = 1e12 .* density(grid, population(problem, last(solution.u), :crystals))
    @show extrema(bulk(problem, last(solution.u), :c)) extrema(physical_density)
    solution
end
"""
    elseif name === :protein_crystallization
        imports *= "using Optim, LinearAlgebra, DelimitedFiles\nusing Distributions: Uniform, LogNormal, Exponential\n"
        push!(pieces, "# Original supplied CSV readings; no digitization or fitting here.\n" *
            "const DATA = joinpath(pkgdir(PopulationBalanceModels), \"examples\", \"data\",\n" *
            "    \"protein_crystallization\", \"tang2018_isothermal\")\n")
        append!(pieces, [take("isothermal_protein_crystallization.jl", section)
            for section in ("constants", "data", "model", "controller", "priors",
                "cdf_covariance", "inference")])
        raw"""
function tutorial_problem(; model=:cnt, bins=200)
    # 11 training concentrations + CDFs at 74,180,420 min; no validation data
    # or derived quantiles in the likelihood. Fixed uncertainty, no fitted noise.
    estimation_problem(; data=load_data(), model, bins)
end

function main(; model=:cnt)
    estimation = tutorial_problem(; model)
    fitted = fit_model(estimation) # Three calibration-only starts; no MCMC.
    fitted.converged || error("MAP failed; inspect the fit diagnostics")
    theta = physical_parameters(estimation, fitted.minimizer, :batch)
    # Freeze calibration before predicting the 660-min held-out distribution.
    p = merge(parameters(), theta)
    solution = forward(p; model, times=[0.0, 74.0, 180.0, 420.0, 660.0])
    @show theta [bulk(solution.problem,u,:c) for u in solution.u]
    op = solution.problem.operator.system.populations.crystals
    N = population(solution.problem, last(solution.u), :crystals)
    @show conditional_quantiles(op.grid, N, [0.1,0.5,0.9]; window=WINDOW)
    (; fitted, solution)
end
"""
    elseif name === :stem_cells
        imports *= "using LinearAlgebra\nusing Distributions: Normal, cdf, ccdf\n"
        push!(pieces, take("stem_cell_population.jl", "model"))
        raw"""
function tutorial_problem(; differentiation=false, bins=600)
    # A short transient in synthetic model time, not the balanced-growth study.
    build_problem(; bins, tf=20.0, differentiation)
end

function main()
    for differentiation in (false, true)
        problem = tutorial_problem(; differentiation)
        dt = min(0.05, suggest_dt(problem.operator, problem.u0, problem.p, 0.0; cfl=0.35))
        solution = simulate(problem; method=SSPRK33(), dt,
            saveat=[0.0,10.0,20.0], negative_tolerance=0.0)
        summary = state_summary(problem, last(solution.u), last(solution.t))
        @show differentiation summary.stem summary.differentiated
        @show summary.number_balance summary.cargo_balance
    end
end
"""
    elseif name === :nanoparticles
        imports *= "using LinearAlgebra\n"
        push!(pieces, take("papers/cej2025_lnp.jl", "model"))
        push!(pieces, "# Numerical/material functions below refer to the fully shown module.\nconst material = CEJ2025LNP\n")
        push!(pieces, take("shin_lnp_reproduction.jl", "initial_conditions"))
        push!(pieces, take("shin_lnp_long_time.jl", "reconstruction"))
        raw"""
function tutorial_problem(; bins=80)
    # Explicit reconstruction assumptions are in make_problem above.
    # This startup check does not reproduce the historical full-horizon figures.
    remake(make_problem(; bins, FRR=3.0); tspan=(0.0, 1e-5))
end

function main()
    problem = tutorial_problem()
    solution = simulate(problem; method=SSPRK33(), dt=1e-8, saveat=[0.0,1e-5])
    @show bulk(problem, last(solution.u), :lipid)
    solution
end
"""
    elseif name === :transport
        push!(pieces, take("inguva_braatz_case2.jl", "model"))
        raw"""
function tutorial_problem(; bins=50)
    # Functions above specify both affine velocities and exact Gaussian bin ICs.
    problem(bins; scheme=WENO5(epsilon=1e-12), p=parameters())
end

function main()
    pb = tutorial_problem()
    grid = pb.operator.grid
    dt = suggest_dt(pb.operator, pb.u0, pb.p, 0.0; cfl=0.4)
    # Keep signed undershoots in this diagnostic comparison; no clipping.
    solution = simulate(pb; method=SSPRK33(), dt, saveat=[1.0], check_nonnegative=false)
    truth = reference(grid, 1.0, pb.p)
    @show sum(abs, last(solution.u)-truth) minimum(density(grid, last(solution.u)))
    solution
end
"""
    elseif name === :estimating_growth
        imports *= "using Optim, Random, LinearAlgebra\n" *
            "using Distributions: Normal, LogNormal, Uniform, pdf, cdf, ccdf, quantile, truncated\n"
        push!(pieces, take("estimate_size_dependent_growth.jl", "inference"))
        raw"""
function tutorial_problem(; prior=:lognormal)
    # Known continuous density + truncated-Gaussian observations, not a histogram.
    data = synthetic_measurements(; seed=2022)
    estimation_problem(data; prior)
end

function main()
    problem = tutorial_problem()
    fitted = fit(problem; nstarts=3, seed=22, maxiters=500)
    fitted.converged || error("MAP failed; inspect the fit diagnostics")
    @show fitted.parameters.shared fitted.convention
    @show posterior_components(problem, fitted.minimizer)
    predictions(problem, fitted.minimizer)
end
"""
    else # milling_uncertainty
        imports *= "using LinearAlgebra\nimport Distributions\n"
        push!(pieces, take("support/milling_model.jl", "model"))
        push!(pieces, "# Numerical adapter only: segmented CVODE_BDF, mass scaling and stage checks.\n" *
            "# All feed, breakage, screen and outlet laws are defined above.\n" *
            "include(joinpath(pkgdir(PopulationBalanceModels),\n" *
            "    \"examples\", \"support\", \"milling_solver.jl\"))\n")
        push!(pieces, take("milling_uncertainty.jl", "uq"))
        raw"""
function tutorial_problem(; bins=40)
    times = [1.0,5.0,29.9,30.0,31.0,60.0,200.0]
    problem(; bins, times)
end

function main()
    uq = tutorial_problem()
    # Demonstration budgets, not the retained 12,000-realization production run.
    mc = propagate(uq, MonteCarlo(32; seed=2018255); max_solves=32)
    validation = propagate(uq, MonteCarlo(16; seed=2018256); max_solves=16)
    box = hcat(fill(-2.0,4), fill(2.0,4)) # Training design, not a truncated input law.
    @show uq_summary(mc; covariance=false).mean.d50
    for (degree, count, seed) in ((1,12,2018257),(2,32,2018258))
        design = pce_design(uq.inputs; n=count, seed, bounds=box)
        pce = fit_pce(uq, PolynomialChaos(degree);
            design, validation, max_solves=count)
        cheap = evaluate_pce(pce, MonteCarlo(10_000; seed=2018259))
        @show degree pce.validation uq_summary(pce).mean.d50
        @show uq_summary(cheap; covariance=false).quantiles.d50
    end
end
"""
    end
    header = "# Minimal $(replace(string(name),'_'=>' ')) tutorial.\n" *
        "# Generated from maintained scientific source sections and a short run block.\n" *
        "# No benchmark exports or retained files are written. See the case-study page.\n\n"
    entry = "\nif abspath(PROGRAM_FILE) == @__FILE__\n    main()\nend\n"
    header * imports * "\n" * join(pieces, "\n") * "\n" * body * entry
end

function write_scripts(destination; root=ROOT)
    mkpath(destination)
    for name in CASES
        write(joinpath(destination, string(name)*".jl"), script(name; root))
    end
    destination
end

end
