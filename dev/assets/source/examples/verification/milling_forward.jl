"""Compare the shared milling model with Barrasso (2013), Figure 7.

Run from a checkout using examples/environments/milling. Including this module
only defines the calculation; it never runs the benchmark or a UQ ensemble.
"""
module MillingForward
using PopulationBalanceModels, LinearAlgebra, Statistics, SHA, TOML, Printf
import JSON3, Pkg

include("../milling_uncertainty.jl")
const MU = MillingUncertainty
const MM = MU.MillingModel
const MS = MU.MillingSolver
const ROOT = normpath(joinpath(@__DIR__, "../.."))
const CONFIG = joinpath(ROOT, "examples/data/milling_uncertainty/barrasso_figure7.toml")
const OUTPUT = joinpath(ROOT, "docs/src/assets/milling_uncertainty/forward")

function reference_data()
    config = TOML.parsefile(CONFIG)
    path = joinpath(dirname(CONFIG), config["reference_file"])
    bytes2hex(sha256(read(path))) == config["reference_sha256"] ||
        error("Barrasso Figure 7 readings changed; check their provenance before running")
    [begin
        quantity, rpm, t, value, px, py = split(line, ',')
        (; quantity, rpm=parse(Float64, rpm), time=parse(Float64, t),
           value=parse(Float64, value), pdf_x=parse(Float64, px), pdf_y=parse(Float64, py))
    end for line in readlines(path)[2:end]]
end

"""Fixed-endpoint refinement of Eq. (13), retaining arithmetic pivot centres."""
function power_grid(bins)
    config = TOML.parsefile(CONFIG)["calculation"]
    x = config["first_pivot_m3"] .* range(1., config["last_grid_index"]; length=bins).^config["grid_power"]
    edges = zeros(bins + 1)
    edges[1] = first(x) / 2
    for i in eachindex(x)
        edges[i+1] = 2x[i] - edges[i]
    end
    Grid1D(edges; coordinate=:volume)
end

"""Prepare positive daughter hats on the power grid, without geometric-offset reuse."""
function daughters(grid, p; order=24)
    x = grid.centers
    rule = PopulationBalanceModels._gauss_legendre(GaussLegendre(order))
    eta = zeros(length(x), length(x))
    for j in eachindex(x), i in 1:j
        if i > 1
            eta[i,j] += MM.hat_integral(log(x[i-1]/x[j]), log(x[i]/x[j]),
                p.sigma, p.daughter_ratio, true, rule)
        end
        if i < j
            eta[i,j] += MM.hat_integral(log(x[i]/x[j]), log(x[i+1]/x[j]),
                p.sigma, p.daughter_ratio, false, rule)
        end
    end
    norm = MM.relative_moment(1, 1., p.sigma, p.daughter_ratio)
    low0 = [MM.relative_moment(0, x[1]/y, p.sigma, p.daughter_ratio)/norm for y in x]
    low1 = [y*MM.relative_moment(1, x[1]/y, p.sigma, p.daughter_ratio)/norm for y in x]
    defect = maximum(abs.(vec(x' * eta) + low1 - x) ./ x)
    defect < 2e-12 || throw(NumericalFailure("unresolved daughter volume; increase quadrature order ($defect)"))
    MM.MillingDaughters(copy(x), eta, low0, low1, p.sigma), defect
end

function problem(; bins=100, rpm=3350., order=24)
    config = TOML.parsefile(CONFIG)["calculation"]
    base = MM.build(; tf=config["final_time_s"])
    grid = power_grid(bins)
    x = grid.centers
    daughter, defect = daughters(grid, base.p; order)
    p = base.p
    fractions = exp.(-0.5 .* ((log.(x) .- log(p.feed_median)) ./ p.feed_log_sd).^2)
    feed = (p.mass_feed / (p.rho*p.number_scale)) .* (fractions/sum(fractions)) ./ x
    selection = p.alpha .* (x ./ p.xref).^p.gamma
    Q = (daughter.eta - I) * Diagonal(selection)
    p = merge(p, (; pivots=x, edges=grid.edges, feed, selection, Q,
        low0=daughter.underflow_number .* selection,
        low1=daughter.underflow_volume .* selection, change_time=Inf, speed_before=rpm))
    p = merge(p, (; retained=MM.retention(grid, p)))
    terms = base.operator.system.populations.mill.model.terms
    model = PBMModel(feed=terms[1],
        breakage=Breakage(terms[2].rate, daughter; overflow=:discard), classification=terms[3])
    op = compile(model, grid; methods=TermMethods(by_term=(breakage=MM.MillingFixedPivot(),)))
    system = PBMSystem((mill=op,); bulk_rhs=base.operator.system.bulk_rhs,
        nonnegative_bulk=(:discharged_mass, :fines_mass))
    initial = config["initial_feed_duration_s"] .* feed
    pb = PBMProblem(system, (mill=initial,), base.tspan;
        bulk0=(discharged_mass=0., fines_mass=0.), p)
    pb, defect
end

"""Integrate the material CDF exactly for constant number density in each volume bin.
This is an observation reconstruction, not a change to daughter deposition.
"""
function finite_bin_median(p, N, t)
    q = MM.outlet(p, N, t)
    weights = p.pivots .* q.resolved
    all(>=(0), weights) && q.underflow_volume >= 0 ||
        throw(NumericalFailure("negative instantaneous outlet contribution"))
    cumulative = q.underflow_volume
    target = (sum(weights) + cumulative) / 2
    target > 0 || throw(NumericalFailure("empty outlet has no median"))
    # The existing exact fines observation handles a sub-grid median.
    target < cumulative && return MM.outlet_median(p, N, t)
    for i in eachindex(weights)
        next = cumulative + weights[i]
        if next >= target && weights[i] > 0
            fraction = (target-cumulative)/weights[i]
            v = sqrt(p.edges[i]^2 + fraction*(p.edges[i+1]^2-p.edges[i]^2))
            return 1e6MM.diameter(v)
        end
        cumulative = next
    end
    throw(NumericalFailure("outlet median is outside the represented bins"))
end

"""Independent truncated-lognormal interval moments, rather than hat quadrature."""
function analytic_daughters(p)
    x = p.pivots
    mu = -log(p.daughter_ratio)
    sigma = p.sigma
    normal = MU.Distributions.Normal()
    norm = exp(mu + sigma^2/2) * MU.Distributions.cdf(normal, (-mu-sigma^2)/sigma)
    function moment(k, a, b, parent)
        dist = MU.Distributions.LogNormal(mu + k*sigma^2, sigma)
        lo, hi = a/parent, b/parent
        probability = lo > exp(mu + k*sigma^2) ?
            MU.Distributions.ccdf(dist, lo) - MU.Distributions.ccdf(dist, hi) :
            MU.Distributions.cdf(dist, hi) - MU.Distributions.cdf(dist, lo)
        parent^k * exp(k*mu + k^2*sigma^2/2) * probability / norm
    end
    eta = zeros(length(x), length(x))
    for j in eachindex(x), i in 1:j-1
        b0 = moment(0, x[i], x[i+1], x[j])
        b1 = moment(1, x[i], x[i+1], x[j])
        eta[i,j] += (x[i+1]*b0-b1)/(x[i+1]-x[i])
        eta[i+1,j] += (b1-x[i]*b0)/(x[i+1]-x[i])
    end
    low0 = [moment(0, 0., x[1], parent) for parent in x]
    low1 = [moment(1, 0., x[1], parent) for parent in x]
    (; eta, low0, low1)
end

function exponential_check(pb, sol)
    p = pb.p
    x = p.pivots
    exact = analytic_daughters(p)
    Q = (exact.eta-I) * Diagonal(p.selection)
    independent = merge(p, (; Q, low0=exact.low0 .* p.selection, low1=exact.low1 .* p.selection))
    feedmass = p.rho*p.number_scale .* x .* p.feed
    A = p.retained .* ((p.speed_before .* Q .* x) ./ x')
    active = findall(>(0), p.retained)
    augmented = [A[active,active] (p.retained .* feedmass)[active]; zeros(1, length(active)+1)]
    initial = p.rho*p.number_scale .* x .* population(pb, pb.u0, :mill)
    d50_errors = Float64[]
    holdup_errors = Float64[]
    for t in (1., 5., 10., 30., 60., 100., 180.)
        index = findfirst(==(t), sol.t)
        mass = zeros(length(x))
        mass[active] = (exp(t*augmented) * vcat(initial[active], 1.))[1:end-1]
        N = population(pb, sol.u[index], :mill)
        push!(d50_errors, MM.outlet_median(independent, mass ./ (p.rho*p.number_scale .* x), t) -
            MM.outlet_median(p, N, t))
        push!(holdup_errors, 1e3*(sum(mass)-p.rho*p.number_scale*dot(x, N)))
    end
    (; rpm=p.speed_before, maximum_d50_difference_um=maximum(abs, d50_errors),
       maximum_holdup_difference_g=maximum(abs, holdup_errors),
       relative_daughter_material_defect=maximum(abs.(vec(x' * exact.eta)+exact.low1-x)./x))
end

function run(; output=OUTPUT)
    config = TOML.parsefile(CONFIG)
    settings = config["calculation"]
    reference = reference_data()
    times = sort!(unique(vcat(.001, collect(1.:settings["final_time_s"]),
        [r.time for r in reference if r.time > .01])))
    source_hashes = copy(MU.source_identity(settings).hashes)
    for path in (@__FILE__, CONFIG, joinpath(dirname(CONFIG), config["reference_file"]))
        source_hashes[relpath(path, ROOT)] = bytes2hex(sha256(read(path)))
    end
    status = read(`git -C $ROOT status --short`, String)
    revision = strip(read(`git -C $ROOT rev-parse HEAD`, String))
    mkpath(output)
    # One separate compilation warmup, outside the measured calculation.
    warmup, _ = problem()
    MS.solve(warmup, [1., 5.])
    rows, points, metrics = Tuple[], Tuple[], Tuple[]
    checks, time_checks, exponential_checks = NamedTuple[], NamedTuple[], NamedTuple[]
    for bins in settings["bins"], rpm in settings["speeds_rpm"]
        println("Barrasso forward comparison: $bins pivots, $rpm RPM"); flush(stdout)
        started = time_ns()
        pb, daughter_defect = problem(; bins, rpm, order=settings["quadrature_order"])
        setup_seconds = (time_ns()-started)/1e9
        started = time_ns()
        sol = MS.solve(pb, times; reltol=settings["reltol"], abstol=settings["abstol_kg"])
        solve_seconds = (time_ns()-started)/1e9
        p = pb.p
        m0 = p.rho*p.number_scale*dot(p.pivots, population(pb, pb.u0, :mill))
        material = [p.rho*p.number_scale*dot(p.pivots, population(pb, u, :mill)) for u in sol.u]
        defects = [material[i]+bulk(pb, u, :discharged_mass)-m0-p.mass_feed*t
            for (i, (t, u)) in enumerate(zip(times, sol.u))]
        for (i, t) in enumerate(times)
            N = population(pb, sol.u[i], :mill)
            push!(rows, (bins, rpm, t, MM.outlet_median(p, N, t), finite_bin_median(p, N, t),
                1e3material[i], defects[i], minimum(N)))
        end
        for quantity in ("d50_um", "holdup_g"), statistic in
            (quantity == "d50_um" ? ("pivot_CDF", "finite_bin_CDF") : ("inventory",))
            observations = filter(r -> r.rpm == rpm && r.quantity == quantity && r.time > .01, reference)
            differences = Float64[]
            for r in observations
                i = findfirst(==(r.time), times)
                N = population(pb, sol.u[i], :mill)
                predicted = quantity == "holdup_g" ? 1e3material[i] :
                    statistic == "pivot_CDF" ? MM.outlet_median(p, N, r.time) : finite_bin_median(p, N, r.time)
                push!(differences, predicted-r.value)
                push!(points, (bins, rpm, quantity, statistic, r.time, r.value, predicted, predicted-r.value))
            end
            push!(metrics, (bins, rpm, quantity, statistic, length(differences),
                sqrt(mean(abs2, differences)), maximum(abs, differences),
                100mean(abs(differences[i])/observations[i].value for i in eachindex(differences))))
        end
        push!(checks, (; bins, rpm, setup_seconds, solve_seconds, steps=sol.steps,
            initial_mass_kg=m0, maximum_material_error_kg=maximum(abs, defects),
            minimum_inventory=minimum(minimum.(sol.u)), daughter_defect))
        if bins == 100
            tight = MS.solve(pb, times; reltol=settings["check_reltol"], abstol=settings["check_abstol_kg"])
            push!(time_checks, (; rpm, maximum_d50_change_um=maximum(abs(
                MM.outlet_median(p, population(pb, sol.u[i], :mill), t)-
                MM.outlet_median(p, population(pb, tight.u[i], :mill), t)) for (i,t) in enumerate(times))))
            push!(exponential_checks, exponential_check(pb, sol))
        end
    end
    maximum(c.maximum_material_error_kg for c in checks) < 1e-9 || error("milling material balance failed")
    minimum(c.minimum_inventory for c in checks) >= 0 || error("negative saved milling population")
    maximum(c.maximum_d50_change_um for c in time_checks) < 1e-3 || error("unresolved forward time integration")
    maximum(c.maximum_d50_difference_um for c in exponential_checks) < 1e-3 || error("BDF/exponential check failed")
    MU.write_csv(joinpath(output, "trajectories.csv"),
        ["bins", "rpm", "time_s", "pivot_CDF_d50_um", "finite_bin_d50_um", "holdup_g", "material_defect_kg", "minimum_inventory"], rows)
    MU.write_csv(joinpath(output, "reference_comparison.csv"),
        ["bins", "rpm", "quantity", "statistic", "time_s", "reference", "prediction", "signed_difference"], points)
    MU.write_csv(joinpath(output, "errors.csv"),
        ["bins", "rpm", "quantity", "statistic", "points", "rmse", "maximum_absolute_difference", "mean_absolute_relative_percent"], metrics)
    MU.write_json(joinpath(output, "checks.json"), (; solves=checks, time_refinement=time_checks, matrix_exponential=exponential_checks))
    open(joinpath(output, "comparison.md"), "w") do io
        println(io, "| Speed (RPM) | Pivots / outlet reconstruction | Diameter RMS difference (µm) | Hold-up RMS difference (g) |")
        println(io, "|--:|:--|--:|--:|")
        for rpm in settings["speeds_rpm"], (bins, statistic) in ((100, "pivot_CDF"), (100, "finite_bin_CDF"), (800, "finite_bin_CDF"))
            d = only(filter(r -> r[1]==bins && r[2]==rpm && r[3]=="d50_um" && r[4]==statistic, metrics))
            h = only(filter(r -> r[1]==bins && r[2]==rpm && r[3]=="holdup_g", metrics))
            label = statistic == "pivot_CDF" ? "pivot CDF" : "finite-bin CDF"
            @printf(io, "| %.0f | %d / %s | %.2f | %.3f |\n", rpm, bins, label, d[6], h[6])
        end
    end
    numerical_files = Dict(name => bytes2hex(sha256(read(joinpath(output, name)))) for name in
        ("trajectories.csv", "reference_comparison.csv", "errors.csv", "checks.json", "comparison.md"))
    versions = Dict(info.name => string(info.version) for info in values(Pkg.dependencies()) if
        info.name in ("PopulationBalanceModels", "Distributions", "Sundials", "SciMLBase", "JSON3"))
    MU.write_json(joinpath(output, "provenance.json"), (; complete=true, mode="forward_comparison",
        revision, working_tree=status, source_hashes, numerical_files, configuration=config,
        julia=string(VERSION), julia_threads=Threads.nthreads(), blas_threads=BLAS.get_num_threads(),
        cpu=Sys.cpu_info()[1].model, package_versions=versions, warmup_solves=1,
        full_solves=length(checks)+length(time_checks), matrix_exponential_checks=length(exponential_checks),
        timing_scope="One measured checked BDF solve after a separate compilation warmup; export and plotting excluded",
        result="Partial reproduction: close hold-up; diameter trajectory differs. No parameter fitting."))
    println("Wrote independent forward comparison to $output")
end
end # module

if abspath(PROGRAM_FILE) == @__FILE__
    args = copy(ARGS)
    output = MillingForward.OUTPUT
    if !isempty(args)
        length(args) == 2 && args[1] == "--output" || error("Usage: milling_forward.jl [--output DIRECTORY]")
        output = abspath(args[2])
    end
    MillingForward.run(; output)
end
