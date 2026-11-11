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

function problem(; bins=MU.SETTINGS["bins"], rpm=3350., order=MU.SETTINGS["daughter_quadrature_order"])
    config = TOML.parsefile(CONFIG)["calculation"]
    pb = MU.forward_problem(;bins,quadrature_order=order,tf=config["final_time_s"],
        initial_feed_duration=config["initial_feed_duration_s"],rpm)
    x = pb.p.pivots
    daughter = pb.operator.system.populations.mill.model.terms[2].daughters
    defect = maximum(abs.(vec(x' * daughter.eta) + daughter.underflow_volume - x) ./ x)
    pb, defect
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
    numerics = MU.numerical_configuration()
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
        pb, daughter_defect = problem(; bins, rpm, order=numerics.daughter_quadrature_order)
        setup_seconds = (time_ns()-started)/1e9
        started = time_ns()
        sol = MS.solve(pb, times; reltol=numerics.reltol, abstol=numerics.abstol_kg)
        solve_seconds = (time_ns()-started)/1e9
        p = pb.p
        m0 = p.rho*p.number_scale*dot(p.pivots, population(pb, pb.u0, :mill))
        material = [p.rho*p.number_scale*dot(p.pivots, population(pb, u, :mill)) for u in sol.u]
        defects = [material[i]+bulk(pb, u, :discharged_mass)-m0-p.mass_feed*t
            for (i, (t, u)) in enumerate(zip(times, sol.u))]
        for (i, t) in enumerate(times)
            N = population(pb, sol.u[i], :mill)
            push!(rows, (bins, rpm, t, MM.outlet_median(p, N, t),
                1e3material[i], defects[i], minimum(N)))
        end
        for quantity in ("d50_um", "holdup_g"), statistic in
            (quantity == "d50_um" ? ("pivot_CDF",) : ("inventory",))
            observations = filter(r -> r.rpm == rpm && r.quantity == quantity && r.time > .01, reference)
            differences = Float64[]
            for r in observations
                i = findfirst(==(r.time), times)
                N = population(pb, sol.u[i], :mill)
                predicted = quantity == "holdup_g" ? 1e3material[i] : MM.outlet_median(p, N, r.time)
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
        if bins == numerics.bins
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
        ["bins", "rpm", "time_s", "pivot_CDF_d50_um", "holdup_g", "material_defect_kg", "minimum_inventory"], rows)
    MU.write_csv(joinpath(output, "reference_comparison.csv"),
        ["bins", "rpm", "quantity", "statistic", "time_s", "reference", "prediction", "signed_difference"], points)
    MU.write_csv(joinpath(output, "errors.csv"),
        ["bins", "rpm", "quantity", "statistic", "points", "rmse", "maximum_absolute_difference", "mean_absolute_relative_percent"], metrics)
    MU.write_json(joinpath(output, "checks.json"), (; solves=checks, time_refinement=time_checks, matrix_exponential=exponential_checks))
    open(joinpath(output, "comparison.md"), "w") do io
        println(io, "| Speed (RPM) | Geometric pivots | Diameter RMS difference (µm) | Hold-up RMS difference (g) |")
        println(io, "|--:|:--|--:|--:|")
        for rpm in settings["speeds_rpm"], bins in (numerics.bins, last(settings["bins"]))
            d = only(filter(r -> r[1]==bins && r[2]==rpm && r[3]=="d50_um" && r[4]=="pivot_CDF", metrics))
            h = only(filter(r -> r[1]==bins && r[2]==rpm && r[3]=="holdup_g", metrics))
            @printf(io, "| %.0f | %d | %.2f | %.3f |\n", rpm, bins, d[6], h[6])
        end
    end
    numerical_files = Dict(name => bytes2hex(sha256(read(joinpath(output, name)))) for name in
        ("trajectories.csv", "reference_comparison.csv", "errors.csv", "checks.json", "comparison.md"))
    versions = Dict(info.name => string(info.version) for info in values(Pkg.dependencies()) if
        info.name in ("PopulationBalanceModels", "Distributions", "Sundials", "SciMLBase", "JSON3"))
    MU.write_json(joinpath(output, "provenance.json"), (; complete=true, mode="forward_comparison",
        revision, working_tree=status, source_hashes, numerical_files, configuration=config,numerics,
        julia=string(VERSION), julia_threads=Threads.nthreads(), blas_threads=BLAS.get_num_threads(),
        cpu=Sys.cpu_info()[1].model, package_versions=versions, warmup_solves=1,
        full_solves=length(checks)+length(time_checks), matrix_exponential_checks=length(exponential_checks),
        timing_scope="One measured checked BDF solve after a separate compilation warmup; export and plotting excluded",
        result="Partial reproduction: close hold-up; diameter trajectory differs. No parameter fitting."))
    println("Wrote shared-model forward comparison to $output")
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
