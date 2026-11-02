# Minimal moment crystallization tutorial.
# Physical definitions and one calculation; numerical helpers are linked on the case page.
# No benchmark exports or retained files are written. See the case-study page.

using PopulationBalanceModels
using SpecialFunctions: gamma

# From examples/qamar_moment_benchmarks.jl (geometry).
const volume_map=AdditiveCoordinate(x->x^3,cbrt)
const daughter_law=UniformVolumeDaughters()

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

# Optional after import PythonPlot. Moment states do not imply a unique density.
plot_results(solution) = plot_observable(solution, MomentObservable(0);
    xlabel="Time (s)", ylabel="Total number")

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
