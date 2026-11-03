# Aggregation/fragmentation: x is length, additive volume is x^3.
using PopulationBalanceModels
using SpecialFunctions: gamma

volume_map = AdditiveCoordinate(x->x^3,cbrt)
model = PBMModel(
    aggregation=Aggregation(ConstantKernel(1.0);coordinate_map=volume_map),
    breakage=Breakage((x,p,t,ctx)->x^3,UniformVolumeDaughters();coordinate_map=volume_map),
)
basis = MomentSet(0:5;coordinate=:length)
operator = compile(model,basis;moment_method=QMOM()) # Three nodes from six moments.
# Exponential volume distribution: N0=1, total volume=0.001, volume=x^3.
initial = [0.001^(k/3)*gamma(1+k/3) for k in basis.orders]
problem = PBMProblem(operator,initial,(0.0,50.0))

if abspath(PROGRAM_FILE) == @__FILE__
    solution = simulate(problem;method=SSPRK33(),dt=0.02,saveat=0.0:1.0:50.0)
    number = observe(MomentObservable(0),solution)
    material = observe(MomentObservable(3),solution)
    @show number material
    # Optional: import PythonPlot
    # plot_observable(solution,MomentObservable(0);xlabel="Time (s)",ylabel="Total number")
end
# Other families, reconstruction and the nucleation pulse are separate recipes
# in qamar_moment_benchmarks.jl; finite moments do not specify a unique PSD.
