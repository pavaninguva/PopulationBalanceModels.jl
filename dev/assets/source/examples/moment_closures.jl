module MomentClosures
using PopulationBalanceModels

"""Quadratic growth: known lower equations and a reconstructed fourth moment."""
function run_mixed(;a=0.1)
    model = PBMModel(
        growth=Growth(PolynomialRate((0.,0.,(p,t,ctx)->p.a))),
        primary=Nucleation((p,t,ctx)->p.J,0.),
    )
    basis = MomentSet(0:3; coordinate=:volume)
    operator = compile(model,basis; moment_method=QMOM())
    initial = [1.,1.,2.,6.]
    problem = PBMProblem(operator,initial,(0.,0.02); p=(a=a,J=0.01))
    derivative = zero(problem.u0)
    rhs!(derivative,problem.u0,operator,problem.p,0.)
    @assert vec(derivative) ≈ [0.01,2a,12a,60a]
    solution = simulate(problem; method=SSPRK33(),dt=0.001)
    println("Resolved family: ",moment_plan(operator).family)
    for equation in moment_plan(operator).terms.growth
        println("Growth M",equation.order,": ",equation.treatment,
                "; missing moments = ",equation.missing)
    end
    println("Initial moment rates: ",vec(derivative))
    println("Final moments: ",vec(last(solution.u)))
    # The exponential initial density has true M4=24. Two-node QMOM gives 20;
    # the closed lower equations do not make this full trajectory exact.
    (;problem,solution)
end

if abspath(PROGRAM_FILE)==@__FILE__
    run_mixed()
end
end
