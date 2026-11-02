"""Public numerical preparation and checked execution; no retained data are written.

Run: julia --project=. examples/prepared_forward.jl
Optional: load SciMLBase and OrdinaryDiffEqTsit5, then call external_example(Tsit5()).
"""
module PreparedForwardExample
using PopulationBalanceModels
using Distributions: Normal, truncated

function problem()
    grid=Grid1D(range(0.,2.;length=41);coordinate=:volume)
    daughters=ContinuousDaughters((r,parent,p,t,ctx)->2.;
        coordinates=:ratio,support=(0.,1.),count=2.)
    model=PBMModel(fragmentation=Breakage((v,p,t,ctx)->p.k,daughters;overflow=:discard))
    operator=compile(model,grid;methods=TermMethods(breakage=FixedPivot()))
    # The probability law is explicitly conditioned on [0,2]. No population
    # measure is silently redistributed by the numerical initialization.
    initial=InitialDistribution(truncated(Normal(.8,.15),0.,2.);total=3.)
    history=OperatingHistory([.5];parameters=((k=.2,),(k=.6,)))
    system=PBMSystem((particles=operator,);
        accounting=(CumulativeAccounting(:particles;properties=(volume=1,)),))
    PBMProblem(system,(particles=initial,),(0.,1.);p=(k=.2,),history)
end

function main()
    pb=problem()
    solver=ForwardSolver(SSPRK33();dt=.01,control=StepControl(cfl=.4))
    solution=simulate(pb;solver,saveat=[0.,.25,.5,.75,1.])
    @show initialization_report(pb)
    @show diagnostics(solution).accepted_steps
    solution
end

function external_example(algorithm)
    # e.g. external_example(Tsit5()) after loading the optional ODE packages.
    solver=ForwardSolver(algorithm;backend=:external,reltol=1e-8,abstol=1e-11)
    solver(problem(),[0.,.25,.5,.75,1.])
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    PreparedForwardExample.main()
end
