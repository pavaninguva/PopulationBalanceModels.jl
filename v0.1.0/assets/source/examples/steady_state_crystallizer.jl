module SteadyStateCrystallizer

using PopulationBalanceModels
using NonlinearSolve
using Printf

"""Unseeded-feed, well-mixed continuous crystallizer. L in length units, time in
time units, J in number/(process volume*time), and G in length/time. Particle
withdrawal has the same residence time as the liquid. No physical initial
condition is needed for this stationary boundary-value problem.
"""
function problem(;bins=160,upper=12.0,p=(G=0.4,J=1.2,tau=2.0))
    grid=Grid1D(range(0,upper;length=bins+1);coordinate=:length)
    model=PBMModel(
        growth=Growth((L,p,t,ctx)->p.G),
        nucleation=Nucleation((p,t,ctx)->p.J,0.0),
        withdrawal=Dilution((L,p,t,ctx)->1/p.tau),
    )
    op=compile(model,grid;
        methods=TermMethods(nucleation=BoundaryInjection()),
        bc=SizeBC(NoFlux(),OpenBoundary()))
    StationaryPBMProblem(op,zeros(bins);p,autonomous=true)
end

function exact_bins(grid,p)
    a=p.G*p.tau
    p.J*p.tau.*(exp.(-grid.edges[1:end-1]/a).-exp.(-grid.edges[2:end]/a))
end

"""An exact raw-moment model coupled to a flowing solute balance.
The independently checked scalar equation is c+alpha*6J*G(c)^3*tau^4=feed.
"""
function coupled_problem(;p=(J=.5,kg=.2,tau=2.,cs=1.,feed=5.,alpha=.3))
    model=PBMModel(
        Growth(PolynomialRate(((p,t,ctx)->p.kg*(ctx.bulk.c-p.cs),)))+
        Nucleation((p,t,ctx)->p.J,0.)+
        Dilution(PolynomialRate(((p,t,ctx)->1/p.tau,))))
    op=compile(model,MomentSet(0:3;coordinate=:length))
    system=PBMSystem((crystals=op,);
        bulk_rhs=(ctx,p,t)->(c=(p.feed-ctx.bulk.c)/p.tau,),
        exchanges=(MomentExchange(:crystals,:c,3;factor=(p,t,ctx)->p.alpha),),
        nonnegative_bulk=(:c,))
    # Numerical guess only: moments of an exponential distribution at c=3.
    G=p.kg*(3-p.cs)
    guess=[factorial(k)*p.J*G^k*p.tau^(k+1) for k in 0:3]
    StationaryPBMProblem(system,(crystals=guess,);bulk=(c=3.,),p,autonomous=true)
end

function run()
    println("bins   integrated L1 error   number-balance error   scaled residual")
    for bins in (40,80,160,320)
        prob=problem(;bins)
        sol=solve_stationary(prob;abstol=1e-10)
        sol.converged || error("stationary crystallizer failed: $(sol.retcode)")
        grid=prob.operator.grid; p=prob.p
        error=sum(abs.(vec(sol.u).-exact_bins(grid,p)))
        outlet=p.G*sol.u[end]/grid.widths[end]
        balance=abs(p.J-sum(sol.u)/p.tau-outlet)
        @printf("%4d   %.6e          %.3e              %.3e\n",
            bins,error,balance,diagnostics(sol).scaled_balance_norm)
    end
    coupled=coupled_problem()
    sol=solve_stationary(coupled;abstol=1e-10)
    sol.converged || error("coupled stationary crystallizer failed")
    c=bulk(sol,:c); M3=population(sol,:crystals)[4]; p=coupled.p
    @printf("Coupled c = %.9f; c + alpha*M3 - feed = %.3e\n",c,c+p.alpha*M3-p.feed)
    branch=continue_stationary(coupled,λ->merge(p,(feed=3.5+1.5λ,)),(0.,1.);
        abstol=1e-9)
    branch.converged || error("continuation did not reach the requested feed")
    @printf("Continuation: %d accepted states, final c = %.9f\n",
        length(branch.solutions),bulk(branch.solutions[end],:c))
    sol
end

end

if abspath(PROGRAM_FILE)==@__FILE__
    SteadyStateCrystallizer.run()
end
