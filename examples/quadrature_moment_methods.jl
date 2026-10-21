module QuadratureMomentMethods
using PopulationBalanceModels
export run_comparison

"""An analytic scaling benchmark: G(x)=0.2x and an initial Gamma(10,0.2) PSD.
All raw moments obey dμk/dt=0.2kμk. Also check fixed-rank and depleted-rank limits.
"""
function run_comparison(;output=joinpath(@__DIR__,"..","docs","src","assets","quadrature_moments"))
    mkpath(output);times=collect(0.:.1:2.)
    initial(k)=prod((2+.2j for j in 0:k-1);init=1.)
    model=PBMModel(growth=Growth(rate=(x,p,t,ctx)->.2x))
    open(joinpath(output,"moments.csv"),"w") do io
        println(io,"method,time,order,moment,exact,relative_error,active_rank")
        for method in (QMOM(2),RankAdaptiveQMOM(3),EQMOM(2),DQMOM(2))
            K=method isa EQMOM ? 4 : method isa RankAdaptiveQMOM ? 5 : 3
            basis=MomentSet(0:K);raw=[initial(k) for k in basis.orders]
            if method isa DQMOM
                q=quadrature(basis,raw,QMOM(2));state=quadrature_state(method,q.nodes,q.weights)
            else
                state=raw
            end
            op=compile(model,basis;method);problem=PBMProblem(op,state,(0.,2.))
            sol=simulate(problem;dt=.005,saveat=times)
            # Save a smooth density from EQMOM's declared kernel at final time.
            if method isa EQMOM
                open(joinpath(output,"density.csv"),"w") do di
                    println(di,"size,density,exact")
                    scale=.2exp(.4)
                    for x in range(.01,8.;length=400)
                        exact=x^9*exp(-x/scale)/(factorial(9)*scale^10)
                        println(di,join((x,reconstructed_density(basis,last(sol.u),method,x),exact),','))
                    end
                end
            end
            for (t,u) in zip(sol.t,sol.u),k in 0:3
                value=moment_value(method,basis,u,k);truth=initial(k)*exp(.2k*t)
                q=quadrature(basis,u,method)
                rank=hasproperty(q,:rank) ? q.rank : length(q.nodes)
                println(io,join((nameof(typeof(method)),t,k,value,truth,abs(value-truth)/truth,rank),','))
            end
        end
    end
    # A three-node capacity is allowed to represent a two-point distribution.
    x=[1.,3.];w=[.4,.6];raw=[sum(w.*x.^k) for k in 0:5]
    q=quadrature(MomentSet(0:5),raw,RankAdaptiveQMOM(3))
    write(joinpath(output,"rank.txt"),"Maximum nodes: 3; recovered active rank: $(q.rank); all-moment residual: $(q.moment_error).\n")
end
if abspath(PROGRAM_FILE)==@__FILE__;run_comparison();end
end
