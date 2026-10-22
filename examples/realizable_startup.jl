using PopulationBalanceModels
using Printf

function run_realizable_startup()
    # J has units number/(process volume*time), G has units length/time.
    J=1.7;G=.8;basis=MomentSet(0:5;coordinate=:length)
    model=PBMModel(growth=Growth(PolynomialRate((G,))),births=Nucleation((p,t,ctx)->J,0.0))
    op=compile(model,basis;closure=RankAdaptiveQMOM(3))
    pb=PBMProblem(op,zeros(6),(0.0,1.0))
    for dt in (.2,.1,.05)
        sol=simulate(pb;dt,saveat=[0.,.25,.5,1.],method=MeasureEuler())
        error=maximum(maximum(abs.(vec(u).- [J*G^k*t^(k+1)/(k+1) for k in basis.orders])) for (u,t) in zip(sol.u,sol.t))
        @assert error<1e-8
        @printf("Empty startup dt=%.3g: maximum moment error %.3g; rejected=%d\n",dt,error,diagnostics(sol).rejected_steps)
    end
    # Combine three mechanisms and a shared solute. Aggregates add L^3.
    additive=AdditiveCoordinate(L->L^3,cbrt)
    model=PBMModel(growth=Growth((L,p,t,ctx)->.08*ctx.bulk.solute),
        nucleation=Nucleation((p,t,ctx)->.3,.05),
        collisions=Aggregation(ConstantKernel(.2);coordinate_map=additive))
    op=compile(model,MomentSet(0:3;coordinate=:length);closure=RankAdaptiveQMOM(2;rtol=1e-10))
    sys=PBMSystem((crystals=op,);exchanges=(MomentExchange(:crystals,:solute,3;factor=.7),),
        nonnegative_bulk=(:solute,),accounting=(CumulativeAccounting(:crystals),))
    pb=PBMProblem(sys,(crystals=zeros(4),),(0.,1.);bulk0=(solute=2.,))
    sol=simulate(pb;dt=.02,saveat=0.:.1:1.,method=MeasureEuler())
    error=maximum(abs(bulk(pb,u,:solute)+.7*population(pb,u,:crystals)[4]-2.) for u in sol.u)
    @assert error<1e-8
    @printf("Growth + nucleation + aggregation: M0=%.6g, material error=%.3g\n",population(pb,sol.u[end],:crystals)[1],error)
    println("Diagnostics: ",diagnostics(sol))
end
if abspath(PROGRAM_FILE)==@__FILE__
    run_realizable_startup()
end
