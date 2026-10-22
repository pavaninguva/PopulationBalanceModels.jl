# Closed, constant-volume dissolution benchmark. Sizes and time are dimensionless;
# all moments are per process volume. rho_kv converts L^3 to solute concentration.
using PopulationBalanceModels
using Printf

function dissolution_reference(k,t;a=1.0,b=2.0,N0=1.0,g=1.0)
    N0/((b-a)*(k+1))*(max(b-g*t,0.0)^(k+1)-max(a-g*t,0.0)^(k+1))
end
function dissolution_problem(kind;resolution=80,tspan=(0.0,2.2),rho_kv=0.7,c0=1.0)
    model=PBMModel(shrinkage=Growth((L,p,t,ctx)->-1.0))
    if kind===:sectional
        grid=Grid1D(range(0.0,3.0;length=resolution+1);coordinate=:length)
        op=compile(model,grid;bc=SizeBC(AbsorbingZero(),NoFlux()),growth_scheme=Upwind())
        initial=[max(0.0,min(r,2.0)-max(l,1.0)) for (l,r) in zip(grid.edges[1:end-1],grid.edges[2:end])]
        exchange=ConservativeExchange(:crystals,:solute,(L,p,t,ctx)->rho_kv*L^3)
    elseif kind===:moments
        basis=MomentSet(0:2resolution-1;coordinate=:length)
        op=compile(model,basis;closure=RankAdaptiveQMOM(resolution;rtol=1e-10),
            boundary=MomentBoundary(left=:absorb))
        initial=[dissolution_reference(k,0.0) for k in basis.orders]
        exchange=MomentExchange(:crystals,:solute,3;factor=rho_kv)
    else
        throw(ArgumentError("choose :sectional or :moments"))
    end
    system=PBMSystem((crystals=op,);exchanges=(exchange,),nonnegative_bulk=(:solute,),
        accounting=(CumulativeAccounting(:crystals;properties=(volume=3,)),))
    PBMProblem(system,(crystals=initial,),tspan;bulk0=(solute=c0,))
end
function run_dissolution(;output=joinpath(@__DIR__,"..","docs","src","assets","dissolution"))
    mkpath(output);times=collect(range(0.0,2.2;length=111));rho_kv=.7
    open(joinpath(output,"dissolution.csv"),"w") do io
        println(io,"method,resolution,time,M0,M1,M2,M3,solute,removed_number,material_error")
        for (kind,resolution) in ((:sectional,80),(:sectional,160),(:moments,2),(:moments,4),(:moments,6))
            pb=dissolution_problem(kind;resolution,rho_kv)
            dt=kind===:sectional ? .35minimum(pb.operator.system.populations.crystals.grid.widths) : .02
            sol=simulate(pb;dt,saveat=times,method=MeasureEuler(reconstruction_rtol=1e-9))
            initial=1.0+rho_kv*moment(pb.operator.system.populations.crystals.grid,population(pb,pb.u0,:crystals),3)
            max_balance=0.0;max_number=0.0
            for (t,u) in zip(times,sol.u)
                pop=population(pb,u,:crystals);grid=pb.operator.system.populations.crystals.grid
                moments=[moment(grid,pop,k) for k in 0:3]
                c=bulk(pb,u,:solute);error=c+rho_kv*moments[4]-initial
                removed=cumulative(pb,u,:crystals,:extinction)
                @printf(io,"%s,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",kind,resolution,t,moments...,c,removed,error)
                max_balance=max(max_balance,abs(error));max_number=max(max_number,abs(moments[1]-dissolution_reference(0,t)))
            end
            @assert max_balance<2e-7*initial
            @printf("%-9s resolution=%3d: max number error %.5g; material error %.3g; accepted=%d rejected=%d\n",
                kind,resolution,max_number,max_balance,sol.steps,diagnostics(sol).rejected_steps)
        end
    end
    println("CSV: ",joinpath(output,"dissolution.csv"))
end
if abspath(PROGRAM_FILE)==@__FILE__
    run_dissolution()
end
