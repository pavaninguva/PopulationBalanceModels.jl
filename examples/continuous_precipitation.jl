module ContinuousPrecipitation
using PopulationBalanceModels
export parameters,build,run_benchmark

"""Case 7A, Zhang et al. (2024), SI Table S8; metres, seconds, kg/m³."""
parameters()=(cfeed=9.,ceq=.4,rho=1200.,kv=.524,kp=5.,u=10.,kb=4e8,b=2.,
    kg=7e-6,xc=1e-6,xmax=900e-6,Q=1.67e-7,area=1.46e-4,porosity=.21,L=.47,Dax=4.2e-5)
supersaturation(p,ctx)=max(ctx.bulk.c/p.ceq-1,zero(ctx.bulk.c))
growth(x,p,t,ctx)=p.kg*supersaturation(p,ctx)
function nucleation(p,t,ctx)
    s=supersaturation(p,ctx)
    solid=1e12*p.rho*p.kv*moment(ctx.grid,ctx.N,3)
    (p.kp*s^p.u+p.kb*s^p.b*solid)/1e12
end

"""Closed size domain; conservative discrete solute coupling; solute and particles share axial transport."""
function build(;nx=32,nz=48,scheme=Upwind(),axial_scheme=scheme,mesh=:log,p=parameters(),tf=200.)
    mesh in (:log,:uniform) || throw(ArgumentError("mesh must be :log or :uniform"))
    edges=mesh===:log ? exp.(range(log(p.xc),log(p.xmax);length=nx+1)) : collect(range(p.xc,p.xmax;length=nx+1))
    grid=Grid1D(edges;coordinate=:length)
    model=PBMModel(growth=Growth(rate=growth),primary_secondary=Nucleation(rate=nucleation,size=first(edges)))
    op=compile(model,grid;methods=TermMethods(growth=scheme,by_term=(primary_secondary=BoundaryInjection(:left),)))
    system=PBMSystem((crystals=op,);nonnegative_bulk=(:c,),
        exchanges=(ConservativeExchange(:crystals,:c,(x,p,t,ctx)->1e12*p.rho*p.kv*x^3),))
    localpb=PBMProblem(system,(crystals=zeros(nx),),(0.,tf);bulk0=(c=0.,),p)
    z=AxialGrid(range(0.,p.L;length=nz+1))
    domain=AxialDispersion(z;velocity=p.Q/(p.area*p.porosity),dispersion=p.Dax,
        inlet=(p,t)->(populations=(crystals=zeros(nx),),bulk=(c=p.cfeed,)))
    axial_problem(localpb,domain;axial_scheme)
end

# Strongly graded meshes and initial supersaturation bursts require small steps.
# This explicit driver chooses a conservative step at every state, including
# the local solute-consumption rate. It never clips populations or concentration.
function advance(problem,times;fraction=.3)
    times=collect(times);u=copy(problem.u0);t=first(problem.tspan);op=problem.operator
    grid=first(op.local_operators).system.populations.crystals.grid
    p=problem.p;dz=minimum(op.domain.grid.widths);dx=minimum(grid.widths)
    transport=op.domain.velocity/dz+2p.Dax/dz^2
    cache=ntuple(_->similar(u),6);du=similar(u);states=typeof(u)[];steps=0
    for target in times
        while t<target
            rhs!(du,u,op,p,t)
            maxG=p.kg*max(maximum(bulk(problem,u,:c))/p.ceq-1,0)
            h=fraction/(transport+maxG/dx+eps())
            for j in axes(u,2)
                c=bulk(problem,u,:c)[j];dc=bulk(problem,du,:c)[j]
                dc<0 && (h=min(h,fraction*max(c-p.ceq,1e-5)/(-dc)))
            end
            h=min(h,target-t,.25)
            h>1e-12 || throw(NumericalFailure("precipitation step collapsed; inspect the state"))
            PopulationBalanceModels._rk_step!(u,cache,problem,t,h,SSPRK33(),true,1e-8)
            PopulationBalanceModels._check_problem_state(problem,u,1e-8)
            t=h==target-t ? target : t+h;steps+=1
        end
        push!(states,copy(u))
    end
    PBMSolution(times,states,problem,steps)
end

function outlet_error(problem,u,reference,v)
    g=first(problem.operator.local_operators).system.populations.crystals.grid
    gr=first(reference.operator.local_operators).system.populations.crystals.grid
    # Compare both cell-average densities on their exact common partition.
    e=sort!(unique(vcat(g.edges,gr.edges)));a=population(problem,u,:crystals)[:,end]./g.widths
    b=population(reference,v,:crystals)[:,end]./gr.widths
    total=0.;norm=0.
    for i in 1:length(e)-1
        x=(e[i]+e[i+1])/2;da=a[clamp(searchsortedlast(g.edges,x),1,length(g))]
        db=b[clamp(searchsortedlast(gr.edges,x),1,length(gr))]
        total+=abs(da-db)*(e[i+1]-e[i]);norm+=abs(db)*(e[i+1]-e[i])
    end
    total/norm
end

"""Figure-9-style fields/outlets and Figure-10-style measured error/runtime.
quick=true is a documentation budget; full mode is expensive and still needs
reference refinement. Timings describe this Julia implementation, not CADET.
"""
function run_benchmark(;output=joinpath(@__DIR__,"..","docs","src","assets","continuous_precipitation"),quick=true)
    mkpath(output);p=parameters();levels=quick ? [(8,12),(12,18),(24,32)] : [(25,37),(50,75),(100,150),(200,300)]
    refsize=quick ? (36,48) : (250,750)
    reference=build(nx=refsize[1],nz=refsize[2],scheme=MUSCL());println("Reference ",refsize);flush(stdout)
    ref=advance(reference,[80.,200.]);rf=last(ref.u)
    open(joinpath(output,"field.csv"),"w") do io
        println(io,"time,z_m,size_m,number_density,concentration_kg_m3,z_left,z_right,x_left,x_right")
        g=first(reference.operator.local_operators).system.populations.crystals.grid
        for (t,u) in zip(ref.t,ref.u),j in axes(u,2),i in eachindex(g.centers)
            println(io,join((t,reference.operator.domain.grid.centers[j],g.centers[i],1e12*u[i,j]/g.widths[i],bulk(reference,u,:c)[j],reference.operator.domain.grid.edges[j],reference.operator.domain.grid.edges[j+1],g.edges[i],g.edges[i+1]),','))
        end
    end
    open(joinpath(output,"outlets.csv"),"w") do oi
        println(oi,"method,nx,nz,size_m,number_density,volume_density")
        open(joinpath(output,"performance.csv"),"w") do pi
            println(pi,"method,nx,nz,seconds,normalized_L1,steps")
            for (label,scheme) in (("upwind",Upwind()),("muscl",MUSCL()))
                # Compile a tiny run before timing; record warm elapsed times.
                advance(build(nx=6,nz=6,scheme=scheme,tf=.01),[.01])
                for (nx,nz) in levels
                    problem=build(;nx,nz,scheme);println(label," ",(nx,nz));flush(stdout)
                    elapsed=@elapsed solution=advance(problem,[200.])
                    u=last(solution.u);err=outlet_error(problem,u,reference,rf)
                    println(pi,join((label,nx,nz,elapsed,err,solution.steps),','))
                    g=first(problem.operator.local_operators).system.populations.crystals.grid
                    N=population(problem,u,:crystals)[:,end];volume=sum(N.*g.centers.^3)
                    for i in eachindex(N)
                        println(oi,join((label,nx,nz,g.centers[i],1e12*N[i]/g.widths[i],N[i]*g.centers[i]^3/(g.widths[i]*volume)),','))
                    end
                end
            end
        end
    end
    # A separate refinement check prevents treating the reference as exact.
    check=build(nx=refsize[1],nz=refsize[2],scheme=MUSCL())
    half=advance(check,[200.];fraction=.15)
    time_error=outlet_error(check,last(half.u),reference,rf)
    write(joinpath(output,"verification.txt"),"Zhang et al. 2024 Case 7A; SI S8.\n"*
        "Reference $(refsize), MUSCL; temporal step-halving L1 difference=$time_error.\n"*
        "Finite reference, not an exact solution; refine both size and axial meshes.\n"*
        "Solute exchange conserves the resolved discrete x^3 measure; differs from independently discretized continuum sinks at finite resolution.\n"*
        "Case 7B is not claimed: the printed lognormal centre lacks an unambiguous SI conversion.\n"*
        "Julia $(VERSION); full Figure 10 CADET scheme/timing reproduction is not claimed.\n")
    (;reference=ref,time_error)
end
if abspath(PROGRAM_FILE)==@__FILE__;run_benchmark(quick=!("--full" in ARGS));end
end
