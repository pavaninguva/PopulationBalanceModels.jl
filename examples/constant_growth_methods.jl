module ConstantGrowthMethods
using PopulationBalanceModels

const growth=0.1
initial_density(a)=a>=0 ? 100exp(-a/0.01) : 0.0
primitive(a)=a<=0 ? 0.0 : -expm1(-a/0.01)
function exact_counts(grid,t)
    [b<=growth*t ? 0.0 : exp(-max(a-growth*t,0)/0.01)*
        (-expm1(-(b-max(a,growth*t))/0.01))
        for (a,b) in zip(grid.edges[1:end-1],grid.edges[2:end])]
end
function smooth_counts(grid,t)
    a=grid.edges[1:end-1].-growth*t;b=grid.edges[2:end].-growth*t
    grid.widths.+0.2.*(cos.(2pi.*a).-cos.(2pi.*b))./(2pi).+
        0.1.*(sin.(4pi.*b).-sin.(4pi.*a))./(4pi)
end

# A small set separates space, time and a complete characteristic update.
combinations()=[
    ("upwind","euler",Upwind(),ForwardEuler()),
    ("muscl","ssprk33",MUSCL(),SSPRK33()),
    ("weno5","ssprk33",WENO5(),SSPRK33()),
    ("weno5","rk4",WENO5(),RK4()),
    ("lax_wendroff","coupled",nothing,LaxWendroff()),
    ("inguva2022","coupled",nothing,Inguva2022())]

function make_problem(grid,initial,scheme,time,tf,bc)
    model=PBMModel(growth=Growth(AffineGrowth(growth,0.0)))
    op=scheme===nothing ? compile(model,grid;evolution=time,bc) : compile(model,grid;growth_scheme=scheme,bc)
    PBMProblem(op,initial,(0.,tf))
end
function measure(grid,initial,truth,scheme,time,tf,dt,bc)
    problem=make_problem(grid,initial,scheme,time,tf,bc)
    # Exactly ONE measured solve. Small, separate warmups below compile the
    # method without repeating every benchmark-sized calculation.
    timed=@timed simulate(problem;method=time,dt,check_nonnegative=false)
    solution=timed.value;N=vec(last(solution.u));f=density(grid,N);error=density(grid,N.-truth)
    (;N,f,rmse=sqrt(sum(error.^2 .*grid.widths)/sum(grid.widths)),
        l1=sum(abs.(error).*grid.widths),maximum_error=maximum(abs.(error)),
        minimum_density=minimum(f),negative_number=-sum(min.(N,0)),
        number_drift=sum(N)-sum(initial),seconds=timed.time,steps=solution.steps)
end

function run(;grids=(50,100,200,400,800,1600),smooth_grids=(40,80,160,320,640),
        courants=(0.5,1.0),tf=4.,output=joinpath(@__DIR__,"..","docs","src","assets","constant_growth"))
    mkpath(output);choices=combinations()
    # Compile each scheme on a tiny grid for BOTH boundary classes. Timings
    # then describe stepping/allocation, not Julia's first-call compilation.
    for (_,_,scheme,time) in choices,periodic in (false,true)
        g=Grid1D(range(0.,1.;length=21));dt=(time isa Inguva2022 ? 1. : .5)/(20growth)
        bc=periodic ? SizeBC(Periodic(),Periodic()) : SizeBC(OpenBoundary(),OpenBoundary())
        simulate(make_problem(g,exact_counts(g,0.),scheme,time,dt,bc);dt,check_nonnegative=false)
    end
    open(joinpath(output,"errors.csv"),"w") do io
        println(io,"scheme,time_method,courant,bins,rmse,l1,max_error,minimum_density,negative_number,number_drift,seconds,steps")
        for n in grids,C in courants,(name,tname,scheme,time) in choices
            name=="inguva2022" && C!=1. && continue
            g=Grid1D(range(0.,1.;length=n+1);coordinate=:length);truth=exact_counts(g,tf)
            # A common zero-inflow/open-outflow condition; initial outgoing tails
            # are negligible at t=4. Characteristic transport requires this pair.
            result=measure(g,exact_counts(g,0.),truth,scheme,time,tf,C/(n*growth),SizeBC(OpenBoundary(),OpenBoundary()))
            println(io,join((name,tname,C,n,result.rmse,result.l1,result.maximum_error,
                result.minimum_density,result.negative_number,result.number_drift,result.seconds,result.steps),','))
            if n==200
                open(joinpath(output,"$(name)_$(tname)_$(C)_$(n).csv"),"w") do data
                    println(data,"size,density,exact")
                    for i in eachindex(result.f);println(data,join((g.centers[i],result.f[i],truth[i]/g.widths[i]),','));end
                end
            end
        end
    end
    open(joinpath(output,"smooth_errors.csv"),"w") do io
        println(io,"scheme,time_method,bins,dt,courant,rmse,l1,observed_rmse_order,observed_l1_order,seconds,steps,minimum_density,number_drift")
        for (name,tname,scheme,time) in choices
            old_rms=NaN;old_l1=NaN;old_n=0
            for n in smooth_grids
                g=Grid1D(range(0.,1.;length=n+1));C=name=="inguva2022" ? 1. : .5;dt=C/(n*growth)
                result=measure(g,smooth_counts(g,0.),smooth_counts(g,tf),scheme,time,tf,dt,SizeBC(Periodic(),Periodic()))
                qr=name!="inguva2022" && isfinite(old_rms) ? log(old_rms/result.rmse)/log(n/old_n) : NaN
                ql=name!="inguva2022" && isfinite(old_l1) ? log(old_l1/result.l1)/log(n/old_n) : NaN
                println(io,join((name,tname,n,dt,C,result.rmse,result.l1,qr,ql,result.seconds,result.steps,result.minimum_density,result.number_drift),','))
                old_rms=result.rmse;old_l1=result.l1;old_n=n
            end
        end
    end
    write(joinpath(output,"inputs.txt"),
        "Inguva et al. (2022), Case 1 Eq.56: g=.1, f0=100exp(-a/.01), domain [0,1], tf=$tf.\n"*
        "All errors compare analytical bin integrals. Zero inflow/open outflow; far-boundary exponential tail is negligible.\n"*
        "Smooth check: f0=1+.2sin(2πa)+.1cos(4πa), periodic [0,1].\n"*
        "Six representative pairings. Common C=.5; Inguva2022 requires C=1 and is labelled separately. Front profiles also compare C=1.\n"*
        "One measured solve per configuration, following tiny separate compilation warmups. No timing medians, repeated production runs or timing bands.\n"*
        "Observed L1 and RMSE orders are measured from successive grids; no fitted or arbitrarily anchored order-guide lines.\n"*
        "Signed density minima and number drift retained. Julia $(VERSION); CPU $(Sys.CPU_NAME).\n")
    println("Growth comparison written to ",output)
end
if abspath(PROGRAM_FILE)==@__FILE__;run();end
end
