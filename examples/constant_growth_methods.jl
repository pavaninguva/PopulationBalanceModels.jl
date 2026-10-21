module ConstantGrowthMethods
using PopulationBalanceModels

const growth=0.1
initial_density(a)=a>=0 ? 100exp(-a/0.01) : 0.0
primitive(a)=a<=0 ? 0.0 : -expm1(-a/0.01)
exact_counts(grid,t)=[primitive(b-growth*t)-primitive(a-growth*t) for (a,b) in zip(grid.edges[1:end-1],grid.edges[2:end])]

function combinations()
    choices=Tuple{String,String,Any,Any}[]
    for (name,scheme) in (("upwind",Upwind()),("muscl",MUSCL()),("weno5",WENO5()),("centered",Centered()),("linear_upwind",LinearUpwind()))
        times=name=="upwind" ? (("euler",ForwardEuler()),("ssprk22",SSPRK22()),("ssprk33",SSPRK33()),("rk4",RK4())) :
              name=="muscl" ? (("ssprk22",SSPRK22()),("ssprk33",SSPRK33()),("rk4",RK4())) : (("ssprk33",SSPRK33()),("rk4",RK4()))
        for (tname,time) in times;push!(choices,(name,tname,scheme,time));end
    end
    append!(choices,[("lax_wendroff","coupled",nothing,LaxWendroff()),("beam_warming","coupled",nothing,BeamWarming())])
    choices
end

function run(;grids=(50,100,200,400),courants=(0.5,1.0),tf=4.0,
        output=joinpath(@__DIR__,"..","docs","src","assets","constant_growth"))
    mkpath(output)
    model=PBMModel(growth=Growth(AffineGrowth(growth,0.0)))
    bc=SizeBC(OpenBoundary(),NoFlux())
    open(joinpath(output,"errors.csv"),"w") do io
        println(io,"scheme,time_method,courant,bins,rmse,mean_absolute_error,max_error,minimum_density,negative_number,number_drift,seconds,steps")
        for n in grids,C in courants,(name,tname,scheme,time) in combinations()
            grid=Grid1D(range(0,1;length=n+1);coordinate=:length);dt=C/(n*growth)
            operator=scheme===nothing ? compile(model,grid;evolution=time,bc) : compile(model,grid;growth_scheme=scheme,bc)
            prob=PBMProblem(operator,exact_counts(grid,0.0),(0.0,tf))
            # Warm exactly this combination before recording its elapsed time.
            simulate(prob;method=time,dt,check_nonnegative=false)
            timed=@timed simulate(prob;method=time,dt,check_nonnegative=false)
            N=vec(last(timed.value.u));truth=exact_counts(grid,tf);f=density(grid,N);err=density(grid,N.-truth)
            println(io,join((name,tname,C,n,sqrt(sum(err.^2 .* grid.widths)),sum(abs.(err).*grid.widths),maximum(abs.(err)),minimum(f),-sum(min.(N,0)),sum(N)-sum(prob.u0),timed.time,timed.value.steps),','))
            if n==200 || n==last(grids)
                open(joinpath(output,"$(name)_$(tname)_$(C)_$(n).csv"),"w") do data
                    println(data,"size,density,exact")
                    for i in eachindex(f);println(data,join((grid.centers[i],f[i],truth[i]/grid.widths[i]),','));end
                end
            end
        end
    end
    open(joinpath(output,"inputs.txt"),"w") do io
        println(io,"Constant-growth model: Inguva, Schickel and Braatz (2022), Case 1, Eq.56 and Figure 1.")
        println(io,"g=0.1; initial f(a)=100exp(-a/0.01), a>=0; domain [0,1]; final time=",tf)
        println(io,"Zero incoming density at left, zero total growth flux at right. Exact comparison uses bin integrals, not center samples.")
        println(io,"No clipping or positivity enforcement: negative density and negative number are exported explicitly.")
        println(io,"Timing is one warmed run, not a portable hardware performance claim; Julia ",VERSION,"; CPU ",Sys.CPU_NAME)
    end
    println("Constant-growth comparison written to ",output)
end
if abspath(PROGRAM_FILE)==@__FILE__;run();end
end
