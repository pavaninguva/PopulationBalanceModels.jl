module InguvaBraatzCase2
using PopulationBalanceModels

export growth_rate, initial_density, analytical_density, reference, problem, mapped_problem, run_benchmark
const ALPHA=(0.05,0.25)
const FINAL_TIME=1.0
const SIGMA=0.05
const NORMAL=GaussianNucleation((p,t,ctx)->1.0;center=0.0,width=1.0,truncation=:physical)

growth_rate(x,y,p,t,ctx)=(0.1+0.05x,0.5+0.25y)
initial_density(x,y)=50exp(-((x-0.4)^2+(y-0.4)^2)/0.005)
foot(x,t,alpha)=(x+2)*exp(-alpha*t)-2
function analytical_density(x,y,t)
    x0=foot(x,t,ALPHA[1]);y0=foot(y,t,ALPHA[2])
    0<=x0<=2 && 0<=y0<=2 ? initial_density(x0,y0)*exp(-sum(ALPHA)*t) : 0.0
end
# Integrate in the initial coordinates: f da1 da2 = f0 da10 da20.
function axis_integrals(axis,t,alpha)
    [begin
        lo=max(0.0,foot(axis.edges[i],t,alpha));hi=min(2.0,foot(axis.edges[i+1],t,alpha))
        hi<=lo ? 0.0 : SIGMA*sqrt(2π)*NORMAL.fraction((lo-0.4)/SIGMA,(hi-0.4)/SIGMA,(;),t,(;))
    end for i in 1:length(axis)]
end
reference(grid,t)=50 .* (axis_integrals(grid.x,t,ALPHA[1])*axis_integrals(grid.y,t,ALPHA[2])')
function problem(n=100;scheme=WENO5(epsilon=1e-12))
    grid=Grid2D(range(0,2;length=n+1),range(0,2;length=n+1))
    boundary=SizeBC(OpenBoundary(),NoFlux())
    model=PBMModel(growth=Growth2D(growth_rate))
    operator=compile(model,grid;methods=TermMethods(growth=scheme),bc=SizeBC2D(x=boundary,y=boundary))
    PBMProblem(operator,reference(grid,0.0),(0.0,FINAL_TIME))
end

"""
Constant-speed problem in travel-time coordinates tau=log(1+a/2)/alpha.
Physical edges are obtained by the inverse map a=2expm1(alpha*tau).
A common tau increment dt makes ExactShift move one cell on both axes.
The end cells are truncated to whole travel-time cells inside [0,2]; the
physical mesh extent is recorded. No partial final cell is silently remapped.
"""
function mapped_problem(steps=20)
    steps>=2 || throw(ArgumentError("at least two steps required"))
    dt=FINAL_TIME/steps
    axes=map(ALPHA) do alpha
        last_index=floor(Int,log(2)/(alpha*dt))
        collect(0:last_index).*dt
    end
    tau_grid=Grid2D(axes...)
    physical_grid=Grid2D(2expm1.(ALPHA[1].*axes[1]),2expm1.(ALPHA[2].*axes[2]))
    boundary=SizeBC(OpenBoundary(),OpenBoundary())
    operator=compile(PBMModel(growth=Growth2D((1.0,1.0))),tau_grid;
        bc=SizeBC2D(x=boundary,y=boundary))
    problem=PBMProblem(operator,reference(physical_grid,0.0),(0.0,FINAL_TIME))
    (;problem,physical_grid,dt)
end
function write_field(path,grid,N)
    open(path,"w") do io
        println(io,"a1,a2,density")
        f=density(grid,N)
        for j in axes(N,2),i in axes(N,1)
            println(io,grid.x.centers[i],',',grid.y.centers[j],',',f[i,j])
        end
    end
end
function metrics(N,truth,grid)
    area=grid.x.widths*grid.y.widths';e=(N.-truth)./area;f=density(grid,N)
    (;RMSE=sqrt(sum(abs2,e)/length(e)),MAE=maximum(abs,e),L1=sum(abs,N.-truth),
        minimum_density=minimum(f),negative_number=-sum(min.(N,0.0)))
end
function run_benchmark(;grids=(50,100,200),output=joinpath(@__DIR__,"..","docs","src","assets","inguva_case2"))
    mkpath(output);rows=NamedTuple[]
    for n in grids
        for (label,scheme) in (("upwind",Upwind()),("weno5",WENO5(epsilon=1e-12)))
            prob=problem(n;scheme);grid=prob.operator.grid
            # Maximum velocities on [0,2]^2 are 0.2 and 1.0; summed CFL <=0.4.
            dt=0.4/(0.2/minimum(grid.x.widths)+1.0/minimum(grid.y.widths))
            sol=simulate(prob;dt,saveat=[1.0],check_nonnegative=false)
            truth=reference(grid,1.0)
            row=merge((;resolution=n,method=label,nx=length(grid.x),ny=length(grid.y),dt,
                upper_a1=last(grid.x.edges),upper_a2=last(grid.y.edges)),metrics(last(sol.u),truth,grid))
            push!(rows,row);println(row)
            if n==100 || n==maximum(grids)
                write_field(joinpath(output,"$(label)_$(n).csv"),grid,last(sol.u))
                label=="upwind" && write_field(joinpath(output,"reference_$(n).csv"),grid,truth)
            end
        end
        mapped=mapped_problem(max(2,n÷5))
        sol=simulate(mapped.problem;method=ExactShift(),dt=mapped.dt,saveat=[1.0])
        grid=mapped.physical_grid;truth=reference(grid,1.0)
        row=merge((;resolution=n,method="mapped_shift",nx=length(grid.x),ny=length(grid.y),dt=mapped.dt,
            upper_a1=last(grid.x.edges),upper_a2=last(grid.y.edges)),metrics(last(sol.u),truth,grid))
        push!(rows,row);println(row)
        if n==100 || n==maximum(grids)
            write_field(joinpath(output,"mapped_shift_$(n).csv"),grid,last(sol.u))
            write_field(joinpath(output,"mapped_reference_$(n).csv"),grid,truth)
        end
    end
    open(joinpath(output,"errors.csv"),"w") do io
        println(io,join(keys(first(rows)),','))
        for row in rows;println(io,join(values(row),','));end
    end
    open(joinpath(output,"environment.txt"),"w") do io
        println(io,"Inguva & Braatz (2023), Case 2, Eqs.57–60; Figure 4 at t=1.")
        println(io,"Julia ",VERSION,"; package ",Base.pkgversion(PopulationBalanceModels),"; ",Sys.KERNEL,' ',Sys.ARCH)
        println(io,"Conservative FV comparison uses bin-integrated numbers; output densities are bin averages.")
        println(io,"Mapped shifts use travel-time coordinates with a common spacing and paired open boundaries.")
        println(io,"Uniform FV grids cover [0,2]^2; mapped grids omit fractional last cells, extents recorded in errors.csv.")
        println(io,"Mapped comparison is a bin-average adaptation, not the paper's nodal implementation.")
        println(io,"HOCADI cannot solve this variable-growth problem directly and is not included.")
    end
    rows
end
if abspath(PROGRAM_FILE)==@__FILE__;run_benchmark();end
end
