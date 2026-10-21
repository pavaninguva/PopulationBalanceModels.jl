module InguvaBraatzCase2
using PopulationBalanceModels

export parameters, growth_rate, initial_density, analytical_density, reference, problem, mapped_problem, run_benchmark
const FINAL_TIME=1.0
const NORMAL=GaussianNucleation((p,t,ctx)->1.0;center=0.0,width=1.0,truncation=:physical)

"""Affine growth Gᵢ=offsetᵢ+slopeᵢ*aᵢ and the initial Gaussian, in the benchmark's units."""
parameters(;offset=(0.1,0.5),slope=(0.05,0.25),amplitude=50.0,
    center=(0.4,0.4),width=(0.05,0.05))=(;offset,slope,amplitude,center,width)
function growth_rate(x,y,p,t,ctx)
    (p.offset[1]+p.slope[1]*x,p.offset[2]+p.slope[2]*y)
end
initial_density(x,y,p=parameters())=p.amplitude*exp(-0.5*((x-p.center[1])/p.width[1])^2-
    0.5*((y-p.center[2])/p.width[2])^2)
foot(x,t,b,s)=iszero(s) ? x-b*t : x*exp(-s*t)+b*expm1(-s*t)/s
function analytical_density(x,y,t,p=parameters())
    x0=foot(x,t,p.offset[1],p.slope[1]);y0=foot(y,t,p.offset[2],p.slope[2])
    0<=x0<=2 && 0<=y0<=2 ? initial_density(x0,y0,p)*exp(-sum(p.slope)*t) : 0.0
end
# Integrate in the initial coordinates: f da1 da2 = f0 da10 da20.
function axis_integrals(axis,t,b,s,center,width)
    [begin
        lo=max(0.0,foot(axis.edges[i],t,b,s));hi=min(2.0,foot(axis.edges[i+1],t,b,s))
        hi<=lo ? 0.0 : width*sqrt(2π)*NORMAL.fraction((lo-center)/width,(hi-center)/width,(;),t,(;))
    end for i in 1:length(axis)]
end
function reference(grid,t,p=parameters())
    integrals=map((grid.x,grid.y),p.offset,p.slope,p.center,p.width) do axis,b,s,center,width
        axis_integrals(axis,t,b,s,center,width)
    end
    p.amplitude.*(integrals[1]*integrals[2]')
end
function problem(n=100;scheme=WENO5(epsilon=1e-12),p=parameters())
    grid=Grid2D(range(0,2;length=n+1),range(0,2;length=n+1))
    boundary=SizeBC(OpenBoundary(),NoFlux())
    model=PBMModel(growth=Growth2D(rate=growth_rate))
    operator=compile(model,grid;methods=TermMethods(growth=scheme),bc=SizeBC2D(x=boundary,y=boundary))
    PBMProblem(operator,reference(grid,0.0,p),(0.0,FINAL_TIME);p)
end

"""
Compile the public InguvaBraatz2023 method on its physical travel-time mesh.
A common travel-time increment dt moves one cell on both axes.
The end cells are truncated to whole travel-time cells inside [0,2]; the
physical mesh extent is recorded. No partial final cell is silently remapped.
"""
function mapped_problem(steps=20;p=parameters())
    steps>=2 || throw(ArgumentError("at least two steps required"))
    all(>(0),p.offset) && all(>=(0),p.slope) ||
        throw(ArgumentError("this travel-time adapter requires positive offsets and nonnegative slopes"))
    dt=FINAL_TIME/steps
    law=SeparableGrowth2D(AffineGrowth(p.offset[1],p.slope[1]),AffineGrowth(p.offset[2],p.slope[2]))
    physical_grid=characteristic_grid(law,((0.0,2.0),(0.0,2.0));dt,endpoint=:truncate)
    boundary=SizeBC(OpenBoundary(),OpenBoundary())
    operator=compile(PBMModel(growth=Growth2D(law)),physical_grid;
        evolution=InguvaBraatz2023(),bc=SizeBC2D(x=boundary,y=boundary))
    problem=PBMProblem(operator,reference(physical_grid,0.0,p),(0.0,FINAL_TIME);p)
    (;problem,physical_grid,dt)
end
function write_field(path,grid,N)
    open(path,"w") do io
        println(io,"a1,a2,density,lower_a1,upper_a1,lower_a2,upper_a2")
        f=density(grid,N)
        for j in axes(N,2),i in axes(N,1)
            println(io,join((grid.x.centers[i],grid.y.centers[j],f[i,j],
                grid.x.edges[i],grid.x.edges[i+1],grid.y.edges[j],grid.y.edges[j+1]),','))
        end
    end
end
function metrics(N,truth,grid)
    area=grid.x.widths*grid.y.widths';e=(N.-truth)./area;f=density(grid,N)
    (;RMSE=sqrt(sum(abs2,e)/length(e)),MAE=maximum(abs,e),L1=sum(abs,N.-truth),
        RMS_area=sqrt(sum(abs2.(e).*area)/sum(area)),
        minimum_density=minimum(f),negative_number=-sum(min.(N,0.0)))
end
function run_benchmark(;grids=(50,100,200),output=joinpath(@__DIR__,"..","docs","src","assets","inguva_case2"))
    mkpath(output);rows=NamedTuple[]
    for n in grids
        for (label,scheme) in (("upwind",Upwind()),("weno5",WENO5(epsilon=1e-12)))
            prob=problem(n;scheme);grid=prob.operator.grid
            dt=suggest_dt(prob.operator,prob.u0,prob.p,0.0;cfl=0.4)
            sol=simulate(prob;dt,saveat=[1.0],check_nonnegative=false)
            truth=reference(grid,1.0,prob.p)
            row=merge((;resolution=n,method=label,nx=length(grid.x),ny=length(grid.y),dt,
                upper_a1=last(grid.x.edges),upper_a2=last(grid.y.edges)),metrics(last(sol.u),truth,grid))
            push!(rows,row);println(row)
            if n==100 || n==maximum(grids)
                write_field(joinpath(output,"$(label)_$(n).csv"),grid,last(sol.u))
                label=="upwind" && write_field(joinpath(output,"reference_$(n).csv"),grid,truth)
            end
        end
        mapped=mapped_problem(max(2,n÷5))
        sol=simulate(mapped.problem;dt=mapped.dt,saveat=[1.0])
        grid=mapped.physical_grid;truth=reference(grid,1.0,mapped.problem.p)
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
        println(io,"Mapped shifts use the public InguvaBraatz2023 compilation, a common travel-time spacing and paired open boundaries.")
        println(io,"Uniform FV grids cover [0,2]^2; mapped grids omit fractional last cells, extents recorded in errors.csv.")
        println(io,"Mapped comparison is a bin-average adaptation, not the paper's nodal implementation.")
        println(io,"HOCADI cannot solve this variable-growth problem directly and is not included.")
    end
    rows
end
if abspath(PROGRAM_FILE)==@__FILE__;run_benchmark();end
end
