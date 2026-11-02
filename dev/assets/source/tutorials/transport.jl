# Minimal transport tutorial.
# Physical definitions and one calculation; numerical helpers are linked on the case page.
# No benchmark exports or retained files are written. See the case-study page.

using PopulationBalanceModels

# Physical definitions from examples/inguva_braatz_case2.jl.
const FINAL_TIME=1.0

const NORMAL=GaussianNucleation((p,t,ctx)->1.0;center=0.0,width=1.0,truncation=:physical)

parameters(;offset=(0.1,0.5),slope=(0.05,0.25),amplitude=50.0,
    center=(0.4,0.4),width=(0.05,0.05))=(;offset,slope,amplitude,center,width)

function growth_rate(x,y,p,t,ctx)
    (p.offset[1]+p.slope[1]*x,p.offset[2]+p.slope[2]*y)
end

foot(x,t,b,s)=iszero(s) ? x-b*t : x*exp(-s*t)+b*expm1(-s*t)/s

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

function tutorial_problem(; bins=50)
    # Exact initial Gaussian bin integrals; one WENO5 configuration.
    problem(bins; scheme=WENO5(epsilon=1e-12), p=parameters())
end

function main()
    pb = tutorial_problem()
    dt = suggest_dt(pb.operator, pb.u0, pb.p, 0.0; cfl=0.4)
    # Keep signed undershoots in this diagnostic comparison; no clipping.
    solution = simulate(pb; method=SSPRK33(), dt, saveat=[1.0], check_nonnegative=false)
    @show observe(MomentObservable((0,0)), solution)
    solution
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
