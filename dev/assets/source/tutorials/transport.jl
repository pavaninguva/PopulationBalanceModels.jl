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

initial_density(x,y,p=parameters())=p.amplitude*exp(-0.5*((x-p.center[1])/p.width[1])^2-
    0.5*((y-p.center[2])/p.width[2])^2)

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

function problem(n=100;scheme=WENO5(epsilon=1e-12),p=parameters(),initial=nothing)
    grid=Grid2D(range(0,2;length=n+1),range(0,2;length=n+1))
    boundary=SizeBC(OpenBoundary(),NoFlux())
    model=PBMModel(growth=Growth2D(rate=growth_rate))
    operator=compile(model,grid;methods=TermMethods(growth=scheme),bc=SizeBC2D(x=boundary,y=boundary))
    # Keep explicit analytical bin integrals as the historical benchmark default.
    # A physical description remains attached for parameter-dependent remake.
    state=initial===nothing ? reference(grid,0.0,p) : initial
    PBMProblem(operator,state,(0.0,FINAL_TIME);p)
end

function tutorial_problem(; bins=50)
    # Explicit joint number density on [0,2]²; no inferred independence/normalization.
    initial = InitialDensity((xy,p,t) -> initial_density(xy[1],xy[2],p);
        support=((0.0,2.0),(0.0,2.0)))
    problem(bins; scheme=WENO5(epsilon=1e-12), p=parameters(), initial)
end

function main(; bins=50, times=[1.0])
    pb = tutorial_problem(; bins)
    dt = suggest_dt(pb.operator, pb.u0, pb.p, 0.0; cfl=0.4)
    # Keep signed undershoots in this diagnostic comparison; no clipping.
    solver = ForwardSolver(SSPRK33(); dt, check_nonnegative=false)
    solution = simulate(pb; solver, saveat=times)
    @show observe(MomentObservable((0,0)), solution)
    solution
end

# Optional after import PythonPlot. No implicit tensor marginal is selected.
plot_results(solution) = plot_field(solution; orientation=:internal_x,
    xlabel="First internal coordinate", ylabel="Second internal coordinate",
    colorbar_label="Joint number density")

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
