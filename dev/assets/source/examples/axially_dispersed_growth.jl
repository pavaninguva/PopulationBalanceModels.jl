module AxiallyDispersedGrowth
using PopulationBalanceModels
using Distributions: Normal, pdf

"""Synthetic empty tube: size in mm, axial position in m, time in s."""
function build_problem(;bins=40,axial_cells=30,tspan=(0.0,5.0))
    sizes=Grid1D(range(0.0,2.0;length=bins+1);coordinate=:length)
    axial=AxialGrid(range(0.0,1.0;length=axial_cells+1))
    p=(seed_concentration=1.0,seed_mean=0.3,seed_std=0.08,growth_rate=0.08)
    # particles/(m^3 fluid * mm). The unrepresented Gaussian tail is not moved.
    inlet=DensityInlet((L,p,t)->p.seed_concentration*pdf(Normal(p.seed_mean,p.seed_std),L);
        quadrature=GaussLegendre(4))
    reactor=AxialDispersion(axial;velocity=0.2,dispersion=0.005,inlet)
    model=PBMModel(growth=Growth((L,p,t,ctx)->p.growth_rate))
    operator=compile(model,sizes;domain=reactor,axial_scheme=MUSCL(),
        methods=TermMethods(growth=Upwind()),bc=SizeBC(OpenBoundary(),OpenBoundary()))
    PBMProblem(operator,zeros(length(sizes),length(axial)),tspan;p)
end

"""Run without file output by default. Pass a new output directory for optional CSV export."""
function run(;output=nothing)
    problem=build_problem()
    operator=problem.operator;sizes=operator.grid;axial=operator.domain.grid
    # A conservative choice for this example; suggest_dt is a local Upwind-based
    # recommendation, not a high-order positivity guarantee or adaptive controller.
    dt=suggest_dt(operator,problem.u0,problem.p,0.0;cfl=0.25)
    solution=simulate(problem;dt,method=SSPRK33(),saveat=[0.0,1.0,3.0,5.0])
    if output!==nothing
        mkpath(output)
        open(joinpath(output,"solution.csv"),"w") do io
            println(io,"time,position,size,density")
            for (t,N) in zip(solution.t,solution.u),c in 1:length(axial)
                f=density(sizes,axial_population(operator,N,c))
                for i in 1:length(sizes)
                    println(io,join((t,axial.centers[c],sizes.centers[i],f[i]),','))
                end
            end
        end
    end
    println((steps=solution.steps,dt,minimum=minimum(last(solution.u)),output))
    solution
end
if abspath(PROGRAM_FILE)==@__FILE__
    run()
end
end
