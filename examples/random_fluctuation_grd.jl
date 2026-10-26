module RandomFluctuationGRD

using PopulationBalanceModels
using SpecialFunctions: erf

# Synthetic length/time units throughout; number is per unit process volume.
function gaussian_bins(grid,mean,variance)
    cdf(L) = (1+erf((L-mean)/sqrt(2variance)))/2
    diff(cdf.(grid.edges))
end

function run(;bins=240)
    grid = Grid1D(range(0,12;length=bins+1);coordinate=:length)
    parameters = (G=.2,D=.08)
    model = PBMModel(
        growth=Growth((L,p,t,ctx)->p.G),
        dispersion=SizeDiffusion((L,p,t,ctx)->p.D),
    )
    operator = compile(model,grid;methods=TermMethods(growth=MUSCL()),bc=SizeBC())
    initial = gaussian_bins(grid,4.,.25)
    problem = PBMProblem(operator,initial,(0.,.5);p=parameters)
    # Constant coefficients allow one recommendation for this whole calculation.
    dt = suggest_dt(operator,problem.u0,parameters,0.;cfl=.45)
    solution = simulate(problem;dt,method=SSPRK33(),saveat=[0.,.25,.5])

    N = vec(solution.u[end])
    t = last(solution.t)
    reference = gaussian_bins(grid,4+parameters.G*t,.25+2parameters.D*t)
    number = sum(N)
    mean = moment(grid,N,1)/number
    variance = moment(grid,N,2)/number-mean^2
    report = (
        bins=bins,dt=dt,steps=solution.steps,
        number_error=number-sum(initial),
        mean=mean,exact_mean=4+parameters.G*t,
        variance=variance,exact_variance=.25+2parameters.D*t,
        integrated_L1=sum(abs.(N-reference)),minimum_density=minimum(N ./ grid.widths),
    )
    (;grid,solution,reference,report)
end

"""Synthetic two-coordinate population with current-stage bulk-dependent GRD.
The fixed particle property alpha*x*y^2 and solute share one discrete inventory.
"""
function run_coupled_tensor(;nx=40,ny=32)
    grid=Grid2D(range(0,12;length=nx+1),range(0,10;length=ny+1))
    p=(Gx=.15,Gy=.08,theta_x=1.5,theta_y=2.,alpha=.001)
    model=PBMModel(
        growth=Growth2D((x,y,p,t,ctx)->(p.Gx*ctx.bulk.c,p.Gy*ctx.bulk.c)),
        dispersion=SizeDiffusion((xy,p,t,ctx)->(
            p.theta_x*(p.Gx*ctx.bulk.c)^2,
            p.theta_y*(p.Gy*ctx.bulk.c)^2)),
    )
    operator=compile(model,grid) # Upwind growth; centred conservative Ito diffusion
    system=PBMSystem((crystals=operator,);
        exchanges=(ConservativeExchange(:crystals,:c,(xy,p,t,ctx)->p.alpha*xy[1]*xy[2]^2),),
        nonnegative_bulk=(:c,))
    initial=gaussian_bins(grid.x,4.,.25)*gaussian_bins(grid.y,3.,.25)'
    problem=PBMProblem(system,(crystals=initial,),(0.,.5);bulk0=(c=1.,),p)
    material(u)=bulk(problem,u,:c)+p.alpha*moment(grid,population(problem,u,:crystals),(1,2))
    total=material(problem.u0)
    # Nonnegative particle material gives c<=total throughout this closed run.
    # Both coefficients increase with c, so this bounds the population rates.
    upper=copy(problem.u0);upper[end]=total
    dt=suggest_dt(problem.operator,upper,p,0.;cfl=.3)
    solution=simulate(problem;dt,saveat=[0.,.25,.5])
    report=(shape=size(grid),dt,steps=solution.steps,
        number_error=sum(population(problem,last(solution.u),:crystals))-sum(initial),
        material_error=maximum(abs(material(u)-total) for u in solution.u),
        minimum_population=minimum(minimum(population(problem,u,:crystals)) for u in solution.u),
        final_solute=bulk(problem,last(solution.u),:c))
    (;grid,solution,report)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    if "--two-coordinate" in ARGS
        println(RandomFluctuationGRD.run_coupled_tensor().report)
    else
        for bins in (120,240,480)
            println(RandomFluctuationGRD.run(;bins).report)
        end
    end
end
