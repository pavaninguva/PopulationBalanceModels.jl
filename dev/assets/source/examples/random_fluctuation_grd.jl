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

end

if abspath(PROGRAM_FILE) == @__FILE__
    for bins in (120,240,480)
        println(RandomFluctuationGRD.run(;bins).report)
    end
end
