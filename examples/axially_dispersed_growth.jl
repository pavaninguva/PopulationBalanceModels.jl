module AxiallyDispersedGrowth
using PopulationBalanceModels

function run(;output=joinpath(@__DIR__,"..","docs","src","assets","axial_growth"))
    sizes=Grid1D(range(0,2;length=81);coordinate=:length)
    axial=AxialGrid(range(0,1;length=61))
    # Feed is integrated over particle-size bins, in particles per fluid volume.
    feed=exp.(-((sizes.centers.-0.3)./0.08).^2).*sizes.widths
    reactor=AxialDispersion(axial;velocity=0.2,dispersion=0.005,inlet=(p,t)->feed)
    model=PBMModel(growth=Growth(AffineGrowth(0.08,0.0)))
    operator=compile(model,sizes;domain=reactor,methods=TermMethods(growth=MUSCL()),
        bc=SizeBC(OpenBoundary(),OpenBoundary()))
    initial=zeros(length(sizes),length(axial))
    problem=PBMProblem(operator,initial,(0.0,5.0))
    dt=suggest_dt(operator,initial,(;),0.0;cfl=0.4)
    solution=simulate(problem;dt,method=SSPRK33(),saveat=[0.0,1.0,3.0,5.0])
    mkpath(output)
    open(joinpath(output,"solution.csv"),"w") do io
        println(io,"time,position,size,density")
        for (t,N) in zip(solution.t,solution.u),c in 1:length(axial)
            f=density(sizes,axial_population(operator,N,c))
            for i in 1:length(sizes);println(io,join((t,axial.centers[c],sizes.centers[i],f[i]),','));end
        end
    end
    println((steps=solution.steps,dt,minimum=minimum(last(solution.u)),output))
    solution
end
if abspath(PROGRAM_FILE)==@__FILE__;run();end
end
