using PopulationBalanceModels

g=Grid1D(range(0,2;length=41);coordinate=:length)
J=(p,t,ctx)->2.0
growth=Growth((x,p,t,ctx)->0.2)

# Same total rate, three different physical representations. Never add all
# three to represent a single nucleation mechanism.
boundary=compile(PBMModel(growth),g;
    bc=SizeBC(InflowFlux(J),NoFlux()))
point=compile(PBMModel(growth+Nucleation(J,0.1)),g)
gaussian=compile(PBMModel(growth+GaussianNucleation(J;
    center=0.1,width=0.015,truncation=:renormalize)),g)

for (label,op) in ((:boundary,boundary),(:point,point),(:gaussian,gaussian))
    problem=PBMProblem(op,zeros(length(g)),(0.0,0.2))
    solution=simulate(problem;dt=0.005)
    println((representation=label,number=sum(solution.u[end]),
        resolved_M1=only(moment(g,solution.u[end],1))))
end
# IntegratedBirth accepts any user-defined bin integral. DistributedNucleation
# remains a midpoint-quadrature alternative for a sufficiently resolved shape.
