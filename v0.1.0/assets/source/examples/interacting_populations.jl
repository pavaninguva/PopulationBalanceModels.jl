using PopulationBalanceModels

gridA=Grid1D(0.5:1:10.5)
gridB=Grid1D([0.5,1.5,3.5,5.5,8.5,11.5])
A=compile(PBMModel(Nucleation((p,t,ctx)->p.k*ctx.bulk.c,2.0)),gridA)
B=compile(PBMModel(),gridB)
system=PBMSystem((precursors=A,products=B);
    interactions=(PopulationTransfer(:precursors,:products,(x,p,t,ctx)->p.transfer),),
    exchanges=(ConservativeExchange(:precursors,:c,(x,p,t,ctx)->x),),
    nonnegative_bulk=(:c,))
problem=PBMProblem(system,(precursors=zeros(length(gridA)),products=zeros(length(gridB))),
    (0.0,3.0);bulk0=(c=5.0,),p=(k=0.1,transfer=0.3))
solution=simulate(problem;dt=0.01,saveat=0:0.25:3)
final=stateview(problem,solution.u[end])
println("Final dissolved concentration: ",final.bulk.c)
println("Precursor/product numbers: ",(sum(final.populations.precursors),sum(final.populations.products)))
println("Conserved total: ",final.bulk.c+moment(gridA,final.populations.precursors,1)+moment(gridB,final.populations.products,1))
