using PopulationBalanceModels

# A smooth periodic problem verifies the transport scheme, not a physical
# periodic particle-size boundary. Upwind supports nonuniform bins; WENO5 here
# requires uniform bins and is fifth order in the interior only.
grid=Grid1D(range(0,2π;length=81))
model=PBMModel(Growth((v,p,t,ctx)->p.velocity))
op=compile(model,grid;growth_scheme=WENO5(epsilon=1e-10),
    bc=SizeBC(Periodic(),Periodic()))
initial=2 .*grid.widths .+cos.(grid.edges[1:end-1]).-cos.(grid.edges[2:end])
problem=PBMProblem(op,initial,(0.0,0.5);p=(velocity=0.7,))
solution=simulate(problem;dt=0.002,saveat=[0.0,0.5])
exact=2 .*grid.widths .+cos.(grid.edges[1:end-1].-0.35).-cos.(grid.edges[2:end].-0.35)
println("Maximum density error: ",maximum(abs.(vec(solution.u[end]).-exact)./grid.widths))
println("Number drift: ",sum(solution.u[end])-sum(initial))
