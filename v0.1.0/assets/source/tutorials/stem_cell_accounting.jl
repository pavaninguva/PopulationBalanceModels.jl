# Optional diagnostic extension of stem_cells.jl. Include after that walkthrough.
# These scalars have no feedback into cargo growth, division or differentiation.
function cell_counters(ctx,p,t)
    cells = ctx.populations.stem
    (divisions=sum(division_rate(x,p,t,ctx)*cells[i] for (i,x) in enumerate(grid.centers)),
     differentiated_cells=sum(differentiation_rate(x,p,t,ctx)*cells[i] for (i,x) in enumerate(grid.centers)),
     differentiated_cargo=sum(x*differentiation_rate(x,p,t,ctx)*cells[i] for (i,x) in enumerate(grid.centers)))
end
accounted_system = PBMSystem((stem=operator,);bulk_rhs=cell_counters,
    exchanges=(ConservativeExchange(:stem,:synthesis_balance,(x,p,t,ctx)->x;
        mechanisms=(:growth,)),),
    accounting=(CumulativeAccounting(:stem;properties=(cargo=1,)),),
    nonnegative_bulk=(:divisions,:differentiated_cells,:differentiated_cargo))
accounted_problem = PBMProblem(accounted_system,(stem=initial,),problem.tspan;p,
    bulk0=(divisions=0.0,differentiated_cells=0.0,differentiated_cargo=0.0,synthesis_balance=0.0))

# Use the same time schedule when comparing the two population trajectories.
# accounted = simulate(accounted_problem;method=SSPRK33(),dt,
#     negative_tolerance=0.0,saveat=[0.0,10.0,20.0])
# observe(BulkObservable(:differentiated_cells),accounted) # Counts AT EXIT, not a second population.
# cumulative(accounted_problem,last(accounted.u),:stem,:breakage_underflow,:cargo)
