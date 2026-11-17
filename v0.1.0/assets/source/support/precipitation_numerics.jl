# Model-specific transport/depletion recommendation, shared by the tutorial and
# maintained example. This is an admissibility bound, not an error estimator.
function precipitation_step_bound(problem,u,t;fraction=.3)
    isfinite(fraction) && 0 < fraction <= 1 || throw(ArgumentError("fraction must be in (0,1]"))
    operator = problem.operator
    grid = first(operator.local_operators).system.populations.crystals.grid
    dz,dx = minimum(operator.domain.grid.widths),minimum(grid.widths)
    transport = operator.domain.velocity/dz + 2operator.domain.dispersion/dz^2
    du = similar(u)
    rhs!(du,u,operator,problem.p,t)
    p=problem.p
    maxG = p.kg*max(maximum(bulk(problem,u,:c))/p.ceq-1,0)
    h = fraction/(transport+maxG/dx+eps())
    for j in axes(u,2)
        c,dc = bulk(problem,u,:c)[j],bulk(problem,du,:c)[j]
        # This retained allowance affects the step bound, never the state.
        dc < 0 && (h = min(h,fraction*max(c-p.ceq,1e-5)/(-dc)))
    end
    h
end
