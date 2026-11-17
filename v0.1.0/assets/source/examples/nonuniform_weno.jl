module NonuniformWENOExample

using PopulationBalanceModels
using SpecialFunctions: erf
using Printf

"""Exact bin integrals for G(x)=x and an initially Gaussian log-size density."""
function reference(grid, t; center=2.0, width=0.3)
    primitive = width*sqrt(pi)/2 .* erf.((log.(grid.edges) .- center .- t)./width)
    diff(primitive)
end

function problem(bins; tspan=(0.0,0.2))
    grid = Grid1D(exp.(range(0.0,4.0;length=bins+1));coordinate=:length)
    model = PBMModel(growth=Growth((x,p,t,ctx)->p.k*x))
    operator = compile(model,grid;
        methods=TermMethods(growth=WENO5()),
        bc=SizeBC(OpenBoundary(),OpenBoundary()))
    PBMProblem(operator,reference(grid,first(tspan)),tspan;p=(k=1.0,))
end

"""
    run(; sizes=(64,128,256))

Compare bin-integrated predictions with characteristic solutions on log meshes.
The initial number density is exp(-((log(x)-2)/0.3)^2)/x. Its characteristic
solution replaces log(x) by log(x)-t in the exponential, leaving 1/x unchanged.
Boundary tails are negligible for this domain and horizon. The conservative
scheme is not positivity preserving: signed undershoots are reported, not clipped.
"""
function run(; sizes=(64,128,256))
    rows = NamedTuple[]
    println("bins    integrated L1 error    observed order    number change    minimum density")
    for bins in sizes
        prob = problem(bins)
        grid = prob.operator.grid
        dt = 0.15minimum(grid.widths./grid.edges[2:end])
        # Separate compilation warmup; this is an accuracy example, not a
        # comparative timing benchmark.
        simulate(remake(prob;tspan=(0.0,dt));method=RK4(),dt,check_nonnegative=false)
        solution = simulate(prob;method=RK4(),dt,check_nonnegative=false)
        final = vec(solution.u[end])
        exact = reference(grid,last(prob.tspan))
        error = sum(abs.(final.-exact))
        order = isempty(rows) ? NaN : log(rows[end].error/error)/log(bins/rows[end].bins)
        row = (;bins,error,order,number_change=sum(final)-sum(prob.u0),
            minimum_density=minimum(final./grid.widths),negative_number=-sum(min.(final,0.0)))
        push!(rows,row)
        @printf("%4d    %.6e           %6.3f          % .3e       % .3e\n",
            bins,error,order,row.number_change,row.minimum_density)
    end
    rows
end

end

if abspath(PROGRAM_FILE)==@__FILE__
    NonuniformWENOExample.run()
end
