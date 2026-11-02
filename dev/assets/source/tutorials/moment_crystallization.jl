# Minimal moment crystallization tutorial.
# Generated from maintained scientific source sections and a short run block.
# No benchmark exports or retained files are written. See the case-study page.

using PopulationBalanceModels
using SpecialFunctions: gamma

# From examples/qamar_moment_benchmarks.jl (geometry).
const volume_map=AdditiveCoordinate(x->x^3,cbrt)
const daughter_law=UniformVolumeDaughters()

# From examples/qamar_moment_benchmarks.jl (problem4).
function method_basis(method,nodes=3)
    # A specification such as QMOM() leaves resolution to the supplied basis.
    # This recipe chooses that basis from `nodes`; explicit family counts win.
    capacity=method isa RankAdaptiveQMOM ? something(method.max_nodes,nodes) :
        method isa Union{QuadratureClosure,EQMOM,DQMOM} ? something(method.nodes,nodes) : nodes
    lastorder=method isa LogNormalClosure ? 2 : method isa EQMOM ? 2capacity : 2capacity-1
    MomentSet(0:lastorder;coordinate=:length)
end

function initial_state(method,basis,raw)
    if method isa DQMOM
        q=quadrature(basis,raw,QMOM(method.nodes))
        return quadrature_state(method,q.nodes,q.weights)
    end
    raw
end

"""Problem 4: constant aggregation, volume-proportional breakage and exponential volume distribution."""
function problem4(;nodes=3,method=QMOM(nodes),tf=50.)
    basis=method_basis(method,nodes)
    model=PBMModel(
        aggregation=Aggregation(ConstantKernel(1.);coordinate_map=volume_map),
        breakage=Breakage((x,p,t,ctx)->x^3,daughter_law;coordinate_map=volume_map))
    mu=exact4(0.,basis.orders)
    operator=compile(model,basis;moment_method=method)
    PBMProblem(operator,initial_state(operator.closure,basis,mu),(0.,tf))
end

# Number obeys dμ0/dt = a0 μ3 - β0 μ0²/2; μ3=V0 is constant.
# This also disambiguates the C0 symbol in the printed Eq.42: here β0=C0=1.
function exact4(t,orders=0:5)
    N0=1.;V0=.001;beta=1.;a0=1.
    steady=sqrt(2a0*V0/beta)/N0
    phi=steady*(1+steady*tanh(steady*beta*N0*t/2))/(steady+tanh(steady*beta*N0*t/2))
    [N0*phi*(V0/(N0*phi))^(k/3)*gamma(1+k/3) for k in orders]
end

function tutorial_problem(; method=QMOM())
    # `nodes` chooses the basis when a family omits its node count.
    # Each family's required orders and DQMOM initial-state conversion are above.
    problem4(; nodes=3, method, tf=50.0)
end

function reported_moment(op, u, k)
    k in op.grid.orders && return moment_value(op.closure, op.grid, u, k)
    q = quadrature(op.grid, u, op.closure)
    reconstructed_moment(op.closure, op.grid, u, q, k)
end

function main(; method=QMOM())
    problem = tutorial_problem(; method)
    @show moment_plan(problem.operator)
    solution = simulate(problem; method=SSPRK33(), dt=0.02, saveat=0.0:1.0:50.0)
    op = problem.operator
    # Lognormal stores only M0:M2; its M3 is reconstructed, not conserved state.
    M(k,u) = reported_moment(op,u,k)
    error = [abs(M(1,u)-exact4(t)[2])/abs(exact4(t)[2])
        for (t,u) in zip(solution.t, solution.u)]
    material_kind = 3 in op.grid.orders ? "tracked" : "reconstructed"
    @show maximum(error) material_kind M(3,problem.u0)/0.001
    @show maximum(abs(M(3,u)/0.001-1) for u in solution.u)
    solution
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
