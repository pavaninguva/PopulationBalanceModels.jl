"""
    StationaryPBMProblem(operator, guess; p=(;), autonomous=false, time=nothing,
                         residual_scale=1, constraints=(), negative_tolerance=1e-12)
    StationaryPBMProblem(system::PBMSystem, populations::NamedTuple; bulk=(;), kwargs...)

Stationary physical balances, with no time derivative or time span. `guess` is a
numerical initial guess, not an initial condition. States retain their usual
meaning: sectional entries are bin integrals; moment entries follow the selected
representation. Use `autonomous=true` to declare time-independent callbacks and
fields, or `time=t` to explicitly freeze prescribed conditions at t. The latter
is an instantaneous stationary problem, not a solution for a changing history.

Positive `residual_scale` (scalar or state-shaped array) divides physical rates
before root finding. Optional `constraints=(row => f, ...)` replace redundant
balance rows by dimensionless equations `f(u,p,t)=0`, e.g. a conserved inventory.
They do not append equations. All original physical balances are checked again
at the root. User bulk accumulators must be removed from the stationary model;
native historical accounting states are automatically excluded from the unknowns.

Residual operators only: direct HOC/characteristic updates and moment boundary
events are unsupported. Callback autonomy is a user declaration, not inferred
from numerical probes. Callbacks must be pure. Nonlinear iterates must remain in
the representation's admissible domain; no clipping or closure repair is applied.
"""
struct StationaryPBMProblem{O,U,P,T,S,C}
    operator::O
    guess::U
    p::P
    time::T
    autonomous::Bool
    residual_scale::S
    constraints::C
    negative_tolerance::Float64
end

_stationary_operator(op)=throw(ArgumentError("$(typeof(op)) has no supported stationary residual; compile a residual-based population instead of a direct evolution method"))
_stationary_operator(::Union{CompiledPBM,CompiledMomentPBM,CompiledPBM2D})=nothing
function _stationary_operator(op::CompiledSystem)
    foreach(_stationary_operator, values(op.system.populations))
end
function _stationary_operator(op::Union{CompiledAxial,CompiledAxialSystem})
    foreach(_stationary_operator, op.local_operators)
end

_stationary_guess(op::Union{CompiledPBM,CompiledMomentPBM},u)=begin
    length(u)==population_state_length(op) || throw(DimensionMismatch("stationary population guess size mismatch"))
    (u isa AbstractVector || size(u)==(population_state_length(op),1)) ||
        throw(DimensionMismatch("a batch population has one state column"))
    reshape(float.(copy(u)),:,1)
end
function _stationary_guess(op::CompiledPBM2D,u)
    size(u)==size(op.grid) || throw(DimensionMismatch("stationary tensor guess must match Grid2D"))
    float.(copy(u))
end
function _stationary_guess(op::CompiledSystem,u)
    u isa AbstractVector && length(u)==op.physical_length ||
        throw(DimensionMismatch("stationary system guess must contain only population and bulk unknowns ($(op.physical_length) entries); omit cumulative histories"))
    float.(copy(u))
end
function _stationary_guess(op::CompiledAxial,u)
    size(u)==(length(op.grid),length(op.domain.grid)) ||
        throw(DimensionMismatch("stationary axial guess must be intrinsic bins × axial cells"))
    float.(copy(u))
end
function _stationary_guess(op::CompiledAxialSystem,u)
    size(u)==(first(op.local_operators).physical_length,length(op.domain.grid)) ||
        throw(DimensionMismatch("stationary axial-system guess must be local physical states × axial cells; omit cumulative histories"))
    float.(copy(u))
end

function StationaryPBMProblem(op,guess; p=NamedTuple(),autonomous=false,time=nothing,
        residual_scale=1,constraints=(),negative_tolerance=1e-12)
    _stationary_operator(op)
    _moment_boundary_events(op) && throw(ArgumentError("stationary moment boundary events are unavailable; use default unbounded-support residual rules or a sectional population"))
    autonomous isa Bool || throw(ArgumentError("autonomous must be Bool"))
    autonomous && time!==nothing && throw(ArgumentError("choose autonomous=true or an explicit frozen time, not both"))
    !autonomous && time===nothing && throw(ArgumentError("declare autonomous=true for stationary conditions, or supply time=t to freeze prescribed callbacks explicitly"))
    (autonomous || time isa Real && isfinite(time)) || throw(ArgumentError("stationary evaluation time must be a finite real value"))
    t=autonomous ? 0.0 : float(time)
    isfinite(negative_tolerance) && negative_tolerance>=0 || throw(ArgumentError("invalid stationary negativity tolerance"))
    u=_stationary_guess(op,guess)
    scale=if residual_scale isa Number
        fill(float(residual_scale),size(u))
    else
        (size(residual_scale)==size(u) || residual_scale isa AbstractVector && length(residual_scale)==length(u)) ||
            throw(DimensionMismatch("residual_scale must match the physical state shape (or be its flattened vector)"))
        reshape(float.(copy(residual_scale)),size(u))
    end
    all(x->isfinite(x) && x>0,scale) || throw(ArgumentError("residual scales must be finite and strictly positive"))
    cs=Tuple(constraints)
    all(c->c isa Pair && first(c) isa Integer && 1<=first(c)<=length(u),cs) ||
        throw(ArgumentError("constraints must be physical linear-index => callback pairs"))
    length(unique(first.(cs)))==length(cs) || throw(ArgumentError("duplicate stationary constraint row"))
    _stationary_check(op,u,negative_tolerance)
    StationaryPBMProblem(op,u,p,t,autonomous,scale,cs,Float64(negative_tolerance))
end

function StationaryPBMProblem(system::PBMSystem,guess::NamedTuple;bulk=NamedTuple(),kwargs...)
    bulk isa NamedTuple || throw(ArgumentError("bulk must be a NamedTuple of scalar guesses"))
    op,u=_pack_system(system,guess,bulk)
    StationaryPBMProblem(op,u[1:op.physical_length];kwargs...)
end

"""
    axial_problem(local_problem::StationaryPBMProblem, domain::AxialDispersion;
                  guess=nothing, axial_scheme=Upwind(), residual_scale=1, constraints=())

Construct a coupled stationary axial problem without creating a transient time
span. The local problem must contain a PBMSystem. By default repeat its physical
guess in every axial cell; native histories remain excluded. The declared
autonomy/frozen time applies to inlet conditions too. Scales and constraints
refer to the full axial problem, not separate local inventory constraints.
"""
function axial_problem(local_problem::StationaryPBMProblem{<:CompiledSystem},domain::AxialDispersion;
        guess=nothing,axial_scheme=Upwind(),residual_scale=1,constraints=())
    op=_compile_axial_system(local_problem.operator,domain,axial_scheme)
    u=guess===nothing ? repeat(local_problem.guess,1,length(domain.grid)) : guess
    StationaryPBMProblem(op,u;p=local_problem.p,autonomous=local_problem.autonomous,
        time=local_problem.autonomous ? nothing : local_problem.time,
        negative_tolerance=local_problem.negative_tolerance,residual_scale,constraints)
end

function remake(prob::StationaryPBMProblem;guess=prob.guess,p=prob.p,
        autonomous=prob.autonomous,time=autonomous ? nothing : prob.time,
        residual_scale=prob.residual_scale,constraints=prob.constraints,
        negative_tolerance=prob.negative_tolerance)
    StationaryPBMProblem(prob.operator,guess;p,autonomous,time,residual_scale,
        constraints,negative_tolerance)
end

function _stationary_check(op,u,tol)
    all(isfinite,u) && all(x->_primal(x)>=-tol,u) ||
        throw(NumericalFailure("stationary population is nonfinite or negative; supply an admissible guess, rescale the balances, use continuation or a different nonlinear algorithm"))
    _check_admissible_state(op,u)
    nothing
end
function _stationary_check(op::CompiledSystem,u,tol)
    all(isfinite,u) || throw(NumericalFailure("nonfinite stationary coupled state"))
    for name in keys(op.ranges)
        _stationary_check(getproperty(op.system.populations,name),population(op,u,name),tol)
    end
    for name in op.system.nonnegative_bulk
        _primal(bulk(op,u,name))>=-tol || throw(NumericalFailure("negative stationary bulk :$name; use an admissible guess, scaling or continuation"))
    end
    nothing
end
function _stationary_check(op::CompiledAxialSystem,u,tol)
    for c in axes(u,2)
        _stationary_check(op.local_operators[c],view(u,:,c),tol)
    end
    nothing
end

# Native ledgers are historical integrals, not physical reservoirs. Evaluate the
# identical full residual with zero history, then retain only physical rows.
_stationary_fullstate(op,u)=u
function _stationary_fullstate(op::CompiledSystem,u)
    full=similar(u,op.length); fill!(full,zero(eltype(u)))
    full[1:op.physical_length].=u
    full
end
function _stationary_fullstate(op::CompiledAxialSystem,u)
    s=first(op.local_operators)
    full=similar(u,s.length,size(u,2)); fill!(full,zero(eltype(u)))
    full[1:s.physical_length,:].=u
    full
end
function _stationary_rates(prob,u,p)
    size(u)==size(prob.guess) || throw(DimensionMismatch("stationary state shape mismatch"))
    _stationary_check(prob.operator,u,prob.negative_tolerance)
    full=_stationary_fullstate(prob.operator,u)
    du=similar(full)
    rhs!(du,full,prob.operator,p,prob.time)
    du
end
_stationary_physical(op,du)=du
_stationary_physical(op::CompiledSystem,du)=view(du,1:op.physical_length)
_stationary_physical(op::CompiledAxialSystem,du)=view(du,1:first(op.local_operators).physical_length,:)

"""
    stationary_residual!(r, u, problem::StationaryPBMProblem, p=problem.p)

Assemble all physical mechanisms and bulk exchanges first, divide by declared
rate scales, then replace explicitly constrained rows. Checks admissibility
before evaluating rates/closures. Shapes match `problem.guess`. Neither the
state nor the problem is mutated; callbacks must also avoid side effects.
"""
function stationary_residual!(r,u,prob::StationaryPBMProblem,p=prob.p)
    size(r)==size(u) || throw(DimensionMismatch("stationary residual/state shape mismatch"))
    du=_stationary_rates(prob,u,p)
    r .= _stationary_physical(prob.operator,du) ./ prob.residual_scale
    for (i,f) in prob.constraints
        r[i]=f(u,p,prob.time)
    end
    all(isfinite,r) || throw(NumericalFailure("nonfinite stationary residual/constraint"))
    nothing
end

"""A stationary state with physical balances, nonlinear retcode and independently
checked convergence. `u` contains no cumulative history; use named accessors.
`residual` is scaled/constrained; `balance_residual` contains original physical
rates, including replaced rows. `solver_result` retains backend diagnostics.
"""
struct StationaryPBMSolution{U,P,R,B,C,S,D}
    u::U
    problem::P
    residual::R
    balance_residual::B
    converged::Bool
    retcode::C
    solver_result::S
    solve_diagnostics::D
end
diagnostics(sol::StationaryPBMSolution)=sol.solve_diagnostics
population(prob::StationaryPBMProblem,u,name::Symbol)=population(prob.operator,u,name)
bulk(prob::StationaryPBMProblem,u,name::Symbol)=bulk(prob.operator,u,name)
stateview(prob::StationaryPBMProblem,u)=stateview(prob.operator,u)
population(sol::StationaryPBMSolution,name::Symbol)=population(sol.problem,sol.u,name)
bulk(sol::StationaryPBMSolution,name::Symbol)=bulk(sol.problem,sol.u,name)
stateview(sol::StationaryPBMSolution)=stateview(sol.problem,sol.u)
axial_stateview(prob::StationaryPBMProblem,u,cell::Integer)=axial_stateview(prob.operator,u,cell)
axial_stateview(sol::StationaryPBMSolution,cell::Integer)=axial_stateview(sol.problem,sol.u,cell)

"""
    nonlinearproblem(problem::StationaryPBMProblem)

Load NonlinearSolve to construct its nonlinear problem. Backend unknowns are
the flattened physical state; no historical ledgers or artificial time is added.
The closure uses the same validation, scales and constraints as solve_stationary.
"""
nonlinearproblem(args...;kwargs...)=throw(ArgumentError("load NonlinearSolve to enable nonlinearproblem"))

"""
    solve_stationary(problem, [algorithm]; abstol=1e-9, reltol=0, maxiters=1000, ...)

Load NonlinearSolve. Default algorithm: RobustMultiNewton. The default
uses finite-difference Jacobians for quadrature closures (whose inversion may
not support dual numbers); other residuals retain the backend's AD default.
An explicitly supplied algorithm controls its own differentiation backend.
The selected physical closure is never changed. A converged backend
retcode alone is insufficient: independently check admissibility, all scaled
physical balances and constrained equations against `abstol + reltol*r0`, where
r0 is the initial scaled/constrained infinity norm. A failed solve returns
`converged=false`; invalid physical/closure evaluations raise an error rather
than clipping. Scaling and admissible starting guesses are the user's choices.
This does not establish uniqueness, stability, spatial accuracy or identifiability.
"""
solve_stationary(args...;kwargs...)=throw(ArgumentError("load NonlinearSolve to enable solve_stationary"))

function accounting_rates(prob::StationaryPBMProblem,u,p=prob.p,t=prob.time)
    t==prob.time || throw(ArgumentError("remake the stationary problem to change its evaluation time"))
    op=prob.operator
    op isa Union{CompiledSystem,CompiledAxialSystem} || throw(ArgumentError("accounting_rates requires a coupled system"))
    _accounting_rate_tree(op,_stationary_rates(prob,u,p))
end
accounting_rates(sol::StationaryPBMSolution)=accounting_rates(sol.problem,sol.u)
cumulative(prob::StationaryPBMProblem,args...)=throw(ArgumentError("stationary states have no cumulative history; use accounting_rates instead"))
