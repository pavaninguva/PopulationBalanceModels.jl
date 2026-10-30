struct CompiledMomentPBM{M,G,C,F,P,B}
    model::M
    grid::G
    closure::C
    fields::F
    prepared::P
    boundary::B
end
_system_population(::CompiledMomentPBM)=true


"""
    compile(model, MomentSet(0:K); moment_method=nothing, fields=...)

Evaluate each supported contribution analytically from tracked moments, using
the selected reconstruction only for missing moments or generic population
integrals. `QMOM()` and other node-based specifications infer resolution from
the basis. Without a keyword, closed supported models use ExactClosure;
otherwise compatible bases use RankAdaptiveQMOM. Odd raw-moment counts never
silently select an assumed density. Inspect `moment_plan(operator)`.

`moment_method=ExactClosure()` strictly requires analytical evolution. Legacy
`method`/`closure` aliases and explicit TermMethods overrides remain available;
supply only one family keyword. Boundary events still require MeasureEuler.
"""
function compile(model::PBMModel,g::MomentSet;moment_method=nothing,closure=nothing,method=nothing,methods=TermMethods(),fields=_emptyfields,boundary=MomentBoundary(),evolution=nothing)
    _validate_size_diffusion(model,g,nothing,BatchDomain(),evolution)
    evolution===nothing || throw(ArgumentError("direct sectional HOC/characteristic evolution requires Grid1D or Grid2D bin integrals; moment populations use their moment integration policies"))
    boundary isa MomentBoundary || throw(ArgumentError("boundary must be MomentBoundary"))
    choices=filter(!isnothing,(moment_method,method,closure))
    length(choices)<=1 || throw(ArgumentError("conflicting moment-family keywords: supply only moment_method (legacy aliases: method or closure)"))
    methods isa TermMethods || throw(ArgumentError("methods must be TermMethods"))
    _validate_plan(methods,model)
    residual_boundary=boundary.left===:error && boundary.right===:error && boundary.lower==0 && isinf(boundary.upper)
    selected=isempty(choices) ? (residual_boundary ? _default_moment_method(model,g,methods) :
        _resolve_moment_method(RankAdaptiveQMOM(),g)) : only(choices)
    selected isa AbstractMomentMethod || throw(ArgumentError("invalid moment method"))
    selected=_resolve_moment_method(selected,g)
    # Legacy ExactClosure + explicit MeasureEuler reconstruction remains usable
    # for event policies. The canonical exact-only request cannot certify a trace.
    moment_method isa ExactClosure && !residual_boundary && throw(ArgumentError(
        "ExactClosure cannot resolve the declared finite/absorbing moment boundary; no boundary trace is determined by global moments. Select RankAdaptiveQMOM with native MeasureEuler for supported boundary events"))
    validate_moment_method(selected,model,g)
    for term in model.terms
        if term isa Aggregation
            g.coordinate in (:volume,:mass) || term.coordinate_map!==nothing || throw(ArgumentError("aggregation needs an additive coordinate or AdditiveCoordinate map"))
        end
        if term isa Breakage && term.coordinate_map===nothing
            g.coordinate in (:volume,:mass) || throw(ArgumentError("breakage needs an additive coordinate or AdditiveCoordinate map"))
        end
    end
    prepared=_prepare_moment_terms(model,g,selected,methods)
    CompiledMomentPBM(model,g,selected,fields,prepared,boundary)
end


function prepare_term(term,::QuadratureMomentRule,g::MomentSet;kwargs...)
    term isa Union{Growth,Death,Dilution,Aggregation,Breakage,Nucleation,MomentSource} || throw(ArgumentError("no quadrature weak form for this term"))
    nothing
end
moment_state_length(::AbstractMomentClosure,g)=length(g)
moment_state_length(m::AbstractDirectMomentMethod,g)=throw(ArgumentError("direct moment families must define their state length"))
population_state_length(op::CompiledMomentPBM)=moment_state_length(op.closure,op.grid)
moment_value(::AbstractMomentClosure,g,state,k)=moment(g,vec(state),k)
moment_value(m::AbstractDirectMomentMethod,g,state,k)=throw(ArgumentError("direct moment families must implement moment_value"))
moment_rate(::AbstractMomentClosure,g,state,derivative,k)=moment(g,vec(derivative),k)
moment_rate(m::AbstractDirectMomentMethod,g,state,derivative,k)=throw(ArgumentError("direct moment families must implement the moment chain rule"))
population_moment(op::CompiledMomentPBM,state,k)=moment_value(op.closure,op.grid,state,k)
moment(ctx::NamedTuple,k)=population_moment(ctx.operator,ctx.N,k)


function _validate_population_state(op::CompiledMomentPBM,u)
    validate_moment_state(op.closure,op.grid,u)
    b=op.boundary
    # Necessary support inequalities, not a certificate for a singular
    # truncated moment problem. MeasureEuler additionally reconstructs every
    # moment by a positive measure inside these boundaries.
    for k in op.grid.orders[1:end-1]
        a=population_moment(op,u,k);next=population_moment(op,u,k+1)
        lower=b.lower*a
        next>=lower-1e-10*max(abs(next),abs(lower)) ||
            throw(ArgumentError("initial moments violate the declared lower support"))
        if isfinite(b.upper)
            upper=b.upper*a
            next<=upper+1e-10*max(abs(next),abs(upper)) ||
                throw(ArgumentError("initial moments violate the declared upper support"))
        end
    end
    nothing
end
function validate_moment_state(method::AbstractMomentClosure,g,u)
    realizability(g,_primal.(u)).valid || throw(ArgumentError("initial moments fail nonnegative-support Hankel checks"))
    method isa Union{QuadratureClosure,LogNormalClosure} && quadrature(g,u,method)
    nothing
end
validate_moment_state(::AbstractDirectMomentMethod,g,u)=throw(ArgumentError("direct moment family must validate its solver state"))
function check_moment_state(method::AbstractMomentClosure,g,u,tolerance)
    isinf(tolerance) && return
    # Relative Hankel tests are scale-sensitive near a newly empty population.
    # Do not mutate/project moments to enforce them.
    realizability(g,_primal.(u);rtol=max(1e-10,tolerance)).valid || throw(NumericalFailure("moments fail Hankel positivity during integration; reduce dt or change closure"))
end
check_moment_state(::AbstractDirectMomentMethod,g,u,tolerance)=throw(ArgumentError("direct moment family must check its solver state"))
_check_population_state(op::CompiledMomentPBM,u,tolerance)=check_moment_state(op.closure,op.grid,u,tolerance)
function PBMProblem(op::CompiledMomentPBM,u0,tspan;p=NamedTuple())
    μ=reshape(float.(copy(vec(u0))),:,1)
    length(μ)==population_state_length(op) || throw(DimensionMismatch("moment-method state size mismatch"))
    _validate_population_state(op,μ)
    length(tspan)==2 && all(isfinite,tspan) && tspan[2]>tspan[1] || throw(ArgumentError("invalid time span"))
    PBMProblem(op,μ,(float(tspan[1]),float(tspan[2])),p)
end
function _check_problem_state(prob::PBMProblem{O},u,tolerance) where {O<:CompiledMomentPBM}
    _checkstate(u,tolerance,"moment-method state",
        "reduce dt; inspect closure admissibility, conditioning and weak-form quadrature")
    _check_population_state(prob.operator,u,tolerance)
end


function _moment_terms!(dμ,μ,op,p,t,ctx;mechanisms=nothing)
    b=op.boundary
    (b.left===:error && b.right===:error && b.lower==0 && isinf(b.upper)) ||
        throw(ArgumentError("moment boundary events require native MeasureEuler; no residual boundary trace is inferred"))
    q=_prepare_checked_moment_state(op.closure,op.grid,μ,p,t,ctx)
    moments=_MomentView(op.closure,op.grid,μ,q)
    for item in op.prepared
        mechanisms!==nothing && !(mechanism(item.term) in mechanisms) && continue
        if item.method isa _AutomaticMomentRule
            _automatic_term!(dμ,μ,item,moments,op,p,t,ctx)
        else
            moment_term!(dμ,μ,item.term,item.method,item.cache.prepared,op.closure,q,op,p,t,ctx)
        end
    end
end
function _check_admissible_state(op::CompiledMomentPBM,u)
    _validate_population_state(op,u)
end
function rhs!(dμ::AbstractMatrix,μ::AbstractMatrix,op::CompiledMomentPBM,p,t;context=nothing)
    size(dμ)==size(μ)==(population_state_length(op),1) || throw(DimensionMismatch("moment state size mismatch"))
    fill!(dμ,zero(eltype(dμ)))
    _moment_terms!(view(dμ,:,1),view(μ,:,1),op,p,t,_context(op,μ,1,t,context))
    all(isfinite,dμ) || throw(NumericalFailure("nonfinite moment derivative"))
    nothing
end
