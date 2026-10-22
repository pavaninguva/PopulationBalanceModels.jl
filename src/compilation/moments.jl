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
    compile(model, MomentSet(0:K); closure=ExactClosure(), fields=...)

Exact rules: polynomial growth/removal, constant-kernel aggregation in additive
coordinates, self-similar polynomial-rate breakage, point creation, MomentSource.
QMOM additionally evaluates general growth, removal, aggregation and breakage
through their weak forms. It does not infer source integrals or boundary losses.
"""
function compile(model::PBMModel,g::MomentSet;closure=ExactClosure(),method=nothing,methods=TermMethods(),fields=_emptyfields,boundary=MomentBoundary())
    boundary isa MomentBoundary || throw(ArgumentError("boundary must be MomentBoundary"))
    selected=method===nothing ? closure : method
    selected isa AbstractMomentMethod || throw(ArgumentError("invalid moment method"))
    methods isa TermMethods || throw(ArgumentError("methods must be TermMethods"))
    validate_moment_method(selected,model,g)
    for term in model.terms
        if term isa Aggregation
            g.coordinate in (:volume,:mass) || term.coordinate_map!==nothing || throw(ArgumentError("aggregation needs an additive coordinate or AdditiveCoordinate map"))
        end
        if term isa Breakage && term.coordinate_map===nothing
            g.coordinate in (:volume,:mass) || throw(ArgumentError("breakage needs an additive coordinate or AdditiveCoordinate map"))
        end
    end
    # A representation's method chooses the default weak-form rule; users may
    # override any named term, e.g. exact affine growth alongside QMOM collisions.
    rule=selected isa ExactClosure ? ExactMomentRule() : QuadratureMomentRule()
    defaults=NamedTuple{Tuple(unique(mechanism.(model.terms)))}(Tuple(rule for _ in unique(mechanism.(model.terms))))
    plan=TermMethods(;by_term=methods.by_term,merge(defaults,methods.defaults)...)
    prepared=_prepare_terms(model,plan,g)
    selected isa ExactClosure && any(item->item.method isa QuadratureMomentRule,prepared) && throw(ArgumentError("ExactClosure does not provide quadrature; select a reconstruction method"))
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
    _checkstate(u,tolerance); _check_population_state(prob.operator,u,tolerance)
end


function _moment_terms!(dμ,μ,op,p,t,ctx;mechanisms=nothing)
    b=op.boundary
    (b.left===:error && b.right===:error && b.lower==0 && isinf(b.upper)) ||
        throw(ArgumentError("moment boundary events require native MeasureEuler; no residual boundary trace is inferred"))
    validate_moment_state(op.closure,op.grid,μ)
    q=prepare_moment_state(op.closure,op.grid,μ,p,t,ctx)
    for item in op.prepared
        mechanisms!==nothing && !(mechanism(item.term) in mechanisms) && continue
        moment_term!(dμ,μ,item.term,item.method,item.cache,op.closure,q,op,p,t,ctx)
    end
end
function rhs!(dμ::AbstractMatrix,μ::AbstractMatrix,op::CompiledMomentPBM,p,t;context=nothing)
    size(dμ)==size(μ)==(population_state_length(op),1) || throw(DimensionMismatch("moment state size mismatch"))
    fill!(dμ,zero(eltype(dμ)))
    _moment_terms!(view(dμ,:,1),view(μ,:,1),op,p,t,_context(op,μ,1,t,context))
    all(isfinite,dμ) || throw(NumericalFailure("nonfinite moment derivative"))
    nothing
end
