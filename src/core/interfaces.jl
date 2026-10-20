"""A numerical representation of one population's internal coordinates."""
abstract type AbstractPopulationRepresentation end
abstract type AbstractMomentBasis <: AbstractPopulationRepresentation end
"""A numerical policy for a single physical PopulationBalanceModels term."""
abstract type AbstractTermMethod end
"""Moment evolution family, including methods with non-moment solver states."""
abstract type AbstractMomentMethod end
"""Moment methods whose solver states are raw moments."""
abstract type AbstractMomentClosure <: AbstractMomentMethod end
"""Extension category for methods evolving quadrature variables directly."""
abstract type AbstractDirectMomentMethod <: AbstractMomentMethod end

function mechanism end
function default_method end
function prepare_term end
function add_term! end
function validate_moment_method end
function prepare_moment_state end
function moment_term! end
function moment_state_length end
function moment_value end
function moment_rate end
function validate_moment_state end
function check_moment_state end

"""Per-class defaults and per-name overrides; per-name choices take precedence."""
struct TermMethods{D,O}
    defaults::D
    by_term::O
end
function TermMethods(;by_term=NamedTuple(),kwargs...)
    by_term isa NamedTuple || throw(ArgumentError("by_term must be a NamedTuple"))
    defaults=(;kwargs...)
    all(m->m isa AbstractTermMethod,values(defaults)) &&
        all(m->m isa AbstractTermMethod,values(by_term)) || throw(ArgumentError("each term method must subtype AbstractTermMethod"))
    TermMethods(defaults,by_term)
end
struct PreparedTerm{T,M,C}
    name::Symbol
    term::T
    method::M
    cache::C
end
function _term_method(plan::TermMethods,name,term,representation)
    haskey(plan.by_term,name) && return getproperty(plan.by_term,name)
    haskey(plan.defaults,mechanism(term)) && return getproperty(plan.defaults,mechanism(term))
    default_method(term,representation)
end
function _validate_plan(plan::TermMethods,model)
    all(n->n in model.names,keys(plan.by_term)) || throw(ArgumentError("by_term contains an unknown model term name"))
    known=(:growth,:birth,:death,:dilution,:nucleation,:aggregation,:breakage,:custom)
    all(n->n in known || any(t->mechanism(t)==n,model.terms),keys(plan.defaults)) || throw(ArgumentError("unknown term-method category"))
end
function _prepare_terms(model,plan,representation;kwargs...)
    _validate_plan(plan,model)
    map(model.names,model.terms) do name,term
        method=_term_method(plan,name,term,representation)
        PreparedTerm(name,term,method,prepare_term(term,method,representation;kwargs...))
    end
end
function prepare_term(term,method,representation;kwargs...)
    throw(ArgumentError("$(typeof(method)) has no implementation for $(typeof(term)) on $(typeof(representation)); extend prepare_term and add_term!"))
end
"""The selected method for every named term, useful for reproducible run records."""
term_methods(op)=NamedTuple{Tuple(item.name for item in op.prepared)}(Tuple(item.method for item in op.prepared))
population_state_length(op)=length(op.grid)
population_moment(op,state,k)=moment(op.grid,state,k)
_mechanism(term)=mechanism(term)

_emptyfields(cell,t)=NamedTuple()
