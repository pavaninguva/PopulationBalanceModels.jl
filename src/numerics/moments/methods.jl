"""Require an exactly closed supported moment system; reject missing moments."""
struct ExactClosure <: AbstractMomentClosure end
"""One-coordinate QMOM with n positive quadrature nodes and moments 0:2n-1."""
struct QuadratureClosure <: AbstractMomentClosure
    nodes::Int
    function QuadratureClosure(n::Integer)
        n>=1 || throw(ArgumentError("at least one quadrature node is needed"))
        new(n)
    end
end
const QMOM=QuadratureClosure
"""Three-moment assumed lognormal density, integrated with n-point normal quadrature."""
struct LogNormalClosure <: AbstractMomentClosure
    quadrature_order::Int
    nodes::Vector{Float64}
    weights::Vector{Float64}
    function LogNormalClosure(;quadrature_order=16)
        quadrature_order isa Integer && quadrature_order>=2 || throw(ArgumentError("quadrature_order must be at least two"))
        q=_normal_gauss_rule(quadrature_order)
        new(quadrature_order,q.nodes,q.weights)
    end
end


default_method(term,::MomentSet)=ExactMomentRule()
validate_moment_method(::AbstractMomentMethod,model,g)=throw(ArgumentError("extend validate_moment_method for this moment family"))
validate_moment_method(::ExactClosure,model,g::MomentSet)=nothing
function validate_moment_method(m::QuadratureClosure,model,g::MomentSet)
    length(g)==2m.nodes || throw(ArgumentError("QMOM needs exactly moments 0:2n-1"))
end
function validate_moment_method(::LogNormalClosure,model,g::MomentSet)
    g.orders==[0,1,2] || throw(ArgumentError("LogNormalClosure requires moments 0:2"))
end
function prepare_term(term,::ExactMomentRule,g::MomentSet;kwargs...)
    term isa Breakage && (!(term.daughters isa SelfSimilarDaughters) || term.coordinate_map!==nothing || term.daughters.measure_order!=1) &&
        throw(ArgumentError("exact breakage currently uses an additive raw coordinate and measure_order=1"))
    for k in g.orders,dep in _needed_moments(term,k)
        dep in g.orders || throw(ArgumentError("moment equation $k needs untracked moment $dep; select another weak-form rule or closure"))
    end
    nothing
end
