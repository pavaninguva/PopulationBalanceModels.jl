"""An AD incompatibility, retaining the original exception. Select `derivatives=:finite_difference` explicitly for a backend that cannot propagate dual numbers; no silent fallback is made."""
struct DifferentiationFailure{E} <: Exception
    cause::E
end
Base.showerror(io::IO,e::DifferentiationFailure)=begin
    print(io,"Automatic differentiation failed. Use generic numeric storage/callbacks or explicitly select derivatives=:finite_difference. Cause: ")
    showerror(io,e.cause)
end
function _derivative_choice(method)
    method in (:auto,:forwarddiff,:finite_difference) || throw(ArgumentError("derivatives must be :auto, :forwarddiff or :finite_difference"))
    method
end
function _ad_evaluate(f)
    try
        f()
    catch e
        e isa Union{MethodError,InexactError,TypeError} || rethrow()
        throw(DifferentiationFailure(e))
    end
end
function _differentiate_gradient(f,z;derivatives=:auto,step=cbrt(eps(Float64)))
    _derivative_choice(derivatives)
    g=derivatives===:finite_difference ? _fd_gradient(f,z;step) : _ad_evaluate(()->ForwardDiff.gradient(f,z))
    all(isfinite,g) || throw(NumericalFailure("nonfinite gradient; inspect differentiability, posterior support and the forward solve"))
    g
end
function _fd_jacobian(f,z;step=cbrt(eps(Float64)))
    isfinite(step) && step>0 || throw(ArgumentError("invalid derivative step"))
    J=zeros(length(f(z)),length(z))
    for i in eachindex(z)
        h=step*max(1,abs(z[i]));a=copy(z);b=copy(z);a[i]+=h;b[i]-=h
        J[:,i]=(f(a)-f(b))/(2h)
    end
    all(isfinite,J) || throw(NumericalFailure("nonfinite derivative verification stencil"))
    J
end
function _differentiate_jacobian(f,z;derivatives=:auto,step=cbrt(eps(Float64)))
    _derivative_choice(derivatives)
    J=derivatives===:finite_difference ? _fd_jacobian(f,z;step) : _ad_evaluate(()->ForwardDiff.jacobian(f,z))
    all(isfinite,J) || throw(NumericalFailure("nonfinite prediction sensitivities"))
    J
end
function _differentiate_hessian(f,z;derivatives=:auto,step=eps(Float64)^(1/4))
    _derivative_choice(derivatives)
    H=derivatives===:finite_difference ? _fd_hessian(f,z;step) : _ad_evaluate(()->ForwardDiff.hessian(f,z))
    all(isfinite,H) || throw(NumericalFailure("nonfinite posterior Hessian"))
    H
end
# Promote fixed initial states before an AD solve. Parameter-dependent initial
# states retain their derivatives; fixed states acquire zero partials.
function _inference_problem(pb,z)
    T=promote_type(eltype(pb.u0),eltype(z))
    T===eltype(pb.u0) ? pb : remake(pb;u0=T.(pb.u0))
end
_is_stationary(problem)=false
_numeric_type(x::Real)=typeof(x)
_numeric_type(x::Union{Tuple,NamedTuple,AbstractArray})=foldl(promote_type,(_numeric_type(v) for v in x);init=Float64)
_numeric_type(x)=Float64

"""
    SciMLSensitivitySolver(algorithm; sensitivity=:auto, reltol=1e-7, abstol=1e-9, kwargs...)

Optional inference solver enabled by SciMLBase, SciMLSensitivity and ReverseDiff
(loaded by SciMLSensitivity). Automatic
selection uses continuous forward sensitivities for at most 16 coordinates and
an interpolating adjoint otherwise. Explicit :forward/:adjoint choices are
available. Includes parameter-dependent initial states and direct likelihood,
response and noise dependence. ODE parameters are the full inference vector.
Supports differentiable fixed-representation residuals; changing grid topology,
direct shifts, eigen-based QMOM inversion and user callbacks may be ineligible.
Hessians use nested ForwardDiff through the primal solve, not an unverified
second derivative of an adjoint. Finite differences are an explicit alternative.
"""
struct SciMLSensitivitySolver{A,K}
    algorithm::A
    sensitivity::Symbol
    options::K
end
function SciMLSensitivitySolver(algorithm;sensitivity=:auto,reltol=1e-7,abstol=1e-9,kwargs...)
    sensitivity in (:auto,:forward,:adjoint) || throw(ArgumentError("sensitivity must be :auto, :forward or :adjoint"))
    haskey(kwargs,:callback) && throw(ArgumentError("event callbacks need sensitivity jump rules and are not supported by this adapter"))
    any(k->k in (:saveat,:save_everystep,:dense,:sensealg),keys(kwargs)) && throw(ArgumentError("saving and sensitivity settings are controlled by the inference adapter"))
    SciMLSensitivitySolver(algorithm,sensitivity,(;reltol,abstol,kwargs...))
end
(s::SciMLSensitivitySolver)(pb,times)=_sensitivity_solve(s,pb,times)
_sensitivity_solve(s,pb,times)=throw(ArgumentError("load SciMLBase and SciMLSensitivity to use SciMLSensitivitySolver"))

_primal(x)=x
_primal(x::ForwardDiff.Dual)=_primal(ForwardDiff.value(x))
