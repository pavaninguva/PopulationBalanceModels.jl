abstract type AbstractPopulationDomain end
struct BatchDomain <: AbstractPopulationDomain end

struct NumericalFailure <: Exception
    message::String
end
Base.showerror(io::IO, e::NumericalFailure) = print(io, e.message)

_nonnegative(x, what) = (isfinite(x) && x >= 0 ? x :
    throw(DomainError(x, "$what must be finite and nonnegative")))
