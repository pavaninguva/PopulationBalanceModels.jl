"""
    PCEBasis(family, diagonal, offdiagonal; shift=0, scale=1)

Probability-orthonormal univariate basis, with `φ₀ = 1` and recurrence
`z*φₖ = bₖ₊₁*φₖ₊₁ + aₖ*φₖ + bₖ*φₖ₋₁`, where
`z = (x-shift)/scale`, `diagonal[k+1] = aₖ` and `offdiagonal[k] = bₖ`.
There must be one more diagonal than offdiagonal entry; the latter are positive.

Extension implementations of [`pce_basis`](@ref) return this description.
The coefficients must be orthonormal under the **probability law** of the
declared distribution, not an unnormalized weight. This is an extension contract;
the constructor checks finite coefficients, not the asserted probability law.
"""
struct PCEBasis
    family::Symbol
    diagonal::Vector{Float64}
    offdiagonal::Vector{Float64}
    shift::Float64
    scale::Float64
    function PCEBasis(family::Symbol, diagonal, offdiagonal; shift=0, scale=1)
        a, b = Float64.(collect(diagonal)), Float64.(collect(offdiagonal))
        !isempty(a) && length(a) == length(b) + 1 ||
            throw(ArgumentError("PCE recurrence needs one more diagonal than offdiagonal coefficient"))
        all(isfinite, a) && all(x -> isfinite(x) && x > 0, b) ||
            throw(ArgumentError("PCE recurrence must be finite with positive offdiagonal coefficients"))
        isfinite(shift) && isfinite(scale) && scale > 0 ||
            throw(ArgumentError("PCE basis shift/scale must be finite, with scale > 0"))
        new(family, a, b, Float64(shift), Float64(scale))
    end
end

"""
    pce_basis(distribution, degree)

Return a [`PCEBasis`](@ref) through `degree`. Built-in probability-orthonormal
families are Normal/Hermite, Uniform/Legendre and Beta/Jacobi. For
`Beta(α,β)`, `z=2x-1` has Jacobi weight `(1-z)^(β-1)*(1+z)^(α-1)`.
An affine physical interval can instead be specified by `UncertainInputs.map`.

Extend this function for another `Distributions.UnivariateDistribution` by
returning a recurrence of the requested degree. Regression and Gaussian
quadrature use this same hook. Distributions must also support ordinary sampling,
`insupport`, and `quantile` when using an LHS design. Arbitrary joint/posterior
laws require an explicit reference transformation; no joint basis is inferred.
"""
function pce_basis(distribution, degree::Integer)
    throw(ArgumentError("no PCE basis for $(typeof(distribution)); use Normal/Uniform/Beta reference marginals and an explicit map, or extend pce_basis"))
end

function _pce_degree(degree)
    degree >= 0 || throw(ArgumentError("PCE basis degree must be nonnegative"))
    Int(degree)
end

function pce_basis(d::Distributions.Normal, degree::Integer)
    p = _pce_degree(degree)
    μ, σ = Distributions.params(d)
    PCEBasis(:hermite, zeros(p+1), sqrt.(Float64.(1:p)); shift=μ, scale=σ)
end

function pce_basis(d::Distributions.Uniform, degree::Integer)
    p = _pce_degree(degree)
    a, b = Distributions.params(d)
    # Orthonormal under dz/2, not the unnormalized Legendre weight dz.
    offdiagonal = [k / sqrt((2.0k-1)*(2.0k+1)) for k in 1:p]
    PCEBasis(:legendre, zeros(p+1), offdiagonal; shift=a/2+b/2, scale=(b-a)/2)
end

function pce_basis(d::Distributions.Beta, degree::Integer)
    p = _pce_degree(degree)
    α, β = Distributions.params(d)
    s = α + β
    isfinite(s) && α > 0 && β > 0 || throw(ArgumentError("Beta PCE needs finite positive shape parameters"))
    a, b = zeros(p+1), zeros(p)
    a[1] = (α-β)/s
    for k in 1:p
        a[k+1] = ((α-β)/(2k+s-2))*((s-2)/(2k+s))
        # The k=1 expression is the exact standard deviation of z=2x-1.
        # Cancelling analytically avoids a removable 0/0 when α+β=1.
        b[k] = k == 1 ? 2sqrt((α/s)*(β/s)/(s+1)) :
            2/(2k+s-2)*sqrt(k*(k+α-1)/(2k+s-3) *
                           (k+β-1)*(k+s-2)/(2k+s-1))
    end
    PCEBasis(:jacobi, a, b; shift=0.5, scale=0.5)
end

function _pce_gauss_rule(basis)
    nodes = eigvals(SymTridiagonal(basis.diagonal,basis.offdiagonal))
    weights = map(nodes) do z
        previous, current, squares, logscale = 0.0, 1.0, 1.0, 0.0
        for k in eachindex(basis.offdiagonal)
            next = ((z-basis.diagonal[k])*current -
                    (k == 1 ? 0.0 : basis.offdiagonal[k-1]*previous))/basis.offdiagonal[k]
            previous, current = current, next
            if max(abs(previous),abs(current)) > 1e100
                previous /= 1e100
                current /= 1e100
                squares /= 1e200
                logscale += log(1e100)
            end
            squares += current^2
        end
        exp(-2logscale-log(squares))
    end
    # Christoffel weights avoid the lost tiny eigenvector components that can
    # corrupt tail contributions in a high-order Hermite rule. No weight floor
    # or renormalization changes the probability measure.
    (;nodes=basis.shift .+ basis.scale.*nodes,weights,family=basis.family)
end

function _pce_marginals(inputs)
    inputs.kind === :independent || throw(ArgumentError(
        "PCE requires independent reference marginals; supply an explicit map for physical dependence, not an empirical posterior basis"))
    marginals = Tuple(values(inputs.source))
    for d in marginals
        pce_basis(d, 0) # Validate availability and distribution scaling early.
    end
    marginals
end

function _pce_univariate(basis::PCEBasis, points)
    degree = length(basis.offdiagonal)
    V = ones(length(points), degree+1)
    for i in eachindex(points)
        z = (points[i]-basis.shift)/basis.scale
        degree == 0 && continue
        V[i,2] = (z-basis.diagonal[1])/basis.offdiagonal[1]
        for k in 1:degree-1
            V[i,k+2] = ((z-basis.diagonal[k+1])*V[i,k+1] -
                        basis.offdiagonal[k]*V[i,k])/basis.offdiagonal[k+1]
        end
    end
    all(isfinite, V) || throw(ArgumentError("nonfinite PCE basis evaluation; reduce degree or check the reference scaling"))
    V
end

_pce_bases(inputs, degree) = Tuple(pce_basis(d,degree) for d in _pce_marginals(inputs))

function _pce_basis(inputs, indices, X; bases=_pce_bases(inputs,maximum(indices)))
    tables = [_pce_univariate(basis,view(X,j,:)) for (j,basis) in enumerate(bases)]
    V = ones(size(X,2), size(indices,1))
    for term in axes(indices,1), j in axes(indices,2), i in axes(X,2)
        V[i,term] *= tables[j][i,indices[term,j]+1]
    end
    V
end

"""
    pce_quadrature(inputs; order, max_points=100_000)

Tensor-product Gaussian nodes and **probability** weights for the declared
independent reference marginals. `order` is a positive node count per coordinate,
or a tuple of counts. An order-q rule integrates univariate polynomials through
degree `2q-1`. Golub–Welsch nodes and Christoffel weights use the same
orthonormal recurrence as regression, retaining small Gaussian tail weights.
The product size is checked against `max_points` before allocation.

The returned recorded design has `samples`, `weights`, names and coordinates;
use `PolynomialChaos(p; construction=:projection)` with [`fit_pce`](@ref).
It changes neither the input law nor the physical input map. Increase quadrature
order separately from polynomial degree and validate nonpolynomial outputs.
This is full tensor quadrature, not a sparse-grid or adaptive construction.
"""
function pce_quadrature(inputs::UncertainInputs; order, max_points=100_000)
    marginals = _pce_marginals(inputs)
    orders = order isa Integer ? ntuple(_ -> order, length(marginals)) : Tuple(order)
    length(orders) == length(marginals) && all(q -> q isa Integer && q > 0, orders) ||
        throw(ArgumentError("quadrature order must be a positive count per reference coordinate"))
    count = prod(big(q) for q in orders)
    max_points isa Integer && max_points > 0 && count <= max_points ||
        throw(ArgumentError("tensor quadrature needs $count points; increase max_points explicitly or reduce orders/dimension"))
    rules = map(marginals, orders) do d,q
        basis = pce_basis(d, q-1)
        _pce_gauss_rule(basis)
    end
    X, weights = zeros(length(marginals), Int(count)), ones(Int(count))
    for (column, index) in enumerate(Iterators.product((1:q for q in orders)...))
        for j in eachindex(marginals)
            X[j,column] = rules[j].nodes[index[j]]
            weights[column] *= rules[j].weights[index[j]]
        end
    end
    all(isfinite, X) && all(w -> isfinite(w) && w > 0, weights) ||
        throw(ArgumentError("quadrature overflow/underflow; reduce orders/dimension or use another recorded rule"))
    (;samples=X, weights, names=inputs.names, coordinates=inputs.coordinates,
      method=:gauss, order=orders, basis_families=Tuple(rule.family for rule in rules),
      design_measure=:probability_quadrature, normalization=:probability_measure,
      changes_input_law=false)
end
