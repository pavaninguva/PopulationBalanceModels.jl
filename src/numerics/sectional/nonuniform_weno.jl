# Finite-volume WENO5 on fixed nonuniform meshes. All geometry is prepared at
# compilation, in a coordinate with the donor cell [-1,0] and the face at zero.
# See Wang, Feng & Spiteri (2008), Applied Mathematics and Computation 196,
# 433–447: quadratic cell-average reconstruction and Jiang–Shu indicators.
struct _WENO5Stencil{T}
    indices::NTuple{5,Int}
    polynomials::NTuple{3,NTuple{3,NTuple{3,T}}}
    weights::NTuple{3,T}
end

struct _NonuniformWENO5{E,V} <: AbstractReconstruction
    epsilon::E
    positive::V
    negative::V
end

_weno_uniform(g) = maximum(abs.(g.widths .- first(g.widths))) <= 1e-10first(g.widths)
_prepared_reconstruction(method, cache) = method
_prepared_reconstruction(::WENO5, cache::_NonuniformWENO5) = cache

function _prepare_reconstruction(method, grid, bc)
    _validate_reconstruction(method, grid)
    method
end

function _weno_average_inverse(edges, firstcell, count)
    T = eltype(edges)
    averages = Matrix{T}(undef, count, count)
    for row in 1:count, power in 0:count-1
        a, b = edges[firstcell+row-1], edges[firstcell+row]
        # A sum avoids cancellation in (b^(power+1)-a^(power+1))/(b-a).
        averages[row,power+1] = sum(b^(power-k)*a^k for k in 0:power)/(power+1)
    end
    all(isfinite, averages) || throw(ArgumentError(
        "WENO5 reconstruction has unresolved cell-width ratios; use a less stretched mesh"))
    inverse = try
        averages \ Matrix{T}(I, count, count)
    catch error
        error isa SingularException || rethrow()
        throw(ArgumentError("WENO5 reconstruction is singular on this mesh; reduce neighbouring cell-width ratios"))
    end
    condition = opnorm(averages, Inf)*opnorm(inverse, Inf)
    isfinite(condition) && condition*eps(T) < 1e-6 || throw(ArgumentError(
        "WENO5 reconstruction is ill-conditioned on this mesh (scaled condition $condition); reduce neighbouring cell-width ratios"))
    inverse
end

function _weno_stencil(grid, face, direction, periodic, ::Type{T}) where T
    m = length(grid)
    # Keep the existing physical-boundary upwind fallback. No invented ghost
    # averages or smooth continuation of an inflow condition are introduced.
    !periodic && !(4 <= face <= m-2) && return nothing
    donor = direction == 1 ? face-1 : face
    indices = ntuple(k -> periodic ? mod1(donor+direction*(k-3),m) : donor+direction*(k-3), 5)
    widths = ntuple(k -> T(grid.widths[indices[k]]/grid.widths[indices[3]]), 5)
    edges = T[-1-widths[2]-widths[1], -1-widths[2], -1, 0, widths[4], widths[4]+widths[5]]
    matrices = ntuple(k -> _weno_average_inverse(edges,k,3), 3)
    polynomials = ntuple(k -> ntuple(row -> ntuple(col -> matrices[k][row,col],3),3),3)
    optimal = _weno_average_inverse(edges,1,5)[1,:]
    d0 = optimal[1]/matrices[1][1,1]
    d2 = optimal[5]/matrices[3][1,3]
    weights = (d0, one(T)-d0-d2, d2)
    combined = zeros(T,5)
    for k in 1:3, col in 1:3
        combined[k+col-1] += weights[k]*matrices[k][1,col]
    end
    all(w -> isfinite(w) && w>0, weights) &&
        norm(combined-optimal,Inf) <= 1e-8max(one(T),norm(optimal,Inf)) || throw(ArgumentError(
        "WENO5 has unresolved or nonpositive optimal weights at face $face; reduce neighbouring cell-width ratios"))
    _WENO5Stencil(indices,polynomials,weights)
end

function _prepare_reconstruction(method::WENO5, grid, bc)
    _validate_reconstruction(method, grid)
    _weno_uniform(grid) && return method # Preserve the uniform-grid fast path.
    T = promote_type(eltype(grid.widths), typeof(float(method.epsilon)))
    periodic = bc.left isa Periodic
    build(direction) = Union{Nothing,_WENO5Stencil{T}}[
        _weno_stencil(grid,face,direction,periodic,T) for face in 1:length(grid)+1]
    _NonuniformWENO5(method.epsilon,build(1),build(-1))
end

function _face_density(n, face, method::_NonuniformWENO5, velocity, periodic, grid)
    stencil = (velocity>=0 ? method.positive : method.negative)[face]
    stencil === nothing && return _face_density(n,face,Upwind(),velocity,periodic)
    _weno_value(n,stencil,method.epsilon)
end

function _weno_value(n, stencil::_WENO5Stencil, epsilon)
    reference = n[stencil.indices[3]]
    values = ntuple(k -> n[stencil.indices[k]]-reference,5)
    candidates = ntuple(3) do k
        polynomial = stencil.polynomials[k]
        a = ntuple(row -> sum(polynomial[row][col]*values[k+col-1] for col in 1:3),3)
        # Integral of p'² + p''² over [-1,0]. This sum-of-squares form
        # avoids an indefinite quadratic form caused by roundoff.
        beta = (a[2]-a[3])^2 + (13//3)*a[3]^2
        (value=reference+a[1], beta=beta)
    end
    # Equivalent to d/(epsilon+beta)^2, but avoids tiny-denominator overflow.
    scale = minimum(c -> epsilon+c.beta,candidates)
    alpha = ntuple(k -> stencil.weights[k]*(scale/(epsilon+candidates[k].beta))^2,3)
    sum(alpha[k]*candidates[k].value for k in 1:3)/sum(alpha)
end
