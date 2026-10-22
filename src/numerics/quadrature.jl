# Eight-point Gauss–Legendre, subdivided in standardized coordinates. Clipping
# only the *integration bounds* at 12 standard deviations discards <4e-33
# probability. State concentrations are never clipped. This avoids an erf
# dependency, midpoint aliasing and cancellation in tiny tail probabilities.
const _GL8_X=(0.1834346424956498,0.5255324099163290,0.7966664774136267,0.9602898564975363)
const _GL8_W=(0.3626837833783620,0.3137066458778873,0.2223810344533745,0.1012285362903763)
function _normal_fraction(a,b)
    lo=max(float(a),-12.0); hi=min(float(b),12.0)
    lo>=hi && return 0.0
    pieces=max(1,ceil(Int,2*(hi-lo))); h=(hi-lo)/pieces; total=0.0
    for j in 1:pieces
        mid=lo+(j-0.5)*h
        for q in eachindex(_GL8_X)
            z=h*_GL8_X[q]/2
            total+=h/2*_GL8_W[q]*(exp(-(mid-z)^2/2)+exp(-(mid+z)^2/2))/sqrt(2π)
        end
    end
    total
end
_source_value(x::Number,p,t,ctx)=x
_source_value(f,p,t,ctx)=f(p,t,ctx)


function _gauss_legendre(q::GaussLegendre)
    n=q.order
    eigenpair=eigen(SymTridiagonal(zeros(n),[j/sqrt(4j^2-1) for j in 1:n-1]))
    (;nodes=eigenpair.values,weights=2 .*eigenpair.vectors[1,:].^2)
end
function _integrate_bin(f,lo,hi,q)
    mid=(lo+hi)/2; half=(hi-lo)/2
    half*sum(q.weights[j]*f(mid+half*q.nodes[j]) for j in eachindex(q.nodes))
end

# Gauss rule for the standard normal measure. At high orders, a generic
# eigensolver can return tiny first eigenvector entries as exactly zero. The
# Christoffel identity w(x)=1/sum(p_k(x)^2), k=0:n-1, recovers their relative
# accuracy from orthonormal probabilists' Hermite polynomials. Rescaling the
# recurrence is arithmetic scaling, not adding weights or modifying moments.
function _normal_gauss_rule(n)
    nodes=eigvals(SymTridiagonal(zeros(n),sqrt.(Float64.(1:n-1))))
    weights=map(nodes) do x
        previous=0.0;current=1.0;squares=1.0;logscale=0.0
        for k in 1:n-1
            next=(x*current-sqrt(k-1)*previous)/sqrt(k)
            previous=current;current=next
            if max(abs(previous),abs(current))>1e100
                previous/=1e100;current/=1e100;squares/=1e200
                logscale+=log(1e100)
            end
            squares+=current^2
        end
        exp(-2logscale-log(squares))
    end
    all(isfinite,weights) && all(>=(0),weights) && isapprox(sum(weights),1;rtol=1e-12) ||
        throw(NumericalFailure("normal quadrature weights lost accuracy"))
    (;nodes,weights)
end
