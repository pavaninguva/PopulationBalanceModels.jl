"""
    conditional_cdf(grid, N, points; window=(first(grid.edges), last(grid.edges)))

Conditional **number** CDF of a one-coordinate sectional population. `N` contains
bin integrals. Integrates the piecewise-constant density, including partial
window cells. Empty windows and negative populations are invalid observations.
No moment reconstruction or smoothing is implicit.
"""
function conditional_cdf(grid::Grid1D,N,points;window=(first(grid.edges),last(grid.edges)))
    a,b,total=_conditional_window(grid,N,window)
    all(isfinite,points) || throw(ArgumentError("CDF evaluation coordinates must be finite"))
    [sum(N[i]*max(zero(a),min(x,b,grid.edges[i+1])-max(a,grid.edges[i]))/grid.widths[i]
         for i in eachindex(grid.widths))/total for x in points]
end

function _conditional_window(grid,N,window)
    length(N)==length(grid) || throw(DimensionMismatch("one population integral per size bin required"))
    a,b=window
    isfinite(a) && isfinite(b) && first(grid.edges)<=a<b<=last(grid.edges) ||
        throw(ArgumentError("conditional observation window must lie inside the size domain"))
    all(isfinite,N) && all(x->_primal(x)>=0,N) || throw(NumericalFailure("conditional number observations require a finite nonnegative population"))
    total=sum(N[i]*max(zero(a),min(b,grid.edges[i+1])-max(a,grid.edges[i]))/grid.widths[i] for i in eachindex(grid.widths))
    _primal(total)>0 || throw(NumericalFailure("conditional number distribution is undefined for an empty observation window"))
    a,b,total
end

"""
    conditional_quantiles(grid, N, probabilities; window=...)

Generalized inverse of `conditional_cdf`, with exact partial-cell integration.
Probabilities must increase. Flat CDF intervals carry no probability mass.
"""
function conditional_quantiles(grid::Grid1D,N,probabilities;window=(first(grid.edges),last(grid.edges)))
    a,b,total=_conditional_window(grid,N,window)
    us=collect(probabilities)
    !isempty(us) && all(isfinite,us) && all(u->0<=u<=1,us) && issorted(us) || throw(ArgumentError("quantile probabilities must increase within [0,1]"))
    result=Vector{typeof(total+a)}(undef,length(us))
    i=1; accumulated=zero(total)
    for (j,u) in enumerate(us)
        target=u*total
        while i<=length(grid)
            lo=max(a,grid.edges[i]); hi=min(b,grid.edges[i+1])
            width=max(zero(a),hi-lo); mass=N[i]*width/grid.widths[i]
            if _primal(mass)>0 && _primal(target)<=_primal(accumulated+mass)
                result[j]=lo+(target-accumulated)*grid.widths[i]/N[i]
                break
            end
            accumulated+=mass; i+=1
        end
        i<=length(grid) || (result[j]=b) # endpoint u=1, after rounding the sum
    end
    result
end

"""Generalized inverse of a tabulated, piecewise-linear CDF. No density smoothing."""
function cdf_quantiles(x,F,probabilities)
    length(x)==length(F)>=2 && all(isfinite,x) && all(diff(x).>0) &&
        all(isfinite,F) && all(diff(F).>=0) && first(F)==0 && last(F)==1 ||
        throw(ArgumentError("CDF table needs increasing coordinates, nondecreasing CDF and endpoints exactly 0 and 1"))
    us=collect(probabilities)
    all(u->isfinite(u) && 0<=u<=1,us) || throw(ArgumentError("probabilities must lie in [0,1]"))
    map(us) do u
        k=searchsortedfirst(F,u)
        k<=1 && return first(x)
        F[k]==F[k-1] && return x[k-1]
        x[k-1]+(u-F[k-1])/(F[k]-F[k-1])*(x[k]-x[k-1])
    end
end

"""Coupled-system conditional number CDF at `points`; requires an explicit size window."""
struct PopulationCDF{X,W} <: AbstractObservable
    population::Symbol
    points::X
    window::W
end
PopulationCDF(population,points;window)=PopulationCDF(population,collect(points),window)
"""Coupled-system conditional number quantiles; requires an explicit size window."""
struct PopulationQuantiles{X,W} <: AbstractObservable
    population::Symbol
    probabilities::X
    window::W
end
PopulationQuantiles(population,probabilities;window)=PopulationQuantiles(population,collect(probabilities),window)
_observation_size(o::PopulationCDF)=length(o.points)
_observation_size(o::PopulationQuantiles)=length(o.probabilities)
function _observe(o::Union{PopulationCDF,PopulationQuantiles},N,op::CompiledSystem,p,t)
    pop=getproperty(op.system.populations,o.population)
    pop isa CompiledPBM && pop.grid isa Grid1D || throw(ArgumentError("conditional distribution observations require a one-coordinate sectional population; moments need an explicit reconstruction"))
    n=population(op,N,o.population)
    o isa PopulationCDF ? conditional_cdf(pop.grid,n,o.points;window=o.window) :
        conditional_quantiles(pop.grid,n,o.probabilities;window=o.window)
end

"""
    ResidualObservation(times, data, observable; scale=1, weight=1, name=:residual)

A deterministic discrepancy block, contributing `weight * mean(((prediction-
data)/scale).^2)` to `objective`. The scale is a comparison unit, **not** a
sensor standard deviation. Cannot be mixed with Gaussian `Observation` blocks.
Sensitivity singular values remain meaningful; confidence intervals do not
follow from these weights.
"""
struct ResidualObservation{T,Y,O,S,W} <: AbstractObservation
    times::T
    data::Y
    observable::O
    scale::S
    weight::W
    name::Symbol
end
function ResidualObservation(times,data,observable::AbstractObservable;scale=1.,weight=1.,name=:residual)
    isfinite(scale) && scale>0 && isfinite(weight) && weight>0 || throw(ArgumentError("discrepancy scale and weight must be finite and positive"))
    checked=Observation(times,data,observable)
    ResidualObservation(checked.times,checked.data,observable,float(scale),float(weight),name)
end
function ResidualObservation(data,observable::AbstractObservable;scale=1.,weight=1.,name=:residual)
    scale>0 && isfinite(scale) && weight>0 && isfinite(weight) || throw(ArgumentError("positive finite discrepancy scale and weight required"))
    checked=Observation(data,observable)
    ResidualObservation(nothing,checked.data,observable,float(scale),float(weight),name)
end
_whiten(o::ResidualObservation,raw)=raw.*(sqrt(o.weight/length(o.data))/o.scale)
_discrepancy(cal)=first(first(cal.experiments).observations) isa ResidualObservation

"""
    WassersteinObservation(time, x, F, population; window, points=256, scale=1, weight=1)

One conditional-number distribution discrepancy, `weight * W₂²/scale²`.
Midpoint quadrature integrates squared quantile differences on (0,1).
`x,F` describe a conditional number CDF over the declared window. Quadrature
points are integration nodes, not independent experimental observations.
"""
function WassersteinObservation(time,x,F,population;window,points=256,scale=1.,weight=1.,name=:distribution)
    points isa Integer && points>=2 || throw(ArgumentError("use at least two quantile integration points"))
    first(x)==first(window) && last(x)==last(window) || throw(ArgumentError("observed CDF endpoints must match the conditioning window"))
    u=((1:points).-0.5)./points
    ResidualObservation([time],reshape(cdf_quantiles(x,F,u),:,1),
        PopulationQuantiles(population,u;window);scale,weight,name)
end
"""`WassersteinObservation(x, F, population; window, ...)` declares the same
conditional-number discrepancy for a stationary operating point, without time."""
function WassersteinObservation(x,F,population;window,points=256,scale=1.,weight=1.,name=:distribution)
    points isa Integer && points>=2 || throw(ArgumentError("use at least two quantile integration points"))
    first(x)==first(window) && last(x)==last(window) || throw(ArgumentError("observed CDF endpoints must match the conditioning window"))
    u=((1:points).-0.5)./points
    ResidualObservation(cdf_quantiles(x,F,u),PopulationQuantiles(population,u;window);scale,weight,name)
end

"""Individual deterministic discrepancy contributions, with no likelihood interpretation."""
function loss_contributions(cal,z)
    _discrepancy(cal) || throw(ArgumentError("loss_contributions requires ResidualObservation blocks"))
    [(;experiment=i,name=o.name,value=sum(abs2,_whiten(o,vec(y.-o.data))))
     for (i,(e,ys)) in enumerate(zip(cal.experiments,predictions(cal,z))) for (o,y) in zip(e.observations,ys)]
end
