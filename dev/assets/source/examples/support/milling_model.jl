module MillingModel
using PopulationBalanceModels, LinearAlgebra
import Distributions

const PARAMETER_NAMES=(:delta,:alpha,:gamma,:sigma)
const MEAN=(delta=.44,alpha=8.82e-6,gamma=.34,sigma=2.10)
const SD=(delta=.07,alpha=1.01e-6,gamma=.08,sigma=.12)
const GL8=PopulationBalanceModels._gauss_legendre(GaussLegendre(8))

parameters()=(mass_feed=.0074,rho=740.,feed_median=4.81e-5,feed_log_sd=.25,
    first_pivot=6.54e-17,last_pivot=1e-3,xref=2.23e-9,daughter_ratio=2.68e5,
    screen=1575e-6,change_time=30.,speed_before=4923.,speed_after=1500.,number_scale=1e12)
admissible(p)=all(isfinite,values(p)) && 0<p.delta<1 && p.alpha>0 && p.sigma>0
speed(p,t)=p.segment_speed==0 ? (t<p.change_time ? p.speed_before : p.speed_after) : p.segment_speed
diameter(v)=cbrt(6v/π)

"""Truncated raw moment of the relative lognormal daughter law on (0, upper).
The first moment at upper=1 normalizes expected daughter COUNT, not probability.
"""
function relative_moment(k,upper,sigma,ratio)
    upper<=0 && return 0.
    mu=-log(ratio)
    exp(k*mu+k^2*sigma^2/2)*Distributions.cdf(Distributions.Normal(),(log(upper)-mu-k*sigma^2)/sigma)
end

function hat_integral(lo,hi,sigma,ratio,rising,rule)
    hi<=lo && return 0.
    midpoint=(lo+hi)/2;half=(hi-lo)/2
    norm=relative_moment(1,1.,sigma,ratio)
    sum(eachindex(rule.nodes)) do i
        z=midpoint+half*rule.nodes[i]
        fraction=expm1(z-lo)/expm1(hi-lo)
        phi=rising ? fraction : 1-fraction
        phi>=0 || error("negative fixed-pivot hat weight")
        half*rule.weights[i]*phi*exp(-.5*((z+log(ratio))/sigma)^2)/(sigma*sqrt(2π)*norm)
    end
end

struct MillingDaughters
    pivots::Vector{Float64}
    eta::Matrix{Float64}
    underflow_number::Vector{Float64}
    underflow_volume::Vector{Float64}
    sigma::Float64
end
function (d::MillingDaughters)(parent,p,t,ctx)
    j=searchsortedfirst(d.pivots,parent)
    j<=length(d.pivots) && d.pivots[j]==parent || throw(ArgumentError("milling daughters are prepared at the fixed pivots"))
    pairs=[(d.pivots[i],d.eta[i,j]) for i in 1:j if d.eta[i,j]>0]
    if d.underflow_number[j]>0
        push!(pairs,(d.underflow_volume[j]/d.underflow_number[j],d.underflow_number[j]))
    end
    pairs
end

"""Positive standard hat projection; analytic sub-grid moments leave as fines.
Geometric pivots make the interior coefficients depend on the pivot offset.
This is preparation, not an alternative PBM time integrator.
"""
function daughter_projection(grid,sigma,ratio;order=8)
    x=grid.centers;n=length(x);h=log(x[end]/x[1])/(n-1)
    rule=order==8 ? GL8 : PopulationBalanceModels._gauss_legendre(GaussLegendre(order))
    left=[hat_integral((k-1)*h,k*h,sigma,ratio,true,rule) for k in 0:-1:-(n-1)]
    right=[k==0 ? 0. : hat_integral(k*h,(k+1)*h,sigma,ratio,false,rule) for k in 0:-1:-(n-1)]
    eta=zeros(n,n)
    for j in 1:n,i in 1:j
        index=j-i+1
        eta[i,j]=right[index]+(i==1 ? 0. : left[index])
    end
    norm=relative_moment(1,1.,sigma,ratio)
    low0=[relative_moment(0,x[1]/parent,sigma,ratio)/norm for parent in x]
    low1=[parent*relative_moment(1,x[1]/parent,sigma,ratio)/norm for parent in x]
    all(>=(0),eta) || error("negative daughter projection coefficient")
    defect=vec(x'*eta)+low1-x
    maximum(abs.(defect)./x)<2e-12 || throw(NumericalFailure("daughter quadrature does not resolve volume conservation; increase its order"))
    MillingDaughters(copy(x),eta,low0,low1,sigma)
end

struct MillingHazard
    alpha::Float64
    gamma::Float64
    xref::Float64
end
(rate::MillingHazard)(x,p,t,ctx)=rate.alpha*speed(p,t)*(x/rate.xref)^rate.gamma
struct MillingFixedPivot <: AbstractTermMethod end
function PopulationBalanceModels.prepare_term(term::Breakage,::MillingFixedPivot,grid::Grid1D;kwargs...)
    term.daughters isa MillingDaughters || throw(ArgumentError("MillingFixedPivot requires prepared milling daughters"))
    term.overflow===:discard || throw(ArgumentError("sub-grid milling daughters must be explicitly accounted as discharged fines"))
    selection=term.rate.alpha.*(grid.centers./term.rate.xref).^term.rate.gamma
    Q=(term.daughters.eta-Matrix{Float64}(I,length(grid),length(grid)))*Diagonal(selection)
    (;Q,work=zeros(length(grid)),low0=term.daughters.underflow_number.*selection,
      low1=term.daughters.underflow_volume.*selection,
      frozen=(term.rate.alpha,term.rate.gamma,term.daughters.sigma))
end
function PopulationBalanceModels.add_term!(du,N,term::Breakage,::MillingFixedPivot,cache,op,p,t,ctx)
    cache.frozen==(p.alpha,p.gamma,p.sigma) || throw(ArgumentError("rebuild the milling operator after changing kinetic parameters"))
    w=speed(p,t)
    # Match the algebraic screen's feed + w*(Q*N) arithmetic. BLAS's fused
    # alpha/beta form rounds differently and leaves a spurious derivative in
    # classes whose exact retention is zero.
    mul!(cache.work,cache.Q,N)
    du .+=w.*cache.work
    w*dot(cache.low0,N),w*dot(cache.low1,N)
end

function normal_interval(dist,a,b)
    a>=b && return 0.
    a>Distributions.median(dist) ? Distributions.ccdf(dist,a)-Distributions.ccdf(dist,b) :
        Distributions.cdf(dist,b)-Distributions.cdf(dist,a)
end
function grid(;bins=100,first=parameters().first_pivot,last=parameters().last_pivot)
    ratio=(last/first)^(1/(bins-1))
    Grid1D((2first/(1+ratio)).*ratio.^(0:bins);coordinate=:volume)
end
function retention(grid,p)
    d=diameter.(grid.centers);critical=(1-p.delta)*p.screen
    [x<=critical ? 0. : x>=p.screen ? 1. : (x-critical)/(p.delta*p.screen) for x in d]
end

function net_production(p,N,t)
    p.feed + speed(p,t)*(p.Q*N)
end
function outlet(p,N,t)
    produced=net_production(p,N,t)
    resolved=(1 .-p.retained).*produced
    low0=speed(p,t)*dot(p.low0,N)
    low1=speed(p,t)*dot(p.low1,N)
    (;resolved,underflow_number=low0,underflow_volume=low1)
end

"""Instantaneous outlet volume-weighted diameter median. The resolved CDF is
linearly interpolated in diameter between cumulative fixed-pivot volume values.
Sub-grid volume is included; an empty outlet is undefined, never seeded.
"""
function outlet_median(p,N,t)
    q=outlet(p,N,t);weights=p.pivots.*q.resolved
    all(>=(0),weights) && q.underflow_volume>=0 || throw(NumericalFailure("negative instantaneous outlet flux"))
    total=sum(weights)+q.underflow_volume
    total>0 || throw(NumericalFailure("outlet median is undefined before particles leave the mill"))
    target=total/2
    if target<q.underflow_volume
        # Resolve the continuous fines rather than inventing a first-bin median.
        lo=0.;hi=first(p.pivots);norm=relative_moment(1,1.,p.sigma,p.daughter_ratio)
        for _ in 1:65
            mid=(lo+hi)/2
            volume=speed(p,t)*sum(p.selection[j]*N[j]*p.pivots[j]*
                relative_moment(1,mid/p.pivots[j],p.sigma,p.daughter_ratio)/norm for j in eachindex(N))
            volume<target ? (lo=mid) : (hi=mid)
        end
        return 1e6diameter((lo+hi)/2)
    end
    cumulative=q.underflow_volume
    previous_d=1e6diameter(first(p.pivots))
    for i in eachindex(weights)
        next=cumulative+weights[i];d=1e6diameter(p.pivots[i])
        if next>=target
            return weights[i]>0 ? previous_d+(d-previous_d)*(target-cumulative)/weights[i] : d
        end
        cumulative=next;previous_d=d
    end
    throw(NumericalFailure("outlet median did not lie in the declared size domain"))
end

"""Published growth-free cone-mill equations, on an explicit reconstruction grid.
N is total particle inventory divided by number_scale (unit reference process
volume); rho*x*N*number_scale is kg, not a suspension concentration.
"""
function build(theta=MEAN;bins=100,tf=200.,last_pivot=parameters().last_pivot,quadrature_order=8)
    admissible(theta) || throw(NumericalFailure("milling inputs require 0<delta<1, alpha>0 and sigma>0"))
    constants=parameters();g=grid(;bins,last=last_pivot)
    (1-theta.delta)*constants.screen>=diameter(first(g.centers)) ||
        throw(NumericalFailure("the first milling pivot must lie below the freely discharged screen range; reduce the lower cutoff"))
    daughters=daughter_projection(g,theta.sigma,constants.daughter_ratio;order=quadrature_order)
    rate=MillingHazard(theta.alpha,theta.gamma,constants.xref)
    selection=theta.alpha.*(g.centers./constants.xref).^theta.gamma
    Q=(daughters.eta-Matrix{Float64}(I,bins,bins))*Diagonal(selection)
    distribution=Distributions.LogNormal(log(constants.feed_median),constants.feed_log_sd)
    fractions=[normal_interval(distribution,g.edges[i],g.edges[i+1]) for i in 1:bins]
    resolved_feed_fraction=sum(fractions)
    feed=constants.mass_feed/constants.rho .* (fractions/resolved_feed_fraction)./g.centers/constants.number_scale
    p0=merge(constants,theta,(;pivots=g.centers,edges=g.edges,selection,Q,
        low0=daughters.underflow_number.*selection,low1=daughters.underflow_volume.*selection,
        feed,segment_speed=0.,resolved_feed_fraction))
    p=merge(p0,(retained=retention(g,p0),))
    feedterm=IntegratedBirth((a,b,p,t,c)->p.feed[searchsortedfirst(p.edges,a)])
    # Classification is an algebraic removal of feed + net breakage production,
    # not an arbitrary first-order Death term. Recompute it explicitly so term
    # ordering does not determine the physical equation.
    screen=CustomTerm((du,N,g,p,t,c)->(du .-=outlet(p,N,t).resolved;nothing))
    model=PBMModel(feed=feedterm,breakage=Breakage(rate,daughters;overflow=:discard),classification=screen)
    op=compile(model,g;methods=TermMethods(by_term=(breakage=MillingFixedPivot(),)))
    balance=function(ctx,p,t)
        q=outlet(p,ctx.populations.mill,t)
        (discharged_mass=p.rho*p.number_scale*(dot(p.pivots,q.resolved)+q.underflow_volume),
         fines_mass=p.rho*p.number_scale*q.underflow_volume)
    end
    system=PBMSystem((mill=op,);bulk_rhs=balance,nonnegative_bulk=(:discharged_mass,:fines_mass))
    PBMProblem(system,(mill=zeros(bins),),(0.,tf);bulk0=(discharged_mass=0.,fines_mass=0.),p)
end

median_observable()=SystemObservable((state,op,p,t)->outlet_median(p,state.populations.mill,t))

"""Analytic Jacobian of the same affine residual, for external linear algebra
and independent matrix-exponential verification. Feed is the affine offset.
"""
function linear_system(pb)
    p=pb.p;n=length(p.pivots);w=speed(p,first(pb.tspan))
    A=zeros(n+2,n+2);A[1:n,1:n]=Diagonal(p.retained)*(w*p.Q)
    A[n+1,1:n]=p.rho*p.number_scale*w*(vec(p.pivots'*Diagonal(1 .-p.retained)*p.Q)+p.low1)
    A[n+2,1:n]=p.rho*p.number_scale*w*p.low1
    b=vcat(p.retained.*p.feed,p.rho*p.number_scale*dot(p.pivots,(1 .-p.retained).*p.feed),0.)
    A,b
end

end # module
