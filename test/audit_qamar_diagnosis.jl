include("fixtures/example_modules.jl")
# Targeted A13 reproducer. Core mode uses only package dependencies.
# --adaptive needs SciMLBase, OrdinaryDiffEqTsit5 and direct SpecialFunctions
# access in an isolated environment.
# Private native hooks below are test instrumentation, not extension APIs.
using PopulationBalanceModels, LinearAlgebra
include("fixtures/qamar_reference.jl")
_load_example(:QamarMomentBenchmarks, "qamar_moment_benchmarks.jl")
using .QamarReference
const P=PopulationBalanceModels
const Q=QamarMomentBenchmarks
const columns=(:scenario,:driver,:dt,:primary_nodes,:states,:quadrature,:precision,:reltol,:event_time,
    :last_saved,:stage,:status,:quantity,:value,:condition,:reconstruction_error,:hankel_margin,:detail)
function record(io;scenario,driver,dt=NaN,primary_nodes=3,states=7,quadrature=0,precision=53,reltol=NaN,
        event_time,last_saved=event_time,stage="reference",status="checked",quantity="",value=NaN,
        condition=NaN,reconstruction_error=NaN,hankel_margin=NaN,detail="")
    row=(;scenario,driver,dt,primary_nodes,states,quadrature,precision,reltol,event_time,last_saved,stage,status,
        quantity,value,condition,reconstruction_error,hankel_margin,detail)
    cell(x)=x isa AbstractString ? "\""*replace(x,"\""=>"\"\"")*"\"" : string(x)
    println(io,join((cell(getproperty(row,k)) for k in columns),','));flush(io)
end
mutable struct TracedOperator{T}
    pb::T
    calls::Int
    checks::Int
    base::Float64
    h::Float64
    phase::String
    stage_time::Float64
    state::Matrix{Float64}
end
function P.rhs!(du,u,op::TracedOperator,p,t)
    op.calls+=1;op.phase="rhs_$(op.calls)";op.stage_time=t;op.state=copy(u)
    rhs!(du,u,op.pb.operator,p,t)
end
function P._check_problem_state(pb::P.PBMProblem{<:TracedOperator},u,tol)
    op=pb.operator;op.checks+=1;op.phase="state_$(op.checks)"
    op.stage_time=op.base+(op.checks==2 ? op.h : op.checks==3 ? op.h/2 : 0.);op.state=copy(u)
    P._check_problem_state(op.pb,u,tol)
end
function trace(method,collisions,dt;tf=.5)
    pb=Q.problem5(;method,collisions,tf)
    op=TracedOperator(pb,0,0,0.,dt,"",0.,copy(pb.u0))
    traced=P.PBMProblem(op,pb.u0,pb.tspan,pb.p)
    u=copy(pb.u0);cache=ntuple(_->similar(u),6);t=0.;lastsaved=0.
    for target in .005:.005:tf
        while t<target
            h=min(dt,target-t);op.calls=0;op.checks=0;op.base=t;op.h=h
            try
                P._rk_step!(u,cache,traced,t,h,SSPRK33(),true,1e-12)
                op.phase="accepted";op.stage_time=t+h;op.state=copy(u)
                P._check_problem_state(pb,u,1e-12);P._check_admissible_state(pb.operator,u)
            catch e
                e isa Union{NumericalFailure,ArgumentError} || rethrow()
                return (;time=op.stage_time,last_saved=lastsaved,stage=op.phase,state=op.state,status="failed",reason=sprint(showerror,e))
            end
            t=h==target-t ? target : t+h
        end
        lastsaved=target
    end
    (;time=tf,last_saved=tf,stage="accepted",state=u,status="completed",reason="")
end

function native_checks(io)
    configurations=vcat([(EQMOM(3),false,h) for h in (2e-4,1e-4)],
        [(EQMOM(3;secondary_nodes=n),true,h) for n in (12,48) for h in (2e-4,1e-4,5e-5)],
        [(LogNormalClosure(quadrature_order=n),true,h) for n in (64,128) for h in (2e-4,1e-4,5e-5)])
    for (method,collisions,dt) in configurations
        result=trace(method,collisions,dt)
        basis=Q.method_basis(method);check=realizability(basis,result.state)
        mu=vec(result.state);scale=mu[2]/mu[1]
        normalized=[mu[k+1]/mu[1]/scale^k for k in basis.orders]
        n=min(3,(length(mu)+1)÷2)
        condition=cond([normalized[i+j-1] for i in 1:n,j in 1:n])
        error=method isa EQMOM && check.valid ? Float64(gamma_certificate(mu).error) : NaN
        record(io;scenario=collisions ? "collisions" : "growth_nucleation",driver=string(nameof(typeof(method)))*"_SSPRK33",
            dt,primary_nodes=method isa EQMOM ? 3 : 0,states=length(result.state),quadrature=method isa EQMOM ? method.secondary_nodes : method.quadrature_order,
            event_time=result.time,last_saved=result.last_saved,stage=result.stage,status=result.status,
            quantity="tracked_state",condition,reconstruction_error=error,hankel_margin=check.min_hankel_eigenvalue,
            detail=result.reason*"; state="*string(vec(result.state)))
    end
end
function controlled(io,secondary,tol;inversion_rtol=1e-7)
    pb=Q.problem5(method=EQMOM(3;secondary_nodes=secondary,rtol=inversion_rtol));u=copy(pb.u0);t=0.;h=.0002
    accepted=0;rejected=0;lastreason="";stage_time=0.;stage_state=copy(u)
    function f(v,s)
        stage_time=s;stage_state=copy(v)
        P._check_problem_state(pb,v,1e-12);P._check_admissible_state(pb.operator,v)
        out=zero(v);rhs!(out,v,pb.operator,pb.p,s);out
    end
    function step(u,t,h)
        a=f(u,t);b=f(u+h/2*a,t+h/2);c=f(u+h/2*b,t+h/2);d=f(u+h*c,t+h)
        out=u+h/6*(a+2b+2c+d)
        P._check_problem_state(pb,out,1e-12);P._check_admissible_state(pb.operator,out)
        out
    end
    while t<.5
        h=min(h,.5-t,.001)
        if h<1e-12
            record(io;scenario="eqmom_collisions",driver="independent_RK4_step_doubling",
                dt=.001,quadrature=secondary,reltol=tol,event_time=stage_time,last_saved=t,stage="trial",status="step_floor",
                quantity="last_admissible_state",hankel_margin=realizability(pb.operator.grid,u).min_hankel_eigenvalue,
                detail="inversion_rtol=$inversion_rtol; accepted=$accepted; rejected=$rejected; $lastreason; state=$(vec(u)); trial=$(vec(stage_state))")
            return
        end
        error=Inf
        try
            coarse=step(u,t,h);fine=step(step(u,t,h/2),t+h/2,h/2)
            error=maximum(abs.(coarse-fine)./(15tol.*(.01 .+ max.(abs.(u),abs.(fine)))))
            if error<=1
                u=fine;t+=h;accepted+=1
            end
        catch e
            (e isa NumericalFailure || (e isa ArgumentError && e.msg=="invalid aggregate coordinate")) || rethrow()
            lastreason=sprint(showerror,e)
        end
        error>1&&(rejected+=1)
        h*=isfinite(error) ? clamp(.8max(error,1e-10)^(-1/5),.1,2.) : .5
    end
    record(io;scenario="eqmom_collisions",driver="independent_RK4_step_doubling",quadrature=secondary,
        reltol=tol,event_time=t,status="completed",detail="inversion_rtol=$inversion_rtol; state=$(vec(u))")
end

function reference_checks(io)
    for bits in (256,512),time in (".45",".475",".4788",".479",".48",".49",".5")
        mu=setprecision(()->characteristic_moments(time),BigFloat,bits)
        q=gamma_certificate(mu;bits)
        roots=gamma_width_roots(mu;bits)
        record(io;scenario="growth_nucleation",driver="independent_characteristics",precision=bits,
            event_time=parse(Float64,time),status=q.error<=1e-7 ? "represented" : "not_reconstructed",quantity="relative_M0_M6_reconstruction",value=Float64(q.error),
            condition=q.condition,reconstruction_error=Float64(q.error),hankel_margin=Float64(q.shifted_margin),
            detail="positive-width boundary; primary weights="*string(Float64.(q.weights)))
        for root in roots
            record(io;scenario="growth_nucleation",driver="independent_width_polynomial",precision=bits,
                event_time=parse(Float64,time),status=root.shifted_margin>0 && root.primary_margin>0 ? "positive_candidate" : "infeasible_candidate",quantity="shifted_LDL_margin_at_detH0_root",value=Float64(root.shifted_margin),
                detail="full allowed width interval; roots=$(length(roots)); degree=$(root.degree); normalized_width=$(Float64(root.width))")
        end
    end
    # Independently specified Gamma mixture: the degree-six hazard needs M12,
    # beyond the seven evolved moments. Check that RHS separately from evolution.
    kernelmoment(x,sigma,k)=prod((x+j*sigma for j in 0:k-1);init=1.)
    x=[.5,1.,2.];w=[.3,.4,.3];sigma=.1
    moment(k)=sum(wi*kernelmoment(xi,sigma,k) for (xi,wi) in zip(x,w))
    raw=[moment(k) for k in 0:6]
    for order in (12,48)
        op=compile(PBMModel(Breakage((x,p,t,c)->x^6,UniformVolumeDaughters();coordinate_map=Q.volume_map)),
            MomentSet(0:6;coordinate=:length);method=EQMOM(3;secondary_nodes=order))
        derivative=zeros(7,1);rhs!(derivative,reshape(raw,:,1),op,NamedTuple(),0.)
        expected=[(6/(k+3)-1)*moment(k+6) for k in 0:6]
        error=maximum(abs(derivative[k+1]/expected[k+1]-1) for k in (0,1,2,4,5,6))
        @assert error<1e-7
        @assert abs(derivative[4])<1e-12*moment(9)
        record(io;scenario="gamma_breakage",driver="analytic_M6_M12",quadrature=order,event_time=0.,
            quantity="maximum_relative_weak_rate_error",value=error,reconstruction_error=error)
    end
    for variance in (.1,.3,.6,1.),n in (32,64,128)
        mu=[20.,10.,5exp(variance)];basis=MomentSet(0:2;coordinate=:length)
        op=compile(PBMModel(Breakage((x,p,t,c)->x^6,UniformVolumeDaughters();coordinate_map=Q.volume_map)),basis;method=LogNormalClosure(quadrature_order=n))
        derivative=zeros(3,1);rhs!(derivative,reshape(mu,:,1),op,NamedTuple(),0.)
        expected=lognormal_breakage_rates(20.,.5,variance)
        record(io;scenario="lognormal_breakage",driver="analytic_M6_M8",primary_nodes=0,states=3,quadrature=n,event_time=0.,
            quantity="log_variance",value=variance,reconstruction_error=maximum(abs.(vec(derivative)./expected.-1)))
    end
end
function log_rhs!(du,y,p,t)
    a,b,v=y;N=exp(a);r=exp(v);J=Q.boundary_density(t)
    # Independent normalized quadrature only for the nonpolynomial aggregation.
    v>=0 || throw(DomainError(v,"negative log-variance in independent closure calculation"))
    z,w=p.rule;u=exp.(sqrt(v).*z.-v/2)
    agg1=0.;agg2=0.
    for i in eachindex(u),j in eachindex(u)
        product=cbrt(u[i]^3+u[j]^3)
        agg1+=w[i]*w[j]*(product-u[i]-u[j])
        agg2+=w[i]*w[j]*(product^2-u[i]^2-u[j]^2)
    end
    if p.analytic
        B=[exp(6b+15v),.5exp(6b+21v),.2exp(6b+27v)]
    else
        B=[sum(w.*u.^(k+6))*(6/(k+3)-1)*exp(6b)/r^(k*(k-1)/2) for k in 0:2]
    end
    d0=J/N-.5*1.5e-5*N+B[1]
    d1=exp(-b)+.5*1.5e-5*N*agg1+B[2]
    d2=2exp(-b-v)+.5*1.5e-5*N/r*agg2+B[3]
    du.=(d0,d1-d0,d0+d2-2d1)
end

if "--adaptive" in ARGS
    @eval using SciMLBase, OrdinaryDiffEqTsit5
end
function adaptive_checks(io)
    mu=[Q.initial_moment5(k) for k in 0:2]
    y0=[log(mu[1]),log(mu[2]/mu[1]),log(mu[1]*mu[3]/mu[2]^2)]
    for analytic in (false,true),n in (64,128),tol in (1e-7,1e-9)
        p=(rule=P._normal_gauss_rule(n),analytic=analytic)
        for limit in (.5,1.)
            cb=ContinuousCallback((u,t,int)->u[3]-limit,int->terminate!(int);save_positions=(false,true))
            sol=solve(ODEProblem(log_rhs!,y0,(0.,.12),p),Tsit5();reltol=tol,abstol=tol*.01,callback=cb,save_everystep=false)
            record(io;scenario="lognormal_collisions",driver=analytic ? "Tsit5_analytic_breakage" : "Tsit5_quadrature_breakage",
                primary_nodes=0,states=3,quadrature=n,reltol=tol,event_time=last(sol.t),status=string(sol.retcode),quantity="log_variance",value=last(sol.u)[3],
                detail="logN=$(last(sol.u)[1]); logmean=$(last(sol.u)[2]); accepted=$(sol.stats.naccept); rejected=$(sol.stats.nreject)")
        end
    end
    # Reparametrize the SAME analytic closure ODE by its increasing width to
    # distinguish rapid approach to a finite-time runaway from fixed-step overflow.
    for tol in (1e-9,1e-11,1e-12)
        p=(rule=P._normal_gauss_rule(128),analytic=true)
        cb=ContinuousCallback((u,t,int)->u[3]-.5,int->terminate!(int);save_positions=(false,true))
        sol=solve(ODEProblem(log_rhs!,y0,(0.,.12),p),Tsit5();reltol=tol,abstol=tol*.01,callback=cb,save_everystep=false)
        function width_rhs!(du,y,p,v)
            dy=zeros(3);log_rhs!(dy,[y[2],y[3],v],p,y[1])
            du.=(1/dy[3],dy[1]/dy[3],dy[2]/dy[3])
        end
        start=[last(sol.t),last(sol.u)[1],last(sol.u)[2]]
        solution=solve(ODEProblem(width_rhs!,start,(.5,8.),p),Tsit5();reltol=tol,abstol=tol*.01,saveat=[.5,1.,2.,4.,8.])
        for (v,y) in zip(solution.t,solution.u)
            record(io;scenario="lognormal_collisions",driver="width_parameterized_analytic_ODE",primary_nodes=0,states=3,quadrature=128,
                reltol=tol,event_time=y[1],quantity="log_variance",value=v,
                detail="logN=$(y[2]); logmean=$(y[3]); analytic polynomial breakage, quadrature aggregation")
        end
    end
end
paths=filter(x->!startswith(x,"--"),ARGS)
output=isempty(paths) ? joinpath(@__DIR__,"..","docs","src","assets","audit","qamar_diagnosis.csv") : first(paths)
mkpath(dirname(abspath(output)))
open(output,"w") do io
    println(io,join(columns,','))
    reference_checks(io)
    if "--adaptive" in ARGS
        adaptive_checks(io)
    else
        native_checks(io)
        for order in (12,48),tol in (1e-7,1e-9)
            controlled(io,order,tol)
        end
        controlled(io,12,1e-9;inversion_rtol=1e-11)
    end
end
println("Wrote ",output)
