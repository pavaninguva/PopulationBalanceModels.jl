"""
Independent nodal audit of Li et al. (2026), supplement Eq. S11.

This script deliberately uses nodal densities, not Grid1D/Grid2D bin integrals.
The 2D calculation is a product of the S11 directional updates for separable,
autonomous velocities, with a Gaussian negligible at the zero-Dirichlet faces.
It does not implement the main paper's multilevel Eqs. 6–16 or its unspecified
high-order physical boundary closure. It is not a new public package solver.
"""
module LiSupplementAudit
using LinearAlgebra, Printf

# Growth, its two derivatives, and the exact backward characteristic.
function law(kind)
    kind==:constant && return (x->0.1,x->0.0,x->0.0,(x,t)->x-0.1t)
    kind==:affine && return (x->0.1+0.05x,x->0.05,x->0.0,(x,t)->(x+2)*exp(-0.05t)-2)
    kind==:exponential && return (x->0.1exp(0.2x),x->0.02exp(0.2x),x->0.004exp(0.2x),
        (x,t)->-log(exp(-0.2x)+0.02t)/0.2)
    error("unknown audit growth law")
end

"""S11 coefficients in their printed p1,...,p5 order; G′ and G′′ are physical-coordinate derivatives."""
function coefficients(x,h,dt,G,Gp,Gpp;variant=:literal)
    variant in (:literal,:consistent) || throw(ArgumentError("unknown coefficient variant"))
    v=G.(x);s=Gp.(x);d=Gpp.(x);lambda=dt/h;C=v.*lambda
    p1=@. 2/3-C^2/6+h^2*s^2/(3v^2)-(h^2/(6v)+v*dt^2/24)*d
    a=@. v*dt*lambda*s/16-h*s/(6v)
    if variant===:consistent
        # Independent modified-equation correction under the stated one-step
        # interpretation. Keep the literal transcription available for audit.
        @. p1=p1+v*dt^2*d/24
        @. a=a-v*dt*lambda*s/48
    end
    p2=@. 1/6+C/4+C^2/12+a
    p3=@. 1/6-C/4+C^2/12-a
    p4=@. 1/6-C/4+C^2/12+a
    p5=@. 1/6+C/4+C^2/12-a
    (;p1,p2,p3,p4,p5)
end
function factor(c)
    lu(Tridiagonal(c.p3[3:end-1],c.p1[2:end-1],c.p2[2:end-2]))
end
function step!(dest,u,c,factor,work;left=0.0,right=0.0)
    for i in eachindex(work)
        j=i+1
        work[i]=c.p1[j]*u[j]+c.p4[j]*u[j+1]+c.p5[j]*u[j-1]
    end
    work[1]-=c.p3[2]*left;work[end]-=c.p2[end-1]*right
    ldiv!(factor,work)
    dest[1]=left;dest[end]=right;dest[2:end-1].=work
    dest
end

"""Density exact solution: G(x)f(x,t)=G(x₀)f₀(x₀)."""
function density(x,t,kind;center=0.6,width=0.08)
    G,_,_,back=law(kind)
    [G(back(a,t))/G(a)*exp(-((back(a,t)-center)/width)^2) for a in x]
end
function run1d(n,kind;courant=.4,tf=1.,variant=:literal)
    x=collect(range(0,2;length=n+1));h=2/n;G,Gp,Gpp,_=law(kind)
    steps=ceil(Int,tf*maximum(G.(x))/(courant*h));dt=tf/steps
    c=coefficients(x,h,dt,G,Gp,Gpp;variant);fac=factor(c);work=zeros(n-1)
    u=G.(x).*density(x,0.,kind);next=similar(u)
    for k in 1:steps
        boundary=G.(x[[1,end]]).*density(x[[1,end]],k*dt,kind)
        step!(next,u,c,fac,work;left=boundary[1],right=boundary[2]);u,next=next,u
    end
    exact=density(x,tf,kind);computed=u./G.(x)
    (;rmse=sqrt(sum(abs2,computed.-exact)/length(x)),minimum=minimum(computed),
      boundary_maximum=maximum(abs.(exact[[1,end]])),steps,dt)
end
function run2d(n,kind;courant=.4,tf=1.,variant=:literal)
    x=collect(range(0,2;length=n+1));h=2/n;G,Gp,Gpp,_=law(kind)
    steps=ceil(Int,tf*maximum(G.(x))/(courant*h));dt=tf/steps
    c=coefficients(x,h,dt,G,Gp,Gpp;variant);fac=factor(c);work=zeros(n-1)
    weight=G.(x)*G.(x)'
    initial=density(x,0.,kind)*density(x,0.,kind;center=.7,width=.1)'
    u=weight.*initial;next=similar(u)
    for _ in 1:steps
        for j in axes(u,2)
            step!(view(next,:,j),view(u,:,j),c,fac,work)
        end
        for i in axes(u,1)
            step!(view(u,i,:),view(next,i,:),c,fac,work)
        end
    end
    exact=density(x,tf,kind)*density(x,tf,kind;center=.7,width=.1)'
    computed=u./weight
    (;rmse=sqrt(sum(abs2,computed.-exact)/length(exact)),minimum=minimum(computed),steps,dt)
end
function run(;output=joinpath(@__DIR__,"..","..","docs","src","assets","li_supplement"))
    mkpath(output)
    open(joinpath(output,"errors.csv"),"w") do io
        println(io,"dimension,growth,intervals,courant,steps,dt,rmse,observed_order,minimum_density,seconds,status")
        for dimension in (1,2),kind in (:constant,:affine,:exponential),courant in (.4,.8,2.)
            previous=NaN
            for n in (dimension==1 ? (50,100,200,400,800,1600,3200) : (50,100,200,400))
                solver=dimension==1 ? run1d : run2d
                try
                    solver(n,kind;courant) # warm compilation / caches
                    elapsed=@elapsed result=solver(n,kind;courant)
                    order=log2(previous/result.rmse);previous=result.rmse
                    status=isfinite(result.rmse) ? "completed" : "nonfinite"
                    println(io,join((dimension,kind,n,courant,result.steps,result.dt,result.rmse,order,result.minimum,elapsed,status),','))
                    @printf("%dD %s n=%d C=%.1f error=%.5g order=%.3f min=%.3g\n",dimension,kind,n,courant,result.rmse,order,result.minimum)
                catch error
                    error isa SingularException || rethrow()
                    println(io,join((dimension,kind,n,courant,0,NaN,NaN,NaN,NaN,NaN,"singular_directional_matrix"),','))
                    println("Singular directional matrix: dimension=$dimension kind=$kind n=$n courant=$courant")
                end
            end
        end
    end
    open(joinpath(output,"consistent_variant.csv"),"w") do io
        println(io,"dimension,growth,intervals,courant,rmse,observed_order,minimum_density")
        for dimension in (1,2),kind in (:affine,:exponential)
            previous=NaN
            for n in (dimension==1 ? (100,200,400,800,1600) : (50,100,200,400))
                solver=dimension==1 ? run1d : run2d
                result=solver(n,kind;courant=.8,variant=:consistent)
                order=log2(previous/result.rmse);previous=result.rmse
                println(io,join((dimension,kind,n,.8,result.rmse,order,result.minimum),','))
            end
        end
    end
    write(joinpath(output,"inputs.txt"),"Li2026 SI Eq S11 nodal audit; Julia $(VERSION).\n"*
        "Domain [0,2] per axis; tf=1; Gaussian initial density exp(-((x-.6)/.08)^2),\n"*
        "2D second factor exp(-((y-.7)/.1)^2). Laws .1, .1+.05x, .1exp(.2x).\n"*
        "1D exact time-dependent Dirichlet boundaries; 2D zero Dirichlet faces with\n"*
        "negligible exact density there. Gf (1D) or GxGyf (2D) is advanced.\n"*
        "No clipping. Negative undershoots retained. Separate directional S11 updates;\n"*
        "not a reproduction of the main paper multilevel ADI / physical boundary closure.\n"*
        "consistent_variant.csv changes two coefficients by an independently derived\n"*
        "modified-equation correction. It is not a literal paper transcription.\n")
end
"""
Check the literal temporal-stage interpretation of main Eqs. 8 and 16 with
an exact nonzero advection eigenvalue i*omega (no spatial error). This is a
consistency diagnostic for that interpretation, not a full Li ADI solver.
"""
function main_stage_check(;output=joinpath(@__DIR__,"..","..","docs","src","assets","li_supplement","main_stage_check.csv"))
    mkpath(dirname(output));omega=.7
    open(output,"w") do io
        println(io,"tau,omega,normalized_temporal_residual")
        for tau in (.1,.05,.025,.0125,.00625)
            # Eq.16 requires bar(f)^k=(f^{k+1}+f^{k-1})/2=cos(omega*tau)*f^k.
            # Eq.8's literal bar(f)-f^k+tau*K*bar(f)=0 then has this residual.
            c=cos(omega*tau)
            residual=abs((c-1)+im*omega*tau*c)/(omega*tau)
            println(io,join((tau,omega,residual),','))
        end
    end
    output
end
if abspath(PROGRAM_FILE)==@__FILE__
    run()
    main_stage_check()
end
end
