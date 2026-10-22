# Independent analytical references for the audit, not package inversion code.
module QamarReference
using LinearAlgebra, SpecialFunctions
export characteristic_moments, gamma_certificate, lognormal_breakage_rates, gamma_width_roots
function characteristic_moments(t,K=6)
    t=BigFloat(t);h=BigFloat(100);c=BigFloat("0.215");d=t-c
    init(k)=(BigFloat("0.01")*BigFloat(2)^(k+1)+BigFloat("99.99")*(BigFloat("0.6")^(k+1)-BigFloat("0.4")^(k+1)))/(k+1)
    a=-h*c;b=h*d
    integrals=BigFloat[sqrt(big(pi))/2*(b<=0 ? erfc(-b)-erfc(-a) : erf(b)-erf(a))]
    K>=1 && push!(integrals,(exp(-a*a)-exp(-b*b))/2)
    for j in 2:K
        push!(integrals,((j-1)*integrals[j-1]+a^(j-1)*exp(-a*a)-b^(j-1)*exp(-b*b))/2)
    end
    [sum(binomial(k,j)*t^(k-j)*init(j) for j in 0:k)+100*t^(k+1)/(k+1)+
        10^6/h*sum(binomial(k,j)*d^(k-j)*(-1/h)^j*integrals[j+1] for j in 0:k) for k in 0:K]
end
function deconv(v,s)
    K=length(v)-1;S=zeros(BigInt,K+1,K+1);S[1,1]=1
    for k in 1:K,j in 1:k
        S[k+1,j+1]=S[k,j]+j*S[k,j+1]
    end
    [k==0 ? v[1] : sum(S[k+1,j+1]*(-s)^(k-j)*v[j+1] for j in 1:k) for k in 0:K]
end
function margin(v,n,shift=0)
    A=[v[i+j-1+shift]/sqrt(abs(v[2i-1+shift]*v[2j-1+shift])) for i in 1:n,j in 1:n]
    L=Matrix{BigFloat}(I,n,n);D=zeros(BigFloat,n)
    for k in 1:n
        D[k]=A[k,k]-sum((L[k,j]^2*D[j] for j in 1:k-1);init=BigFloat(0))
        D[k]<=0 && return D[k]
        for i in k+1:n
            L[i,k]=(A[i,k]-sum((L[i,j]*L[k,j]*D[j] for j in 1:k-1);init=BigFloat(0)))/D[k]
        end
    end
    minimum(D)
end

"""Independent Stirling-second-kind deconvolution and BigFloat LDL boundary search.
The final small Jacobi matrix is converted only AFTER the critical width solve.
A small error certifies a positive Gamma mixture; a large error alone does not
prove that every possible common width is infeasible.
"""
function gamma_certificate(mu,n=3;bits=256)
    setprecision(BigFloat,bits) do
        b=BigFloat.(mu); scale=b[2]/b[1];v=b./b[1]./scale.^(0:length(b)-1)
        lo=BigFloat(0);hi=(v[3]-v[2]^2)/v[2]
        for _ in 1:bits-16
            mid=(lo+hi)/2;r=deconv(v,mid)
            all(>=(0),r)&&margin(r,n+1)>0&&margin(r,n,1)>0 ? (lo=mid) : (hi=mid)
        end
        sigma=(lo+hi)/2;r=deconv(v,sigma)
        A=Symmetric([r[i+j-1] for i in 1:n,j in 1:n]);B=[r[i+j] for i in 1:n,j in 1:n]
        L=cholesky(A).L;J=L\B/L';q=eigen(Symmetric(Float64.(J)))
        x=BigFloat.(q.values);w=BigFloat.(q.vectors[1,:].^2)
        errors=[abs(sum(w.*[prod((xi+j*sigma for j in 0:k-1);init=big(1.)) for xi in x])-v[k+1])/max(1,abs(v[k+1])) for k in 0:length(v)-1]
        h0=[r[i+j-1] for i in 1:n+1,j in 1:n+1]
        (;width=sigma*scale,normalized_width=sigma,nodes=x.*scale,weights=w.*b[1],
          error=maximum(errors),errors,condition=cond(Float64.(A)),h0det=det(h0),
          shifted_margin=margin(r,n,1),primary_margin=margin(r,n),normalized=v)
    end
end

# Uniform-volume daughters have integral x^k b(x|y) dx = 6y^k/(k+3).
# For a lognormal law E[x^j]=mean^j*exp(j(j-1)*v/2), exactly.
lognormal_breakage_rates(N,mean,v)=
    [(6/(k+3)-1)*N*mean^(k+6)*exp((k+6)*(k+5)*v/2) for k in 0:2]

# Polynomial determinant isolation over the full physically allowed width interval.
padd(a,b)= [ (i<=length(a) ? a[i] : big(0.))+(i<=length(b) ? b[i] : big(0.)) for i in 1:max(length(a),length(b)) ]
function pmul(a,b)
    c=zeros(BigFloat,length(a)+length(b)-1)
    for i in eachindex(a),j in eachindex(b);c[i+j-1]+=a[i]*b[j];end;c
end
function pdet(A)
    n=size(A,1);n==1&&return A[1,1]
    out=BigFloat[0]
    for j in 1:n
        sub=A[2:n,setdiff(1:n,[j])]
        out=padd(out,(-1)^(j+1).*pmul(A[1,j],pdet(sub)))
    end
    out
end
function hpoly(v,upper)
    K=length(v)-1;S=zeros(BigInt,K+1,K+1);S[1,1]=1
    for k in 1:K,j in 1:k;S[k+1,j+1]=S[k,j]+j*S[k,j+1];end
    r=[k==0 ? BigFloat[v[1]] : [pow<k ? S[k+1,k-pow+1]*(-upper)^pow*v[k-pow+1] : big(0.) for pow in 0:k] for k in 0:K]
    p=pdet([r[i+j-1] for i in 1:4,j in 1:4])
    while length(p)>1&&abs(p[end])<big"1e-60"*maximum(abs,p);pop!(p);end
    p
end
function splitbern(b)
    l=[first(b)];r=[last(b)];c=copy(b)
    while length(c)>1
        c=(c[1:end-1]+c[2:end])/2;push!(l,first(c));pushfirst!(r,last(c))
    end
    l,r
end
function isolate(b,a=big(0.),c=big(1.),depth=0)
    signs=sign.(filter(x->abs(x)>big"1e-65"*maximum(abs,b),b));variations=sum(signs[1:end-1].!=signs[2:end])
    variations==0 && return Tuple{BigFloat,BigFloat}[]
    depth>=100 && return [(a,c)]
    l,r=splitbern(b);m=(a+c)/2;vcat(isolate(l,a,m,depth+1),isolate(r,m,c,depth+1))
end

function gamma_width_roots(mu;bits=256)
    setprecision(BigFloat,bits) do
        b=BigFloat.(mu);v=b./b[1]./(b[2]/b[1]).^(0:6);upper=v[3]-1
        p=hpoly(v,upper);d=length(p)-1
        bern=[sum(p[k+1]*binomial(big(i),big(k))/binomial(big(d),big(k)) for k in 0:i) for i in 0:d]
        [begin
            width=(a+c)/2*upper;r=deconv(v,width)
            (;width,shifted_margin=margin(r,3,1),primary_margin=margin(r,3),
              determinant=evalpoly(width/upper,p),degree=d)
        end for (a,c) in isolate(bern)]
    end
end
end
