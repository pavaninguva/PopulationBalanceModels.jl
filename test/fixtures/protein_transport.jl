function protein_transport_fixture()
    coefficient=[-22.2 -3.35 1.1 2.2;
                 .1 .03 -.2 .4;
                 -.02 .01 .05 -.1;
                 0. .002 .03 .02]
    low=Matrix(Diagonal([.1,.05,.4,.3]));low[2,1]=.03;low[4,2]=.12
    high=Matrix(Diagonal([.2,.07,.3,.25]));high[3,1]=.1;high[4,2]=-.08
    (;mixture_means=[-1.,1.5],mixture_scales=[.2,1.],mixture_weights=[.2,.8],
      polynomial_coefficients=coefficient,low_cholesky=low,high_cholesky=high,
      low_mean=[.02,-.01,.03,-.02],high_mean=[-.01,.02,-.02,.01],
      parent_center=70.,parent_scale=30.,blend_center=37.,blend_width=3.)
end

# Historical five-coordinate sampler contract, deliberately tested on a cheap
# parameter-observation target, not a second scientific PBM implementation.
module LegacyProteinCoordinates
using PopulationBalanceModels
using Distributions: Uniform,LogNormal,Exponential
const C0=19.965841161400512
barrier(gamma)=16pi*(1e-3gamma)^3*(2.97e-26)^2/(3*(1.380649e-23*293.15)^3)
parameters(;logAJ=41.,gamma=.76,AG=1.,g=2.4,grd_strength=10.)=(;logAJ,gamma,AG,g,grd_strength)
function parameter_specs(;p=parameters(),model=:cnt_grd)
    specs=(ParameterSpec(:logAJ;initial=p.logAJ,transform=LogitTransform(0.,100.),
        prior=Prior(Uniform(0.,100.);coordinates=:physical)),
      ParameterSpec(:gamma;initial=p.gamma,transform=LogTransform(),
        prior=Prior(LogNormal(log(.64),.7);coordinates=:physical)),
      ParameterSpec(:AG;initial=p.AG,transform=LogTransform(),
        prior=Prior(Uniform(log(.01),log(1500.));coordinates=:inference)),
      ParameterSpec(:g;initial=p.g,transform=LogitTransform(1.,3.),
        prior=Prior(Uniform(1.,3.);coordinates=:physical)))
    model===:cnt ? specs : (specs...,ParameterSpec(:grd_strength;initial=p.grd_strength,
        transform=LogTransform(),prior=Prior(Exponential(10.);coordinates=:physical)))
end
function estimation_problem(;model=:cnt_grd,bins=30)
    specs=parameter_specs(;model)
    op=compile(PBMModel(),Grid1D([0.,.5,1.]))
    pb=PBMProblem(op,[0.,0.],(0.,1.))
    ob=FunctionObservable((u,op,p,t)->[getproperty(p,s.name) for s in specs])
    m=Measurement([1.],zeros(length(specs),1),ob;likelihood=GaussianLikelihood(sigma=.01))
    e=InferenceExperiment(:batch,pb,m;solver=(p,ts)->PBMSolution(ts,[copy(p.u0) for _ in ts],p,0))
    EstimationProblem(specs,(e,);fixed=model===:cnt ? (cstar=2.16,grd_strength=0.) : (cstar=2.16,))
end
end
