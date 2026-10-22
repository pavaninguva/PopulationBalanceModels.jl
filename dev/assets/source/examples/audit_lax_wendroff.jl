module LaxWendroffAudit
using PopulationBalanceModels
export reference_step,run_audit

# Independently expand the three coefficients of the MATLAB model_1.m update.
# This is a nodal stencil with zero left ghost and a reflected right ghost.
# It is used for verification, not as a replacement for conservative FV fluxes.
function reference_step(f,C)
    a=C*(C+1)/2;b=1-C^2;c=C*(C-1)/2
    out=similar(f)
    out[1]=b*f[1]+c*f[2]
    for i in 2:length(f)-1;out[i]=a*f[i-1]+b*f[i]+c*f[i+1];end
    out[end]=(1-C^2)*f[end]+C^2*f[end-1]
    out
end

function run_audit(;output=joinpath(@__DIR__,"..","docs","src","assets","constant_growth"))
    mkpath(output)
    open(joinpath(output,"lax_wendroff_audit.csv"),"w") do io
        println(io,"method,courant,points,requested_time,actual_time,minimum_density,maximum_density,interior_step_difference")
        for C in (.5,1.),points in (200,201)
            dx=1/(points-1);dt=C*dx/.1;f0=100 .*exp.(-collect(range(0.,1.;length=points))./.01)
            steps=ceil(Int,4/dt-.001) # MATLAB loop stopping rule, not a shortened last step
            f=copy(f0)
            for _ in 1:steps;f=reference_step(f,C);end
            # Feed identical numerical values into the FV update. Only the
            # interior stencil is compared; nodal and FV boundaries differ.
            g=Grid1D(range(0.,points*dx;length=points+1))
            op=compile(PBMModel(Growth(AffineGrowth(.1,0.))),g;evolution=LaxWendroff(),bc=SizeBC(OpenBoundary(),OpenBoundary()))
            pb=PBMProblem(op,f0.*dx,(0.,4.))
            firststep=vec(simulate(remake(pb;tspan=(0.,dt));dt,check_nonnegative=false).u[end])./dx
            difference=maximum(abs.(firststep[2:end-1].-reference_step(f0,C)[2:end-1]))
            println(io,join(("reference_nodal",C,points,4.,steps*dt,minimum(f),maximum(f),difference),','))
            # A separate physical FV run uses exact initial bin integrals and
            # lands exactly on the requested time.
            N0=exp.(-g.edges[1:end-1]./.01).*(-expm1.(-g.widths./.01))
            out=simulate(PBMProblem(op,N0,(0.,4.));dt,check_nonnegative=false)
            d=vec(last(out.u))./g.widths
            println(io,join(("package_finite_volume",C,points,4.,last(out.t),minimum(d),maximum(d),difference),','))
        end
    end
end
if abspath(PROGRAM_FILE)==@__FILE__;run_audit();end
end
