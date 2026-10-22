using PopulationBalanceModels
using Printf

function run_realizable_axial_transport()
    basis=MomentSet(0:3;coordinate=:length)
    op=compile(PBMModel(),basis;closure=RankAdaptiveQMOM(2;rtol=1e-10))
    localpb=PBMProblem(PBMSystem((particles=op,);nonnegative_bulk=(:solute,)),
        (particles=ones(4),),(0.,.3);bulk0=(solute=3.3,))
    z=AxialGrid([(i/30)^1.2 for i in 0:30])
    feed=(p,t)->(populations=(particles=[2*.4^k for k in 0:3],),bulk=(solute=4-.7*2*.4^3,))
    domain=AxialDispersion(z;velocity=.7,dispersion=.01,inlet=feed)
    pb=axial_problem(localpb,domain;axial_scheme=Upwind())
    u=copy(pb.u0)
    for i in eachindex(z.centers)
        N=1+.2sin(2pi*z.centers[i]);L=.8+.1cos(2pi*z.centers[i])
        population(pb,u,:particles)[:,i].=[N*L^k for k in 0:3]
        bulk(pb,u,:solute)[i]=4-.7*N*L^3
    end
    pb=remake(pb;u0=u)
    sol=simulate(pb;dt=.01,method=MeasureEuler())
    for state in sol.u,i in eachindex(z.centers)
        quadrature(basis,view(population(pb,state,:particles),:,i),RankAdaptiveQMOM(2;rtol=1e-10))
    end
    flux=diagnostics(sol).reactor_flux_integrals
    inventory_error=maximum(abs.(sol.u[end]*z.widths-pb.u0*z.widths-flux.inlet+flux.outlet))
    material_error=maximum(abs.(bulk(pb,sol.u[end],:solute) .+ 0.7 .* vec(population(pb,sol.u[end],:particles)[4,:]) .- 4))
    @assert inventory_error<1e-10 && material_error<1e-10
    @printf("Nonuniform axial transport: inventory error=%.3g; material error=%.3g\n",inventory_error,material_error)
    @printf("Accepted=%d; CFL rejections=%d\n",sol.steps,diagnostics(sol).rejected_steps)
end
if abspath(PROGRAM_FILE)==@__FILE__
    run_realizable_axial_transport()
end
