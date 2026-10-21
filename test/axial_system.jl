@testset "Axial populations and solute" begin
    grid=Grid1D([0.,.3,1.,2.]);z=AxialGrid([0.,.2,.5,1.])
    model=PBMModel(Growth((x,p,t,ctx)->.1*ctx.bulk.c))
    op=compile(model,grid)
    sys=PBMSystem((particles=op,);nonnegative_bulk=(:c,),exchanges=(ConservativeExchange(:particles,:c,(x,p,t,ctx)->x^3),))
    pb=PBMProblem(sys,(particles=[1.,2.,1.],),(0.,.1);bulk0=(c=3.,))
    feed=(p,t)->(populations=(particles=[.5,1.,2.],),bulk=(c=4.,))
    for boundary in (:danckwerts,:periodic,:closed),scheme in (Upwind(),MUSCL(),WENO3())
        d=AxialDispersion(z;velocity=boundary===:closed ? 0. : .3,dispersion=.02,
            boundary,inlet=boundary===:danckwerts ? feed : nothing)
        ap=axial_problem(pb,d;axial_scheme=scheme)
        u=ap.u0.*[.7,1.,1.2]';du=similar(u);rhs!(du,u,ap.operator,(;),0.)
        F=axial_fluxes(ap.operator,u,(;),0.)
        weights=vcat(grid.centers.^3,1.)
        # Local precipitation exchanges cancel exactly; only boundary flux remains.
        @test sum(weights.*du.*z.widths')≈sum(weights.*(F[:,1]-F[:,end])) atol=1e-12
        @test axial_stateview(ap,u,2).bulk.c==bulk(ap,u,:c)[2]
        sol=simulate(ap;dt=.001)
        @test minimum(sol.u[end])>=0
    end
    @test_throws ArgumentError axial_problem(pb,AxialDispersion(z;velocity=1.,dispersion=0.);axial_scheme=WENO5())
    wrong=AxialDispersion(z;velocity=1.,dispersion=0.,inlet=(p,t)->(populations=(particles=[1.,2.],),bulk=(c=4.,)))
    ap=axial_problem(pb,wrong)
    @test_throws DimensionMismatch axial_fluxes(ap.operator,ap.u0,(;),0.)
    # A spatially dependent field reaches each local reaction callback.
    sop=compile(PBMModel(Birth((x,p,t,ctx)->ctx.fields.z)),grid)
    localpb=PBMProblem(PBMSystem((particles=sop,)),(particles=zeros(3),),(0.,1.))
    ap=axial_problem(localpb,AxialDispersion(z;velocity=0.,dispersion=0.))
    du=similar(ap.u0);rhs!(du,ap.u0,ap.operator,(;),0.)
    @test du≈grid.widths*z.centers'
    # Raw moments are conserved transport variables; DQMOM's w*x states are not.
    m=DQMOM(1);mp=compile(PBMModel(),MomentSet(0:1);method=m)
    direct=PBMProblem(PBMSystem((p=mp,)),(p=quadrature_state(m,[1.],[1.]),),(0.,1.))
    @test_throws ArgumentError axial_problem(direct,AxialDispersion(z;velocity=0.,dispersion=0.))
end
