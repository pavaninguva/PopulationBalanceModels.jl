@testset "SizeDiffusion SciML residual and coupled exchange" begin
    g = Grid1D([0.,.02,.07,.2,.4,.7,1.];coordinate=:length)
    op = compile(PBMModel(growth=Growth((L,p,t,c)->.01),
        dispersion=SizeDiffusion((L,p,t,c)->p.D*(1+L)*c.bulk.c/10)),g)
    sys = PBMSystem((A=op,);exchanges=(ConservativeExchange(:A,:c,(L,p,t,c)->L^3),))
    pb = PBMProblem(sys,(A=copy(g.widths),),(0.,.05);bulk0=(c=10.,),p=(D=.02,))
    ode = odeproblem(pb)
    adaptive = SciMLBase.solve(ode,Tsit5();reltol=1e-10,abstol=1e-12)
    native = simulate(pb;dt=.00005)
    @test SciMLBase.successful_retcode(adaptive)
    @test adaptive.u[end] ≈ native.u[end] rtol=2e-9
    @test bulk(pb,adaptive.u[end],:c)+moment(g,population(pb,adaptive.u[end],:A),3) ≈
        10+moment(g,g.widths,3) atol=2e-13
end
