@testset "Stationary problem contracts (without optional solver)" begin
    g=Grid1D(range(0,2;length=9))
    op=compile(PBMModel(Birth((x,p,t,c)->p.b)+Death((x,p,t,c)->p.d)),g)
    guess=copy(g.widths)
    @test_throws ArgumentError StationaryPBMProblem(op,guess)
    @test_throws ArgumentError StationaryPBMProblem(op,guess;autonomous=true,time=0)
    @test_throws ArgumentError StationaryPBMProblem(op,guess;autonomous=true,residual_scale=0)
    @test_throws ArgumentError StationaryPBMProblem(op,guess;time=:unknown)
    @test_throws NumericalFailure StationaryPBMProblem(op,-guess;autonomous=true)
    prob=StationaryPBMProblem(op,guess;p=(b=2.,d=.5),autonomous=true,residual_scale=2)
    r=similar(prob.guess); stationary_residual!(r,prob.guess,prob)
    @test vec(r)≈.75g.widths
    @test guess==g.widths # neither input packing nor residual changes guesses
    changed=remake(prob;p=(b=1.,d=1.))
    stationary_residual!(r,changed.guess,changed)
    @test iszero(r)
    @test prob.p==(b=2.,d=.5)
    @test StationaryPBMProblem(op,guess;autonomous=true,residual_scale=ones(8)).residual_scale==ones(8,1)
    @test_throws ArgumentError remake(prob;constraints=(1=>(u,p,t)->0,1=>(u,p,t)->0))
    timed=compile(PBMModel(Birth((x,p,t,c)->t)),g)
    frozen=StationaryPBMProblem(timed,guess;time=3)
    stationary_residual!(r,frozen.guess,frozen)
    @test vec(r)≈3g.widths
    direct=compile(PBMModel(Growth(AffineGrowth(1.,0.))),g;
        evolution=HOCFiniteVolume(),bc=SizeBC(Periodic(),Periodic()))
    @test_throws ArgumentError StationaryPBMProblem(direct,guess;autonomous=true)
    mo=compile(PBMModel(),MomentSet(0:3);boundary=MomentBoundary(left=:absorb))
    @test_throws ArgumentError StationaryPBMProblem(mo,[1.,1.,1.,1.];autonomous=true)
end
