# JET is a development-only dependency. Run in the isolated CI environment.
using PopulationBalanceModels, Test, JET
@test JET.JET_AVAILABLE
@testset "Selected concrete residuals" begin
    model = PBMModel(growth=Growth((x,p,t,c)->p.G), death=Death((x,p,t,c)->p.d))
    op = compile(model, Grid1D(0.:1.:10.))
    pb = PBMProblem(op, ones(10), (0.,1.); p=(G=.1,d=.2))
    du = zero(pb.u0)
    JET.@test_call target_modules=(PopulationBalanceModels,) rhs!(du,pb.u0,op,pb.p,0.)

    op = compile(PBMModel(Growth(AffineGrowth(.1,.2))), MomentSet(0:3))
    pb = PBMProblem(op, [1.,2.,4.,8.], (0.,1.))
    du = zero(pb.u0)
    JET.@test_call target_modules=(PopulationBalanceModels,) rhs!(du,pb.u0,op,pb.p,0.)
end
