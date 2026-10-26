include("fixtures/example_modules.jl")
_load_example(:LiSupplementAudit, "verification/li2026_supplement.jl")
using LinearAlgebra
@testset "Literal S11 audit, representation and failure limits" begin
    # Independent analytical characteristics exercise G, G′ and G′′.
    for kind in (:constant,:affine,:exponential),dimension in (1,2)
        solver=dimension==1 ? LiSupplementAudit.run1d : LiSupplementAudit.run2d
        e=[solver(n,kind;courant=.4).rmse for n in (100,200)]
        @test e[1]/e[2]>12
    end
    # A coarse fourth-order-looking curve must not mask the fine-grid result.
    e=[LiSupplementAudit.run1d(n,:affine;courant=.8).rmse for n in (800,1600)]
    @test 1.9<log2(e[1]/e[2])<2.2
    @test_throws LinearAlgebra.SingularException LiSupplementAudit.run1d(200,:constant;courant=2.)
    for kind in (:affine,:exponential)
        e=[LiSupplementAudit.run1d(n,kind;courant=.8,variant=:consistent).rmse for n in (800,1600)]
        @test 3.95<log2(e[1]/e[2])<4.05
        e=[LiSupplementAudit.run2d(n,kind;courant=.8,variant=:consistent).rmse for n in (100,200)]
        @test e[1]/e[2]>12
    end
end
