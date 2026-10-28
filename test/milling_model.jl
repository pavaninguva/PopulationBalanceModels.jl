using Test, PopulationBalanceModels, LinearAlgebra
import Distributions
include("../examples/support/milling_model.jl")

@testset "Milling physical model, conservative daughters and algebraic discharge" begin
    p=MillingModel.parameters()
    grid=MillingModel.grid()
    @test first(grid.centers)≈6.54e-17 rtol=2e-15
    @test last(grid.centers)≈1e-3 rtol=2e-14
    d=MillingModel.daughter_projection(grid,2.1,p.daughter_ratio)
    @test minimum(d.eta)>=0
    @test vec(grid.centers'*d.eta)+d.underflow_volume≈grid.centers rtol=2e-12

    # Independently integrate the log-volume daughter law by composite Simpson.
    # Unlike production, this integrates the complete smooth law, without hats.
    z=collect(range(-70.,0.;length=14001));h=z[2]-z[1]
    simpson(y)=h/3*(first(y)+last(y)+4sum(y[2:2:end-1])+2sum(y[3:2:end-2]))
    density=exp.(-.5 .* ((z.+log(p.daughter_ratio))./2.1).^2)./(2.1sqrt(2π))
    m0=simpson(density);m1=simpson(exp.(z).*density)
    @test m1≈MillingModel.relative_moment(1,1.,2.1,p.daughter_ratio) rtol=2e-12
    @test vec(sum(d.eta;dims=1))+d.underflow_number≈fill(m0/m1,length(grid)) rtol=2e-12

    pb=MillingModel.build()
    @test all(iszero,pb.u0)
    @test pb.p.rho*pb.p.number_scale*dot(pb.p.pivots,pb.p.feed)≈.0074 rtol=5e-15
    @test pb.p.resolved_feed_fraction>1-1e-12
    @test_throws NumericalFailure MillingModel.outlet_median(pb.p,zeros(length(grid)),0.)
    @test !MillingModel.admissible(merge(MillingModel.MEAN,(delta=-.01,)))
    @test !MillingModel.admissible(merge(MillingModel.MEAN,(sigma=0.,)))
    @test_throws NumericalFailure MillingModel.build(merge(MillingModel.MEAN,(delta=.999,)))
    @test MillingModel.admissible(merge(MillingModel.MEAN,(gamma=-.5,)))

    # An independently assembled affine system checks every residual and bulk
    # row. In particular the screen removes NET production, not hazard * N.
    N=1e-11.*exp.(-((log.(grid.centers).-log(2e-5))./1.5).^2)
    u=vcat(N,0.,0.);du=similar(u)
    rhs!(du,u,pb.operator,pb.p,12.)
    A,b=MillingModel.linear_system(pb)
    @test du≈A*u+b rtol=2e-14 atol=1e-20
    resolved_rate=pb.p.rho*pb.p.number_scale*dot(grid.centers,du[1:end-2])
    # The arbitrary test inventory makes large breakage/exchange terms cancel.
    # Scale roundoff with those terms, rather than only the much smaller feed.
    balance_scale=abs(resolved_rate)+abs(du[end-1])+p.mass_feed
    @test resolved_rate+du[end-1]≈p.mass_feed atol=64eps()*balance_scale
    @test all(iszero,du[findall(==(0),pb.p.retained)])

    # Compare prepared matrix breakage with the ordinary package FixedPivot
    # event loop, including its analytically accounted sub-grid daughters.
    op=pb.operator.system.populations.mill
    term=op.model.terms[2]
    ordinary=compile(PBMModel(term),grid)
    prepared=compile(PBMModel(term),grid;methods=TermMethods(breakage=MillingModel.MillingFixedPivot()))
    a=zeros(length(N),1);b=similar(a)
    rhs!(a,reshape(N,:,1),ordinary,pb.p,12.)
    rhs!(b,reshape(N,:,1),prepared,pb.p,12.)
    @test a≈b rtol=2e-13 atol=1e-24
    changed=merge(pb.p,(alpha=2pb.p.alpha,))
    @test_throws ArgumentError rhs!(b,reshape(N,:,1),prepared,changed,12.)

    # Independent volume-CDF median for diameters 1,2,3 micrometres.
    # Counts 8,1,1 give volume weights 8,8,27, not number weights 8,1,1.
    volumes=(π/6*1e-18).*[1.,8.,27.]
    synthetic=(pivots=volumes,feed=[8.,1.,1.],retained=zeros(3),Q=zeros(3,3),
        low0=zeros(3),low1=zeros(3),segment_speed=1.)
    expected=2+(43/2-16)/27
    @test MillingModel.outlet_median(synthetic,zeros(3),0.)≈expected atol=2e-14
end
