# Execute inexpensive examples in separate modules; keep generated output in a
# temporary directory. Full paper benchmarks have their own recorded commands.
using Test, PopulationBalanceModels
root=pkgdir(PopulationBalanceModels)
for name in ("batch_population_balance", "interacting_populations", "aggregation_and_breakage",
             "nucleation_sources", "moment_closures", "named_term_methods",
             "periodic_weno_transport", "coupled_moment_balance", "two_coordinate_transport")
    scope=Module(gensym(:Example))
    Base.include(scope,joinpath(root,"examples",name*".jl"))
    @info "Executed standalone example" name
end
@testset "Positive-measure runnable examples" begin
    startup=Module(:StartupExample);Base.include(startup,joinpath(root,"examples","realizable_startup.jl"))
    startup.run_realizable_startup()
    axial=Module(:AxialExample);Base.include(axial,joinpath(root,"examples","realizable_axial_transport.jl"))
    axial.run_realizable_axial_transport()
    dissolving=Module(:DissolutionExample);Base.include(dissolving,joinpath(root,"examples","realizable_dissolution.jl"))
    mktempdir() do output
        dissolving.run_dissolution(;output)
        @test isfile(joinpath(output,"dissolution.csv"))
    end
end
