using Test, PopulationBalanceModels, LinearAlgebra
include("../examples/stem_cell_population.jl")
const SC = StemCellPopulation

@testset "OCT4 division: independent beta moments and initialization" begin
    for q in (1.0,40.0), order in (4,8)
        law = SC.beta_daughters(;q,order)
        for k in 0:6
            # Analytic Beta(q,q) ratio moment, independent of the Jacobi recurrence.
            exact = 2prod((q+j)/(2q+j) for j in 0:k-1;init=1.0)
            @test sum(law.counts .* law.ratios.^k) ≈ exact rtol=3e-13 atol=3e-15
        end
        @test all(>(0),law.counts)
        @test all(r->0<r<1,law.ratios)
    end
    @test_throws ArgumentError SC.beta_daughters(q=0.5)
    @test_throws ArgumentError SC.beta_daughters(order=1)
    errors = Float64[]
    for bins in (300,600)
        grid = Grid1D(range(0.,120.;length=bins+1);coordinate=:oct4)
        cells = SC.initial_population(grid)
        @test sum(cells) ≈ 1 atol=3e-15
        @test moment(grid,cells,1) ≈ 50 atol=1e-12
        @test all(>=(0),cells)
        push!(errors,abs(moment(grid,cells,2)-50^2-0.5^2))
        @test SC.initial_population(grid;concentration=3.) ≈ 3cells
    end
    @test errors[1]/errors[2] ≈ 4 rtol=1e-7
end

@testset "Native cell cohort, phenotype exits and distinct truncation accounting" begin
    for different in (false,true)
        solution = SC.solve_case(bins=100,tf=4.,differentiation=different,saveat=[0.,2.,4.])
        problem = solution.problem
        grid = problem.operator.system.populations.stem.grid
        @test term_methods(problem.operator.system.populations.stem).expression isa MUSCL
        @test diagnostics(solution).policy == :SSPRK33
        for (t,u) in zip(solution.t,solution.u)
            stats = SC.state_summary(problem,u,t)
            @test stats.minimum_count >= 0
            @test abs(stats.number_balance) < 3e-13
            @test abs(stats.cargo_balance) < 3e-11
        end
        @test bulk(problem,last(solution.u),:differentiated_cells) > 0 || !different
        different || @test bulk(problem,last(solution.u),:differentiated_cells) == 0
        events = SC.event_distributions(problem,last(solution.u),4.)
        @test sum(events.total) ≈ 1
        @test sum(events.dividing) ≈ 1
        @test sum(events.newborn) ≈ 1 atol=2e-13
        @test events.unresolved_newborn_fraction < 1e-12
        @test sum(grid.centers .* events.newborn) ≈ sum(grid.centers .* events.dividing)/2 rtol=1e-12
        @test isnothing(events.differentiating) == !different
        different && @test sum(events.differentiating) ≈ 1
    end
    # All daughters of a first-pivot cell are unresolved. The ledger must
    # expose their number AND true cargo, rather than manufacture a population.
    problem = SC.build_problem(bins=20,tf=1.,differentiation=true)
    u = zero(problem.u0)
    population(problem,u,:stem)[1] = 5.
    du = similar(u); rhs!(du,u,problem.operator,problem.p,0.)
    x = first(problem.operator.system.populations.stem.grid.centers)
    rate = SC.division_rate(x,problem.p,0.,nothing)*5
    exit_rate = SC.differentiation_rate(x,problem.p,0.,nothing)*5
    @test cumulative(problem,du,:stem,:breakage_underflow) ≈ 2rate
    @test cumulative(problem,du,:stem,:breakage_underflow,:cargo) ≈ x*rate
    @test bulk(problem,du,:differentiated_cells) ≈ exit_rate
    @test bulk(problem,du,:differentiated_cargo) ≈ x*exit_rate
    @test sum(population(problem,du,:stem)) + exit_rate + 2rate ≈ rate
end

@testset "Independent binary-division cascade with differentiation" begin
    # Only powers-of-two pivots are populated. Equal division therefore needs
    # no interpolation; the finite cascade has a closed form at every level.
    gamma, kappa, horizon = 0.4, 0.1, 2.0
    grid = Grid1D(0.5:1.:16.5)
    model = PBMModel(division=Breakage((x,p,t,ctx)->gamma,
        SelfSimilarDaughters((0.5,),(2.,));overflow=:discard),
        differentiation=Death((x,p,t,ctx)->kappa))
    pop = compile(model,grid)
    system = PBMSystem((stem=pop,);accounting=(CumulativeAccounting(:stem;properties=(cargo=1,)),),
        bulk_rhs=(ctx,p,t)->(exits=kappa*sum(ctx.populations.stem),
            exit_cargo=kappa*sum(grid.centers .* ctx.populations.stem),
            divisions=gamma*sum(ctx.populations.stem)))
    initial = zeros(16);initial[16]=1.
    problem = PBMProblem(system,(stem=initial,),(0.,horizon);
        bulk0=(exits=0.,exit_cargo=0.,divisions=0.))
    a = gamma+kappa
    exact = zeros(16)
    integrated = zeros(16)
    for j in 0:4
        pivot = 16÷2^j
        exact[pivot] = exp(-a*horizon)*(2gamma*horizon)^j/factorial(j)
        integral = factorial(j)/a^(j+1)*(1-exp(-a*horizon)*sum((a*horizon)^m/factorial(m) for m in 0:j))
        integrated[pivot] = (2gamma)^j/factorial(j)*integral
    end
    errors = Float64[]
    for dt in (0.05,0.025)
        solution = simulate(problem;dt,method=SSPRK33(),negative_tolerance=0.)
        u = last(solution.u)
        push!(errors,maximum(abs,population(problem,u,:stem)-exact))
        @test bulk(problem,u,:exits) ≈ kappa*sum(integrated) atol=2e-7
        @test bulk(problem,u,:exit_cargo) ≈ kappa*sum(grid.centers .* integrated) atol=2e-6
        @test cumulative(problem,u,:stem,:breakage_underflow) ≈ 2gamma*integrated[1] atol=2e-7
        @test cumulative(problem,u,:stem,:breakage_underflow,:cargo) ≈ gamma*integrated[1] atol=1e-7
        @test sum(population(problem,u,:stem))+bulk(problem,u,:exits)+
            cumulative(problem,u,:stem,:breakage_underflow)-bulk(problem,u,:divisions) ≈ 1 atol=3e-14
        @test moment(grid,population(problem,u,:stem),1)+bulk(problem,u,:exit_cargo)+
            cumulative(problem,u,:stem,:breakage_underflow,:cargo) ≈ 16 atol=3e-13
    end
    @test 7.5 < errors[1]/errors[2] < 8.5
end
