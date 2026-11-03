# Zhang Case 7A: metres, seconds, kg/m^3; particles are stored / 10^12.
using PopulationBalanceModels
include(joinpath(@__DIR__,"..","support","precipitation_numerics.jl"))

p = (cfeed=9.0,ceq=0.4,rho=1200.0,kv=0.524,kp=5.0,u=10.0,kb=4e8,b=2.0,
     kg=7e-6,xc=1e-6,xmax=900e-6,Q=1.67e-7,area=1.46e-4,porosity=0.21,L=0.47,Dax=4.2e-5)
supersaturation(p,ctx) = max(ctx.bulk.c/p.ceq-1,zero(ctx.bulk.c))
growth(x,p,t,ctx) = p.kg*supersaturation(p,ctx)
function nucleation(p,t,ctx)
    s = supersaturation(p,ctx)
    solid = 1e12*p.rho*p.kv*moment(ctx.grid,ctx.N,3)
    (p.kp*s^p.u+p.kb*s^p.b*solid)/1e12
end
nx,nz = 50,50 # Coarse startup demonstration, not the retained reference.
edges = exp.(range(log(p.xc),log(p.xmax);length=nx+1))
model = PBMModel(growth=Growth(growth),primary_secondary=Nucleation(nucleation,first(edges)))
grid = Grid1D(edges;coordinate=:length)
operator = compile(model,grid;methods=TermMethods(
    growth=WENO5(),nucleation=BoundaryInjection(:left)),bc=SizeBC())
exchange = ConservativeExchange(:crystals,:c,(x,p,t,ctx)->1e12*p.rho*p.kv*x^3)
system = PBMSystem((crystals=operator,);exchanges=(exchange,),nonnegative_bulk=(:c,))
local_problem = PBMProblem(system,(crystals=zeros(nx),),(0.0,5.0);bulk0=(c=0.0,),p)
reactor = AxialDispersion(AxialGrid(range(0.0,p.L;length=nz+1));
    velocity=p.Q/(p.area*p.porosity),dispersion=p.Dax,
    inlet=(p,t)->(populations=(crystals=zeros(nx),),bulk=(c=p.cfeed,)))
problem = axial_problem(local_problem,reactor;axial_scheme=MUSCL())

# Preserve the transport/depletion restriction and native stage checks. The
# companion implements a physical step bound, not integration-error control.
bound = (pb,u,t)->precipitation_step_bound(pb,u,t;fraction=0.3)
solver = ForwardSolver(RK4();dt=0.25,negative_tolerance=1e-8,
    control=StepControl(;cfl=nothing,bound,min_dt=1e-12))
if abspath(PROGRAM_FILE) == @__FILE__
    solution = simulate(problem;solver,saveat=[0.0,5.0])
    @show observe(CellObservable(BulkObservable(:c),nz),solution)
    # Optional: import PythonPlot. Plot labels retain the /10^12 state scaling.
    # plot_field(solution;population=:crystals,orientation=:axial_x,
    #     xlabel="Axial position (m)",ylabel="Size (m)",colorbar_label="Scaled number density")
    # plot_distribution(solution;population=:crystals,cell=nz,times=[5.0],
    #     xlabel="Size (m)",ylabel="Scaled number density")
end
