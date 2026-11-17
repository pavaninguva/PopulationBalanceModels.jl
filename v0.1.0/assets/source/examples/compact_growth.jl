module CompactGrowth
using PopulationBalanceModels, SpecialFunctions

const periodic=SizeBC(Periodic(),Periodic())
const open_ends=SizeBC(OpenBoundary(),OpenBoundary())

"""Smooth periodic translation; initial and exact states are bin integrals."""
function smooth_1d(n=64;time_dependent=false)
    grid=Grid1D(range(0,1;length=n+1);coordinate=:length)
    factor=PrescribedRate((p,t,ctx)->1+t^2)
    law=time_dependent ? TimeScaledGrowth(AffineGrowth(.4,0),factor) : AffineGrowth(.4,0)
    model=PBMModel(growth=Growth(law))
    op=compile(model,grid;evolution=HOCFiniteVolume(),bc=periodic)
    primitive(x,t)=2x+sin(2pi*(x-.4*(time_dependent ? t+t^3/3 : t)))/(2pi)
    initial=diff(primitive.(grid.edges,0.))
    problem=PBMProblem(op,initial,(0.,.5))
    reference(t)=reshape(diff(primitive.(grid.edges,t)),:,1)
    (;problem,reference)
end

"""Unequal velocities on unequal axes; no directional mesh or CFL matching."""
function smooth_2d(nx=48,ny=36)
    grid=Grid2D(range(0,1;length=nx+1),range(0,2;length=ny+1))
    model=PBMModel(growth=Growth2D((.4,.7)))
    bc=SizeBC2D(x=periodic,y=periodic)
    op=compile(model,grid;evolution=HOCFiniteVolume(),bc)
    Fx(x,t)=x+.2sin(2pi*(x-.4t))/(2pi)
    Fy(y,t)=y+.3sin(pi*(y-.7t))/pi
    reference(t)=diff(Fx.(grid.x.edges,t))*diff(Fy.(grid.y.edges,t))'
    problem=PBMProblem(op,reference(0.),(0.,.25))
    (;problem,reference)
end

"""Prescribed incoming number flux and natural outflow, with an exact solution."""
function incoming_population(n=64)
    grid=Grid1D(range(0,1;length=n+1))
    velocity=.4
    incoming=PrescribedRate((p,t,ctx)->velocity*(2+sin(-2pi*velocity*t)))
    bc=SizeBC(InflowFlux(incoming),OpenBoundary())
    model=PBMModel(growth=Growth(AffineGrowth(velocity,0)))
    op=compile(model,grid;evolution=HOCFiniteVolume(),bc)
    primitive(x,t)=2x-cos(2pi*(x-velocity*t))/(2pi)
    reference(t)=reshape(diff(primitive.(grid.edges,t)),:,1)
    problem=PBMProblem(op,reference(0.),(0.,.4))
    (;problem,reference)
end

"""Positive autonomous growth: a Gaussian travels above a steady background."""
function variable_growth(n=80;family=:affine)
    family in (:affine,:exponential) || throw(ArgumentError("choose :affine or :exponential"))
    grid=Grid1D(range(0,2;length=n+1))
    law=family===:affine ? AffineGrowth(.4,.15) : ExponentialGrowth(.4,.15)
    # A stationary number flux of one gives background density 1/G(x).
    background(x)=family===:affine ? log(.4+.15x)/.15 : -exp(-.15x)/(.4*.15)
    gaussian(x)=.12sqrt(pi)/2*erf((x-.65)/.12)
    foot(x,t)=family===:affine ? (x+.4/.15)*exp(-.15t)-.4/.15 : -log(exp(-.15x)+.4*.15t)/.15
    model=PBMModel(growth=Growth(law))
    op=compile(model,grid;evolution=HOCFiniteVolume(),bc=SizeBC(InflowFlux(1.),OpenBoundary()))
    reference(t)=reshape(diff(background.(grid.edges))+diff(gaussian.(max.(0,foot.(grid.edges,t)))),:,1)
    problem=PBMProblem(op,reference(0.),(0.,.4))
    (;problem,reference)
end

"""Different positive spatial laws on a rectangular grid; signed-error diagnostic."""
function variable_2d(nx=48,ny=36)
    grid=Grid2D(range(0,2;length=nx+1),range(0,3;length=ny+1))
    laws=SeparableGrowth2D(AffineGrowth(.4,.1),ExponentialGrowth(.3,.12))
    op=compile(PBMModel(growth=Growth2D(laws)),grid;
        evolution=HOCFiniteVolume(),bc=SizeBC2D(x=open_ends,y=open_ends))
    F(x)=.12sqrt(pi)/2*erf((x-.65)/.12)
    H(y)=.18sqrt(pi)/2*erf((y-.8)/.18)
    fx(x,t)=(x+4)*exp(-.1t)-4
    fy(y,t)=-log(exp(-.12y)+.3*.12t)/.12
    reference(t)=diff(F.(max.(0,fx.(grid.x.edges,t))))*diff(H.(max.(0,fy.(grid.y.edges,t))))'
    problem=PBMProblem(op,reference(0.),(0.,.25))
    (;problem,reference)
end

"""Boundary births are a single incoming number flux; interior births are bin integrals."""
function nucleating_population(n=80;empty=true)
    grid=Grid1D(range(0,2;length=n+1))
    births=PrescribedRate((p,t,ctx)->1+.2sin(2pi*t))
    model=PBMModel(growth=Growth(AffineGrowth(.4,0)),births=Nucleation(births,0.))
    op=compile(model,grid;evolution=HOCFiniteVolume(),bc=open_ends)
    initial=empty ? zeros(n) : 2.5 .*grid.widths
    PBMProblem(op,initial,(0.,.5))
end

function nucleation_reference(problem,t)
    all(iszero,problem.u0) || throw(ArgumentError("this characteristic reference is for the empty initial population"))
    primitive(s)=s+.2*(1-cos(2pi*s))/(2pi)
    edges=problem.operator.grid.edges
    reshape([primitive(max(0.,t-a/.4))-primitive(max(0.,t-b/.4))
        for (a,b) in zip(edges[1:end-1],edges[2:end])],:,1)
end

function run_examples()
    for case in (smooth_1d(),smooth_1d(;time_dependent=true),smooth_2d(),incoming_population(),variable_growth())
        (;problem,reference)=case
        sol=simulate(problem;dt=.0002)
        println((;dimension=problem.operator.grid isa Grid1D ? 1 : 2,
            L1=sum(abs,sol.u[end]-reference(last(sol.t))),
            number=diagnostics(sol).number_accounting[end],steps=sol.steps))
    end
    filled=simulate(nucleating_population(;empty=false);dt=.001)
    println((;positive_nucleation=diagnostics(filled).number_accounting[end],minimum_bin=minimum(filled.u[end])))
    # Singular births generate a moving front. Retain and report its signed
    # dispersive lobes; disabling the check is explicit and performs no clipping.
    sol=simulate(nucleating_population();dt=.001,check_nonnegative=false)
    println((;nucleation=diagnostics(sol).number_accounting[end],minimum_bin=minimum(sol.u[end])))
    case=variable_2d()
    sol=simulate(case.problem;dt=.0002,check_nonnegative=false)
    println((;variable_2d_L1=sum(abs,sol.u[end]-case.reference(.25)),minimum_bin=minimum(sol.u[end])))
end

if abspath(PROGRAM_FILE)==@__FILE__
    run_examples()
end
end
