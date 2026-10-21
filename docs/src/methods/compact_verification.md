# Compact-method verification

A compact stencil obtains a derivative or a new time level by solving a banded
linear system. This can give high accuracy with few neighboring points, but it
does not by itself guarantee stable boundaries, positivity, or the advertised
order for variable coefficients. We check those properties separately.

## What HOCADI currently solves

The public `HOCADI` method is the periodic constant-velocity compact/Cayley
variant specified on the [time-integration page](time.md). Its Fourier factors
have unit modulus. The existing dense-matrix comparisons and long-period tests
check that implementation. Negative lobes near a sharp front remain possible.
It does not implement the size-dependent algorithm in
[Li et al. (2026)](https://doi.org/10.1016/j.partic.2026.01.024).

## Check the supplement independently

`examples/audit_li2026_supplement.jl` transcribes the five coefficients of the
supplement's Eq. S11. For one size coordinate the physical equation is

```math
\frac{\partial f(a,t)}{\partial t}
+\frac{\partial [G(a)f(a,t)]}{\partial a}=0.
```

For positive, autonomous growth, defining ``u(a,t)=G(a)f(a,t)`` gives

```math
\frac{\partial u(a,t)}{\partial t}
+G(a)\frac{\partial u(a,t)}{\partial a}=0.
```

The printed S11 step is

```math
p_{1,i}u_i^{n+1}+p_{2,i}u_{i+1}^{n+1}+p_{3,i}u_{i-1}^{n+1}
=p_{1,i}u_i^n+p_{4,i}u_{i+1}^n+p_{5,i}u_{i-1}^n.
```

The audit treats ``\tau`` as the time step and ``\lambda=\tau/h``. It uses
**nodal densities** on a uniform mesh. Package `Grid1D` and `Grid2D` states are
bin integrals, so applying these coefficients directly to those states would
introduce a separate representation error. The audit is deliberately separate
from `compile` and `simulate`.

We test three laws on ``0\le a\le2``:
``G=0.1``, ``G=0.1+0.05a``, and ``G=0.1\exp(0.2a)``. The last law exercises
both first and second growth derivatives in the printed coefficients. The
initial density is ``\exp[-((a-0.6)/0.08)^2]``. The exact comparison follows
characteristics:

```math
\frac{da(t)}{dt}=G(a(t)),\qquad
f(a,t)=\frac{G(a_0)}{G(a)}f(a_0,0).
```

The one-dimensional run supplies exact time-dependent Dirichlet endpoints.
For two dimensions, separable rates allow ``u=G_xG_yf`` and a product of the two
S11 directional updates. A second Gaussian factor is centered at 0.7 with width
0.1. Its exact solution is negligible at all four faces over ``0\le t\le1``;
that diagnostic uses zero Dirichlet faces. This directional product is not a
transcription of the main paper's multilevel ADI equations or its physical
boundary closure.

## Results and unresolved issues

![Refinement of the literal S11 coefficients and the size-dependent order change](../assets/li_supplement/refinement.png)

At a maximum Courant number of 0.4, the 50–400-interval results initially look
fourth order for all three laws. Extending the one-dimensional test to 3200
intervals changes the conclusion: the variable-growth errors eventually
approach second order. For ``G=0.1+0.05a`` at a maximum Courant number of 0.8,
the last two refinement slopes are approximately 2.03 and 2.00. The exponential
law gives approximately 2.04 and 1.99. The constant-growth test continues to
show fourth-order behavior before roundoff dominates.

The maximum Courant number is a requested bound; the actual time step is
shortened uniformly to reach the final time. The CSV records the actual step.
At a requested bound of 2, this Dirichlet implementation can become singular
or amplify errors enormously. For example, with constant growth and an actual
Courant number of 2, S11 has ``p_1=0`` and ``p_3=0``. The interior tridiagonal
matrix with both endpoint values prescribed is singular. This is a concrete
boundary/step limitation of this transcription, not evidence that every
compact or periodic implementation is unstable.

The main paper's multilevel equations, the relationship to S11, and its
high-order physical boundary closure therefore still need reconciliation.
We do not substitute altered coefficients or claim general fourth-order
size-dependent support. The script retains singular failures, nonfinite
errors, negative minima, and coarse-grid oscillations in
[errors.csv](../assets/li_supplement/errors.csv).

```sh
julia --project=. examples/audit_li2026_supplement.jl
python docs/benchmarks/plot_compact_audit.py --usetex
```

## A separately specified accuracy correction

The audit also contains an independently derived coefficient variant. It keeps
the literal S11 result available for comparison. Under the one-step
interpretation above, substituting a smooth exact solution into S11 leaves the
following leading time-error contribution after division by the time step:

```math
\frac{\tau^2}{24}\left[
\big(G^2G''-G(G')^2\big)\frac{\partial u}{\partial a}
-G^2G'\frac{\partial^2u}{\partial a^2}\right].
```

This term vanishes for constant growth and generally remains for variable
growth. It explains why a constant-growth test did not expose the order loss.
Writing ``\lambda=\tau/h``, the independent variant changes the printed
coefficients by

```math
\widetilde p_1=p_1+\frac{G\tau^2G''}{24},\qquad
\delta=\frac{G\tau\lambda G'}{48},
```

```math
\widetilde p_2=p_2-\delta,\quad
\widetilde p_3=p_3+\delta,\quad
\widetilde p_4=p_4-\delta,\quad
\widetilde p_5=p_5+\delta.
```

These changes cancel that residual; they are not printed coefficients attributed
to Li et al. Select `variant=:consistent` in the audit's `run1d` or `run2d`.
For affine and exponential growth, the fine one-dimensional slopes return to
approximately 4.00, and the two-dimensional product tests also approach fourth
order. The values are retained in
[consistent_variant.csv](../assets/li_supplement/consistent_variant.csv).

This is evidence for a specified smooth nodal variant over the tested regime.
It does not resolve the main paper's multilevel interpretation, the singular
large-step Dirichlet case, positivity, or a general physical boundary closure.
Neither audit variant is exposed as a production `compile` method. The public
periodic `HOCADI` behavior remains unchanged.

## What the other supplied papers add

The literature review also identifies schemes with distinct implementation
requirements:

| Source | Consequence for this package |
|:--|:--|
| [Zhang et al., CADET Part I (2024)](https://doi.org/10.1016/j.compchemeng.2024.108612) | Two-stencil third-order WENO and van Leer limiting motivate the new `WENO3` and `VanLeer` growth choices. Local cell geometry is used on nonuniform 1D meshes. These are not a CADET backend. |
| [Ketcheson et al., PyClaw (2012)](https://doi.org/10.1137/110856976) and the supplied Clawpack paper | Wave-propagation methods are a separate solver family. The package's conservative face-flux methods do not implement Clawpack's Riemann-solver machinery. |
| Kumar et al., Powder Technology 189 (2009), 218–229 | Cell-average and conservative finite-volume collision algorithms differ from the implemented fixed-pivot event deposition. They need their own number/volume and overflow tests before being exposed. |
| Zhang et al., CADET Part II (2025), CCE 192, 108860 | Weighted finite-volume and moment-conserving collision methods are further candidates. A shared collision interface does not mean these algorithms are already implemented. |
| Inguva et al., Biotechnology Advances 84 (2025), 108643 | Additional nanoparticle physics motivates explicit material closures. CFD and phase-field coupling remain outside the active package scope. |

New growth choices are available through `TermMethods(growth=...)` and tested
for signed velocities and number conservation. Their formal orders and boundary
fallbacks are described on the [growth page](growth.md); whole-solve convergence
is measured in the [constant-growth comparison](../case_studies/constant_growth.md).
