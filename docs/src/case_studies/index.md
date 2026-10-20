# Case studies

The three studies follow the same sequence: write the population balance,
identify the initial and boundary conditions, define its rates in Julia, select
term methods, solve, and compare the resulting distributions or observations.

1. [Batch cooling crystallization](crystallization.md), Example 2 in Gunawan,
   Fusman, and Braatz (2004): a seeded crystal distribution coupled to the
   dissolved-solute concentration.
2. [Lipid nanoparticle formation](nanoparticles.md), Shin et al. (2025): a
   population initially containing no particles, with nucleation, growth,
   coalescence, and the model outputs shown in Figure 4.
3. [A two-coordinate population balance](transport.md), Case 2 in Inguva and
   Braatz (2023): size-dependent growth with an analytical solution.

Each page gives the equations in full derivative notation and shows how the
terms, numerical methods, and initial conditions enter the code. The scripts
export CSV data; the plotting commands can be run locally after a calculation.
The documentation build does not run the expensive simulations.
