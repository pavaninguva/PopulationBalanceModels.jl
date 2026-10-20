# Case studies

These three studies test different parts of the model-to-solver path.

1. [Two-coordinate transport](transport.md) uses the Gaussian constant-growth
   problem in Inguva and Braatz (2023). An analytical solution separates numerical
   transport error from physical modeling uncertainty. It includes 2D solution
   plots and a separate compact/ADI stability audit.
2. [Cooling crystallization](crystallization.md) uses the coupled growth,
   nucleation, and concentration equations in Gunawan et al. (2004). The current
   numerical comparison uses declared demonstration initialization because the
   printed seed is negative. Independent moment equations check the coupling.
3. [Lipid nanoparticle formation](nanoparticles.md) uses the mechanism structure
   in Shin et al. (2025). It identifies the inputs needed to reproduce Figure 4,
   exports number and intensity PSDs, and keeps the runnable synthetic example
   distinct from the uncompleted paper comparison.

Each study specifies equations, inputs, term methods, time integration, and code
entry points. Agreement with an analytical solution, consistency with an
independent moment ODE, and agreement with experimental observations are
separate tests.

Shorter worked examples cover [a batch balance](batch.md),
[interacting populations](interacting.md), [collisions](collisions.md), and
[kinetic estimation](estimation.md). They introduce APIs without claiming paper
reproduction.
