# Parameter-specific DLVO preparation, shared numerical formulas in lnp_material.jl.
# The closures below are for this explicitly fixed mixture and grid; rebuild
# them if either changes. No simulation or long-time driver is imported.
function prepare_attachment(grid;water_fraction,temperature,relative_permittivity)
    n = length(grid)
    table = Matrix{Float64}(undef,n,n)
    quadrature = dlvo_quadrature(128)
    for j in 1:n,i in 1:j
        value = dlvo_attachment(grid.centers[i]/2,grid.centers[j]/2;
            temperature,ionic_strength=100water_fraction,relative_permittivity,quadrature)
        table[i,j] = table[j,i] = value
    end
    indices = Dict(x=>i for (i,x) in enumerate(grid.centers))
    (L,M,w,T)->table[indices[L],indices[M]]
end
