# Diagonal Ito diffusion: ∂xx(Dx*n) + ∂yy(Dy*n), with bin-integrated states.
# Geometry is reusable; coefficients and densities belong to the current stage.
default_method(::SizeDiffusion,::Grid2D)=FiniteVolumeDiffusion()

function _tensor_diffusivities(term::SizeDiffusion,xy,p,t,ctx)
    law=term.diffusivity
    value=law isa Union{Real,Tuple,AbstractArray} ? law : law(xy,p,t,ctx)
    pair=value isa Real ? (value,value) : value
    pair isa Union{Tuple,AbstractVector} && length(pair)==2 || throw(ArgumentError(
        "Grid2D SizeDiffusion requires a scalar or (Dx,Dy) diagonal diffusivities; use D((x,y),p,t,ctx); cross diffusion is unsupported"))
    for D in pair
        D isa Real && isfinite(D) && _primal(D)>=0 || throw(DomainError(
            D,"SizeDiffusion at coordinates $xy, time $t requires finite nonnegative diagonal diffusivities"))
    end
    (pair[1],pair[2])
end

function prepare_term(term::SizeDiffusion,::FiniteVolumeDiffusion,g::Grid2D;
                      bc=SizeBC2D(),kwargs...)
    _validate_size_diffusion(PBMModel(term),g,bc,BatchDomain(),nothing)
    term.diffusivity isa Union{Real,Tuple,AbstractArray} &&
        _tensor_diffusivities(term,(first(g.x.centers),first(g.y.centers)),nothing,0,nothing)
    (x=_diffusion_geometry(g.x,bc.x),y=_diffusion_geometry(g.y,bc.y))
end

function add_term!(dN,N,term::SizeDiffusion,::FiniteVolumeDiffusion,
                   geometry,op::CompiledPBM2D,p,t,ctx)
    g=op.grid
    weighted_x=similar(dN);weighted_y=similar(dN)
    for j in axes(N,2),i in axes(N,1)
        Dx,Dy=_tensor_diffusivities(term,(g.x.centers[i],g.y.centers[j]),p,t,ctx)
        n=N[i,j]/(g.x.widths[i]*g.y.widths[j])
        weighted_x[i,j]=Dx*n
        weighted_y[i,j]=Dy*n
    end
    for j in axes(N,2)
        for i in 1:size(N,1)-1
            flux=-(weighted_x[i+1,j]-weighted_x[i,j])*geometry.x.inverse_distances[i]*g.y.widths[j]
            dN[i,j]-=flux
            dN[i+1,j]+=flux
        end
        if op.bc.x.left isa Periodic
            flux=-(weighted_x[1,j]-weighted_x[end,j])*geometry.x.wrap_inverse_distance*g.y.widths[j]
            dN[end,j]-=flux
            dN[1,j]+=flux
        end
    end
    for i in axes(N,1)
        for j in 1:size(N,2)-1
            flux=-(weighted_y[i,j+1]-weighted_y[i,j])*geometry.y.inverse_distances[j]*g.x.widths[i]
            dN[i,j]-=flux
            dN[i,j+1]+=flux
        end
        if op.bc.y.left isa Periodic
            flux=-(weighted_y[i,1]-weighted_y[i,end])*geometry.y.wrap_inverse_distance*g.x.widths[i]
            dN[i,end]-=flux
            dN[i,1]+=flux
        end
    end
    nothing
end
