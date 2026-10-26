"""Display-only posterior density summaries on bounded, sheared grids.

These summarize saved draws; they do not change the statistical target or
replace posterior inference.
"""
import numpy as np

def marginal_density(values, support, points=400):
    """Gaussian KDE with reflection at finite support endpoints (display only)."""
    values = np.asarray(values)
    h = 1.06*np.std(values,ddof=1)*len(values)**(-.2)
    lower, upper = support
    x = np.linspace(max(lower,values.min()-4*h),min(upper,values.max()+4*h),points)
    density = np.zeros_like(x)
    for centers in (values, 2*lower-values if np.isfinite(lower) else [],
                    2*upper-values if np.isfinite(upper) else []):
        if len(centers):
            density += np.exp(-.5*((x[:,None]-centers)/h)**2).sum(axis=1)
    density /= len(values)*h*np.sqrt(2*np.pi)
    return x,density

def probability_levels(density, probabilities=(.95,.68)):
    """Highest-density thresholds by integrated grid mass, not peak fractions.

    The affine grid used below has constant cell area, which cancels here.
    """
    ranked = np.sort(density[density>0])[::-1]
    mass = np.cumsum(ranked)/ranked.sum()
    return np.sort([ranked[min(np.searchsorted(mass,p),len(ranked)-1)] for p in probabilities])

def joint_density(x,y,xsupport,ysupport,points=181):
    """KDE on a sheared/standardized grid resolving a thin correlation ridge.

    The shear has determinant one; scaling has a constant Jacobian. Masking
    outside declared support leaves contours inside the parameter domain.
    This is a summary of the saved draws, not an inference approximation.
    """
    slope = np.cov(x,y)[0,1]/np.var(x,ddof=1)
    residual = y-y.mean()-slope*(x-x.mean())
    scales = np.array([x.std(ddof=1),residual.std(ddof=1)])
    centers = np.column_stack(((x-x.mean())/scales[0],residual/scales[1]))
    h = len(x)**(-1/6)
    u = np.linspace(centers[:,0].min()-4*h,centers[:,0].max()+4*h,points)
    v = np.linspace(centers[:,1].min()-4*h,centers[:,1].max()+4*h,points)
    U,V = np.meshgrid(u,v)
    X = x.mean()+scales[0]*U
    Y = y.mean()+slope*(X-x.mean())+scales[1]*V
    density = np.zeros(U.size)
    query = np.column_stack((U.ravel(),V.ravel()))
    for start in range(0,len(query),256):
        delta = (query[start:start+256,None,:]-centers[None,:,:])/h
        density[start:start+256] = np.exp(-.5*np.sum(delta*delta,axis=2)).sum(axis=1)
    density = density.reshape(U.shape)/(len(x)*2*np.pi*h*h*np.prod(scales))
    inside = (X>=xsupport[0])&(X<=xsupport[1])&(Y>=ysupport[0])&(Y<=ysupport[1])
    density[~inside] = 0.
    return X,Y,density

