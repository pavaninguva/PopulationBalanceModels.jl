"""Explicit coordinate maps for display; these never alter an inference target."""
import math
import numpy as np


def physical(z, meta):
    z=np.asarray(z,dtype=float);kind=meta.get('transform','identity');scale=meta.get('scale',1.)
    with np.errstate(over='ignore',invalid='ignore'):
        if kind=='identity':return meta.get('location',0.)+scale*z
        if kind=='log':return np.exp(np.log(scale)+z)
        if kind=='shifted_log':return meta['bound']+(-1 if meta.get('upper_bound',False) else 1)*np.exp(np.log(scale)+z)
        if kind=='logit':
            s=np.empty_like(z);positive=z>=0;s[positive]=1/(1+np.exp(-z[positive]));e=np.exp(z[~positive]);s[~positive]=e/(1+e)
            return meta['lower']+(meta['upper']-meta['lower'])*s
    raise ValueError(f"Unsupported transform {kind}; provide explicitly transformed plotting values")


def inverse(theta,meta):
    """Inference coordinate and absolute dz/dtheta, with invalid support exposed."""
    theta=np.asarray(theta,dtype=float);kind=meta.get('transform','identity');scale=meta.get('scale',1.)
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        if kind=='identity':return (theta-meta.get('location',0.))/scale,np.full_like(theta,1/scale)
        if kind=='log':
            valid=theta>0
            return np.where(valid,np.log(theta)-np.log(scale),np.nan),np.where(valid,1/theta,np.nan)
        if kind=='shifted_log':
            delta=(meta['bound']-theta) if meta.get('upper_bound',False) else (theta-meta['bound'])
            return np.where(delta>0,np.log(delta)-np.log(scale),np.nan),np.where(delta>0,1/delta,np.nan)
        if kind=='logit':
            a,b=meta['lower'],meta['upper'];valid=(theta>a)&(theta<b)
            return np.where(valid,np.log(theta-a)-np.log(b-theta),np.nan),np.where(valid,(b-a)/((theta-a)*(b-theta)),np.nan)
    raise ValueError(f"Unsupported transform {kind}")


def display(values,meta,*,source='inference',coordinate=None):
    coordinate=coordinate or meta.get('plotting_coordinate','physical')
    values=np.asarray(values,dtype=float)
    if coordinate=='inference':return values.copy() if source=='inference' else inverse(values,meta)[0]
    # Preserve the exact linear map when log of a log-transformed parameter is
    # requested; avoid an unnecessary exp/log round trip and its overflow.
    if coordinate=='log' and source=='inference' and meta.get('transform')=='log':
        return values+np.log(meta.get('scale',1.))
    theta=physical(values,meta) if source=='inference' else values.copy()
    if coordinate=='physical':return theta
    if coordinate=='log':
        with np.errstate(divide='ignore',invalid='ignore'):return np.log(theta)
    raise ValueError('Display coordinates must be physical, inference or log')


def gaussian_marginal(x,mean,variance,meta,*,coordinates='inference'):
    """Exact marginal pushforward, including the display-coordinate Jacobian."""
    if variance<=0:raise ValueError('Gaussian variance must be positive')
    x=np.asarray(x,dtype=float);coordinate=meta.get('plotting_coordinate','physical')
    if coordinate=='inference':
        if coordinates=='inference':v,jac=x,np.ones_like(x)
        else:
            v=physical(x,meta);_,dz=inverse(v,meta);jac=1/dz
    else:
        theta=np.exp(x) if coordinate=='log' else x
        jac=theta if coordinate=='log' else np.ones_like(x)
        if coordinates=='inference':
            v,dz=inverse(theta,meta);jac=jac*dz
        else:v=theta
    with np.errstate(over='ignore',invalid='ignore'):
        density=np.exp(-.5*(v-mean)**2/variance)/np.sqrt(2*np.pi*variance)*np.abs(jac)
    return np.where(np.isfinite(density),density,0.)


def prior_density(meta,x):
    """Analytical legacy-export priors, normalized in the declared display coordinate.

    Julia's general Distributions.jl priors arrive as density grids instead.
    Unknown exported families require an explicit callable/grid, never a guess.
    """
    x=np.asarray(x,dtype=float);coordinate=meta.get('plotting_coordinate','physical')
    if coordinate=='inference':
        if meta['density_coordinates']=='inference':value,jac=x,np.ones_like(x)
        else:
            value=physical(x,meta);jac=1/inverse(value,meta)[1]
    else:
        value=np.exp(x) if coordinate=='log' else x
        jac=value.copy() if coordinate=='log' else np.ones_like(x)
        if meta['density_coordinates']=='inference':
            value,dz=inverse(value,meta);jac=jac*dz
    args=meta['arguments'];kind=meta['family'];density=np.zeros_like(value)
    if kind=='uniform':
        a,b=args;density=np.where((value>=a)&(value<=b),1/(b-a),0.)
    elif kind=='normal':
        mu,sd=args;density=np.exp(-.5*((value-mu)/sd)**2)/(sd*np.sqrt(2*np.pi))
    elif kind=='lognormal':
        mu,sd=args;valid=value>0;v=value[valid]
        density[valid]=np.exp(-.5*((np.log(v)-mu)/sd)**2)/(v*sd*np.sqrt(2*np.pi))
    elif kind=='exponential':
        scale=args[0];valid=value>=0;density[valid]=np.exp(-value[valid]/scale)/scale
    else:raise ValueError(f'Unsupported exported prior family {kind}; supply evaluated prior density')
    with np.errstate(invalid='ignore'):result=density*np.abs(jac)
    return np.where(np.isfinite(result),result,0.)
