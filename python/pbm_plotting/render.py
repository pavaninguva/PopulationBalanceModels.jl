"""Numerical-value rendering shared by Julia and repository plotters.

No solver, optimizer, sampler, filesystem data reader or PCE trainer is imported.
Arrays follow output-by-time and field-row-by-column conventions explicitly.
"""
from functools import wraps
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from .style import style
from .density import marginal_on_grid, joint_density, probability_levels
from .coordinates import display, gaussian_marginal, prior_density


def styled(function):
    @wraps(function)
    def call(*args, preset=None, font_size=None, palette=None, **kwargs):
        preset=preset or ('publication' if plt.rcParams['text.usetex'] else 'default')
        effective=plt.rcParams['font.size'] if font_size is None else font_size
        overrides={}
        if palette is not None:
            from cycler import cycler
            overrides['axes.prop_cycle']=cycler(color=list(palette))
        with style(preset, font_size=effective,**overrides):
            existing=kwargs.get('ax',kwargs.get('axes'))
            if existing is not None:
                fig=np.asarray(existing,dtype=object).flat[0].figure
                if font_size is not None:
                    from matplotlib.text import Text
                    for text in fig.findobj(Text):text.set_fontsize(font_size)
                if 'figsize' in kwargs and kwargs['figsize'] is not None:
                    fig.set_size_inches(kwargs['figsize'])
            result=function(*args, **kwargs)
            result[0]._pbm_preset=preset
            if preset=="preview":result[0].set_label("PBM preview (mathtext)")
            return result
    return call


def _axis(ax=None, figsize=(5.,3.2)):
    if ax is not None:return ax.figure,ax
    return plt.subplots(figsize=figsize,layout="constrained")


def _series(values, n):
    values=np.array(values,dtype=float,copy=True)
    if values.ndim==1:values=values.reshape(1,-1)
    if values.ndim!=2 or values.shape[1]!=n:
        raise ValueError("Values must have shape (outputs, coordinates/times)")
    return values


def _finish(ax,xlabel,ylabel,legend,axis):
    if xlabel is not None:ax.set_xlabel(xlabel)
    if ylabel is not None:ax.set_ylabel(ylabel)
    if axis:ax.set(**axis)
    if legend and any(not name.startswith('_') for name in ax.get_legend_handles_labels()[1]):
        ax.legend(**(legend if isinstance(legend,dict) else {}))


@styled
def plot_distribution(values,edges,*,ax=None,labels=None,representation="density",curve="stairs",
                      xlabel=None,ylabel=None,styles=None,legend=True,axis=None,figsize=(5.,3.2),
                      band=None,band_label=None):
    """Bin averages or bin integrals on actual edges, with no normalization.

    Connecting bin centers is an explicit display choice, not reconstruction.
    """
    edges=np.array(edges,dtype=float,copy=True)
    if edges.ndim!=1 or len(edges)<2 or not np.all(np.diff(edges)>0):raise ValueError('Increasing bin edges required')
    y=_series(values,len(edges)-1)
    if representation=='bin_integrals':y=y/np.diff(edges)
    elif representation!='density':raise ValueError('representation must be density or bin_integrals')
    if not np.all(np.isfinite(y)):raise ValueError('Nonfinite distribution values')
    if band is not None:
        if not band_label:raise ValueError('Distribution bands require their explicit meaning in band_label')
        lower,upper=(_series(v,len(edges)-1) for v in band)
        if lower.shape!=y.shape or upper.shape!=y.shape or np.any(lower>upper):raise ValueError('Distribution band shape/order mismatch')
    fig,ax=_axis(ax,figsize)
    for i,row in enumerate(y):
        options=dict((styles or [{}]*len(y))[i]);options.setdefault('label',labels[i] if labels is not None else None)
        if curve=='stairs':ax.stairs(row,edges,**options)
        elif curve=='centers':ax.plot((edges[:-1]+edges[1:])/2,row,**options)
        else:raise ValueError('curve must be stairs or centers')
        if band is not None:
            color=options.get('color',f'C{i}')
            if curve=='centers':ax.fill_between((edges[:-1]+edges[1:])/2,lower[i],upper[i],color=color,alpha=.15,lw=0,label=band_label if i==0 else None)
            else:ax.fill_between(edges,np.r_[lower[i],lower[i,-1]],np.r_[upper[i],upper[i,-1]],step='post',color=color,alpha=.15,lw=0,label=band_label if i==0 else None)
    _finish(ax,xlabel,ylabel,legend,axis)
    return fig,ax


@styled
def plot_observable(times,values=None,*,data=None,sigma=None,covariance=None,band=None,band_label=None,
                    ax=None,labels=None,xlabel=None,ylabel=None,styles=None,data_style=None,
                    legend=True,axis=None,figsize=(5.,3.2)):
    """Prepared trajectories/data; error bars are marginal SD, never joint regions.

    A covariance follows vec(output-by-time), i.e. Fortran order. Bands require
    an explicit label describing their probability/physical meaning.
    """
    x=np.array(times,dtype=float,copy=True)
    if x.ndim!=1 or not len(x) or not np.all(np.isfinite(x)):raise ValueError('Finite abscissa required')
    y=None if values is None else _series(values,len(x))
    observed=None if data is None else _series(data,len(x))
    if y is None and observed is None:raise ValueError('Supply prepared predictions or measured data')
    rows=(y if y is not None else observed).shape[0]
    if observed is not None and observed.shape[0]!=rows:raise ValueError('Data/prediction dimensions differ')
    if sigma is not None and covariance is not None:raise ValueError('Supply sigma or covariance')
    errors=None
    if covariance is not None:
        C=np.asarray(covariance,dtype=float)
        if C.shape!=(rows*len(x),)*2 or not np.all(np.isfinite(C)) or not np.allclose(C,C.T) or np.any(np.diag(C)<0):
            raise ValueError('Covariance must match vec(data), be finite and symmetric with nonnegative diagonal')
        errors=np.sqrt(np.diag(C)).reshape((rows,len(x)),order='F')
    elif sigma is not None:
        errors=np.broadcast_to(np.asarray(sigma,dtype=float),(rows,len(x)))
        if not np.all(np.isfinite(errors)) or np.any(errors<0):raise ValueError('Finite nonnegative standard deviations required')
    if errors is not None and observed is None:raise ValueError('Measurement errors require measured data')
    if band is not None:
        if not band_label:raise ValueError('A band needs an explicit band_label; no probability meaning is inferred')
        lower,upper=(_series(v,len(x)) for v in band)
        if lower.shape!=(rows,len(x)) or upper.shape!=lower.shape or np.any(lower>upper):raise ValueError('Invalid band shape/order')
    fig,ax=_axis(ax,figsize)
    for i in range(rows):
        options=dict((styles or [{}]*rows)[i]);options.setdefault('label',labels[i] if labels is not None else None)
        if y is not None:
            line=ax.plot(x,y[i],**options)[0];color=line.get_color()
        else:color=options.get('color',f'C{i}')
        if band is not None:ax.fill_between(x,lower[i],upper[i],color=color,alpha=.18,lw=0,label=band_label if i==0 else None)
        if observed is not None:
            marker=dict(marker='o',ls='none',ms=3.5,color=color,label='Observations' if i==0 else None)
            marker.update(data_style or {})
            ax.errorbar(x,observed[i],yerr=None if errors is None else errors[i],**marker)
    if covariance is not None:
        ax._pbm_covariance=np.array(covariance,copy=True)
        ax.text(.01,.01,'Marginal SD; correlated errors',transform=ax.transAxes,va='bottom',fontsize='x-small')
    _finish(ax,xlabel,ylabel,legend,axis)
    return fig,ax


@styled
def plot_fit(blocks,*,axes=None,figsize=None,legend=True,**kwargs):
    """One axis per prepared experiment/measurement block; no prediction calls."""
    blocks=list(blocks)
    if not blocks:raise ValueError('Select at least one experiment/measurement block')
    if axes is None:
        fig,axes=plt.subplots(len(blocks),1,squeeze=False,figsize=figsize or (5.,2.7*len(blocks)),layout='constrained')
    else:fig=np.asarray(axes,dtype=object).flat[0].figure
    flat=np.asarray(axes,dtype=object).reshape(-1)
    if len(flat)!=len(blocks):raise ValueError('One supplied axis per selected block required')
    for ax,block in zip(flat,blocks):
        values=dict(block);x=values.pop('times');y=values.pop('values')
        identity=values.pop('identity',None)
        values.update(kwargs)
        plot_observable(x,y,ax=ax,legend=legend,**values)
        ax._pbm_measurement_identity=identity
    return fig,axes


@styled
def plot_field(values,xedges,yedges,*,orientation,ax=None,colorbar=True,colorbar_label=None,
                xlabel=None,ylabel=None,axis=None,figsize=(5.,3.7),**kwargs):
    """Actual rectangle edges. orientation='rows_y' or 'rows_x' is mandatory."""
    x,y=np.asarray(xedges,dtype=float),np.asarray(yedges,dtype=float)
    if np.any(np.diff(x)<=0) or np.any(np.diff(y)<=0):raise ValueError('Increasing field edges required')
    z=np.array(values,dtype=float,copy=True)
    if orientation=='rows_x':z=z.T
    elif orientation!='rows_y':raise ValueError('Declare field orientation rows_y or rows_x')
    if z.shape!=(len(y)-1,len(x)-1) or not np.all(np.isfinite(z)):raise ValueError('Field shape/finiteness does not match actual edges')
    fig,ax=_axis(ax,figsize)
    mesh=ax.pcolormesh(x,y,z,shading='flat',**kwargs)
    if colorbar:fig.colorbar(mesh,ax=ax,label=colorbar_label)
    ax._pbm_mesh=mesh
    _finish(ax,xlabel,ylabel,False,axis)
    return fig,ax


@styled
def plot_uq(times,mean,*,quantiles=None,probabilities=None,sd=None,band=None,band_label=None,
            ax=None,labels=None,diagnostics=None,**kwargs):
    """Prepared named MC/PCE moments; quantiles must be supplied, not Gaussianized."""
    if labels is None:
        count=_series(mean,len(times)).shape[0]
        labels=['Propagated mean'] if count==1 else [f'Output {i+1} mean' for i in range(count)]
    if quantiles is not None:
        probabilities=np.asarray(probabilities,dtype=float)
        q=np.asarray(quantiles,dtype=float)
        if len(probabilities)!=2 or q.shape[-1]!=2:raise ValueError('Select exactly two pointwise quantile endpoints')
        band=(q[...,0],q[...,1]);band_label=band_label or f'Pointwise output quantiles [{probabilities[0]:g}, {probabilities[1]:g}]'
    if sd is not None:
        if band is not None:raise ValueError('Choose quantiles or an explicitly labelled SD band')
        if not band_label:raise ValueError('An SD band needs its declared multiplier/meaning in band_label')
        band=(np.asarray(mean)-np.asarray(sd),np.asarray(mean)+np.asarray(sd))
    result=plot_observable(times,mean,ax=ax,band=band,band_label=band_label,labels=labels,**kwargs)
    result[0]._pbm_uq_diagnostics=diagnostics
    if diagnostics and (diagnostics.get('failed',0) or diagnostics.get('passed') is False):
        warnings.warn('UQ/validation diagnostic screen did not pass; inspect figure._pbm_uq_diagnostics',RuntimeWarning)
        result[1].text(.01,.99,'Inspect UQ failures/validation',transform=result[1].transAxes,va='top',fontsize='small')
    return result


@styled
def plot_parameters(samples=None,*,metadata,gaussian=None,point=None,axes=None,limits=None,
                    labels=None,priors=None,bounds=None,probabilities=(.68,.95),raw_pairs=(),
                    diagnostics=None,figsize=None,legend=True,kde_points=101,max_kde_draws=4000,
                    point_only=False):
    """Joint retained-draw KDE or exact transformed Gaussian contours/marginals.

    Gaussian curves are mapped probability ellipses in their native coordinates,
    not highest-density regions in a nonlinear display coordinate. MCMC levels
    integrate the KDE grid mass. No inference or Gaussian fitting is performed.
    """
    metadata=list(metadata);n=len(metadata)
    if not n:raise ValueError('Select at least one parameter')
    if point_only:
        if point is None:raise ValueError('A point estimate requires its selected center')
        mapped=[np.asarray(priors[i]['x']) if priors is not None else np.array([v-.1,v+.1]) for i,v in enumerate(point)]
    elif gaussian is not None:
        center=np.asarray(gaussian['center'],dtype=float);cov=np.asarray(gaussian['covariance'],dtype=float)
        if center.shape!=(n,) or cov.shape!=(n,n) or not np.all(np.isfinite(cov)) or not np.allclose(cov,cov.T):raise ValueError('Gaussian center/covariance dimension mismatch')
        np.linalg.cholesky(cov) # reject invalid curvature; no jitter or pseudoinverse
        source=gaussian.get('coordinates')
        if source not in ('physical','inference'):raise ValueError('Record Gaussian covariance coordinates')
        mapped=[display(center[i]+np.sqrt(cov[i,i])*np.linspace(-5,5,1001),m,source=source) for i,m in enumerate(metadata)]
    else:
        samples=np.array(samples,dtype=float,copy=True)
        if samples.ndim!=2 or samples.shape[1]!=n or len(samples)<2:raise ValueError('Draws must have shape (draws, parameters) in declared display coordinates')
        mapped=[samples[:,i] for i in range(n)]
    nonfinite=sum(int(np.count_nonzero(~np.isfinite(x))) for x in mapped)
    if nonfinite:
        raise ValueError(f'Nonfinite display transformations: {nonfinite}; no filtering or renormalization is performed')
    bounds=list(bounds or [(-np.inf,np.inf)]*n)
    if limits is None:
        limits=[]
        for values,bound in zip(mapped,bounds):
            q=np.quantile(values,[.0005,.9995]);margin=.15*(q[1]-q[0])
            if not margin>0:margin=.1*max(1,abs(q[0]))
            limits.append((max(q[0]-margin,bound[0]),min(q[1]+margin,bound[1])))
    if axes is None:
        fig,axes=plt.subplots(n,n,squeeze=False,figsize=figsize or (max(4,1.65*n),max(3.5,1.55*n)),layout='constrained')
    else:fig=np.asarray(axes,dtype=object).flat[0].figure
    axes=np.asarray(axes,dtype=object)
    if axes.shape!=(n,n):raise ValueError('Corner axes must have shape parameters × parameters')
    point=None if point is None else np.asarray(point,dtype=float)
    def label(index):
        if labels is not None:return labels[index]
        meta=metadata[index];name=meta.get('name',str(index+1))
        if plt.rcParams['text.usetex']:name=name.replace('_',r'\_').replace('%',r'\%')
        coordinate=meta.get('plotting_coordinate','physical')
        return f'z[{name}]' if coordinate=='inference' else f'ln({name})' if coordinate=='log' else name
    for i in range(n):
        for j in range(n):
            ax=axes[i,j]
            if j>i:ax.set_axis_off();continue
            if i==j:
                x=np.linspace(*limits[i],601)
                if point_only:density=np.zeros_like(x)
                elif gaussian is None:density=marginal_on_grid(samples[:,i],x,bounds[i])
                else:
                    curve=display(center[i]+np.sqrt(cov[i,i])*np.linspace(-8,8,1601),metadata[i],source=source)
                    x=np.unique(np.r_[x,curve[(curve>=limits[i][0])&(curve<=limits[i][1])]])
                    density=gaussian_marginal(x,center[i],cov[i,i],metadata[i],coordinates=source)
                if not point_only:
                    ax.fill_between(x,0,density,color='#0072B2',alpha=.14,lw=0);ax.plot(x,density,color='#0072B2',lw=1.1)
                if priors is not None:
                    prior=priors[i]
                    if callable(prior):ax.plot(x,prior(x),ls='--',color='.35',lw=1)
                    else:ax.plot(prior['x'],prior['density'],ls='--',color='.35',lw=1)
                elif 'family' in metadata[i]:ax.plot(x,prior_density(metadata[i],x),ls='--',color='.35',lw=1)
                if point is not None:ax.axvline(point[i],color='#D55E00',lw=.8,ls=':')
                ax.set(xlim=limits[i],ylim=(0,None));ax.tick_params(labelleft=False)
            else:
                if point_only:pass
                elif gaussian is not None:
                    indices=[j,i];factor=np.linalg.cholesky(cov[np.ix_(indices,indices)])
                    angle=np.linspace(0,2*np.pi,501);circle=np.column_stack([np.cos(angle),np.sin(angle)])
                    for probability,color in zip(sorted(probabilities,reverse=True),['#86b2cc','#0072B2']):
                        ellipse=center[indices]+np.sqrt(-2*np.log1p(-probability))*circle@factor.T
                        ax.plot(display(ellipse[:,0],metadata[j],source=source),display(ellipse[:,1],metadata[i],source=source),color=color,lw=.8)
                elif (j,i) in raw_pairs:
                    selected=np.linspace(0,len(samples)-1,min(2000,len(samples)),dtype=int)
                    ax.scatter(samples[selected,j],samples[selected,i],s=1,color='#0072B2',alpha=.25,linewidths=0,rasterized=True)
                else:
                    selected=np.linspace(0,len(samples)-1,min(max_kde_draws,len(samples)),dtype=int)
                    X,Y,Z=joint_density(samples[selected,j],samples[selected,i],bounds[j],bounds[i],points=kde_points)
                    levels=probability_levels(Z,probabilities)
                    ax.contourf(X,Y,Z,levels=[*levels,Z.max()*1.001],colors=['#e7f0f7','#bad7e8'])
                    ax.contour(X,Y,Z,levels=levels,colors=['#86b2cc','#0072B2'],linewidths=.7)
                if point is not None:ax.plot(point[j],point[i],'x',color='#D55E00',ms=4,mew=.8)
                ax.set(xlim=limits[j],ylim=limits[i])
            if i==n-1:ax.set_xlabel(label(j))
            else:ax.tick_params(labelbottom=False)
            if j==0 and i!=j:ax.set_ylabel(label(i))
            elif i!=j:ax.tick_params(labelleft=False)
            ax.locator_params(axis='x',nbins=3)
            if i!=j:ax.locator_params(axis='y',nbins=3)
    if legend:
        label='Local Gaussian (mapped)' if gaussian is not None else 'Retained posterior KDE/draws'
        handles=[] if point_only else [Line2D([],[],color='#0072B2',label=label)]
        if priors is not None or any('family' in m for m in metadata):handles.append(Line2D([],[],color='.35',ls='--',label='Prior density'))
        if point is not None:
            point_label='Unconverged estimate' if point_only and diagnostics and diagnostics.get('passed') is False else 'MAP center'
            handles.append(Line2D([],[],color='#D55E00',marker='x',ls='none',label=point_label))
        if n>1 and not point_only:
            for probability,color in zip(sorted(probabilities,reverse=True),['#e7f0f7','#bad7e8']):
                percent=r'\%' if plt.rcParams['text.usetex'] else '%'
                meaning='mapped probability region' if gaussian is not None else 'KDE joint region'
                handles.append(Patch(facecolor=color,label=f'{100*probability:g}{percent} {meaning}'))
        if n>1:axes[0,-1].legend(handles=handles,loc='upper right',fontsize='small')
        else:axes[0,0].legend(handles=handles,fontsize='small')
    fig._pbm_parameter_diagnostics=dict(diagnostics or {},nonfinite_display=nonfinite,
        contour_meaning='No uncertainty region (point estimate only)' if point_only else 'Mapped Gaussian probability ellipses' if gaussian is not None else 'KDE highest-density grid regions',
        kde_max_draws=max_kde_draws)
    if diagnostics and diagnostics.get('passed') is False:
        warnings.warn('Optimization did not converge; inspect figure._pbm_parameter_diagnostics' if point_only else
            'Posterior diagnostic screen did not pass; see figure._pbm_parameter_diagnostics',RuntimeWarning)
        fig.text(.5,.995,'Optimization did not converge' if point_only else 'Posterior diagnostic screen not passed',ha='center',va='top')
    return fig,axes
