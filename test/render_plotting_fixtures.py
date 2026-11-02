"""Plot-only visual checks. Required --output prevents overwriting paper/docs assets.

MPLBACKEND=Agg python test/render_plotting_fixtures.py --output /tmp/pbm-figures
Uses the retained protein/growth/milling values and a labelled small field fixture.
No numerical producer, fit, sampler, UQ propagation or table exporter is run.
"""
import argparse
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'paper'),str(ROOT/'paper/_support')]
from pbm_plotting import plot_field,plot_uq,plot_observable,save_figure,style
from _support import figures


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main(output):
    output.mkdir(parents=True,exist_ok=True)
    figures.RETAINED=True;figures.TABLES=False
    def capture(fig,case,name,info):
        save_figure(fig,output/(name+'.png'),vector_first=True)
        plt.close(fig)
    with style('publication'):
        growth=load('growth_visual_check',ROOT/'paper/growth/plot.py')
        with patch.object(growth,'export',side_effect=capture):growth.growth()
        milling=load('milling_visual_check',ROOT/'paper/milling/plot.py')
        with patch.object(milling,'export',side_effect=capture):milling.milling()
        import protein_plot
        with patch.object(protein_plot,'OUT',output/'protein'),patch.object(protein_plot,'DOCS',output/'protein/docs'):
            protein_plot.plot(figures_only=True)
            # A second MAP-only rendering is explicitly separate and has no
            # latent patches/corners, despite the full producer being available.
            with patch.object(protein_plot,'OUT',output/'protein_map_only'),patch.object(protein_plot,'DOCS',output/'protein_map_only/docs'):
                protein_plot.plot(figures_only=True,map_only=True)
        for width,font in [(3.35,8.),(7.,11.)]:
            x=np.array([0.,.1,.4,.7,1.]);y=np.geomspace(1.,100.,16)
            X,Y=np.meshgrid((x[1:]+x[:-1])/2,(y[1:]+y[:-1])/2)
            field=np.exp(-((X-.55)/.22)**2-((np.log(Y)-np.log(15))/.65)**2)
            # Single-column composition stacks panels; fonts are never shrunk
            # to force a two-column legend/colorbar into a narrow figure.
            fig,axes=plt.subplots(2,1,figsize=(width,width*1.65),layout='constrained') if width<4 else \
                plt.subplots(1,2,figsize=(width,width*.58),layout='constrained')
            plot_field(field,x,y,orientation='rows_y',ax=axes[0],xlabel=r'$z$ (m)',ylabel=r'$L$ ($\mu$m)',
                colorbar_label='Fixture density',font_size=font,cmap='magma')
            t=np.linspace(0,1,41);mean=1+np.sin(t)
            q=np.stack([mean-.2,mean+.3],axis=-1)[None,:,:]
            plot_uq(t,mean,quantiles=q,probabilities=[.025,.975],ax=axes[1],font_size=font,
                xlabel=r'$t$ (s)',ylabel='Fixture output',band_label='Pointwise output quantiles\n[0.025, 0.975]',
                legend=dict(loc='upper left',fontsize=font-1))
            save_figure(fig,output/f'field_uq_{width:g}.png',vector_first=True)
            plt.close(fig)
    print(output)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    main(parser.parse_args().output)
