"""Scientific and rendering contracts; only fixtures/retained values, no Julia solves."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from pbm_plotting import (plot_distribution,plot_observable,plot_fit,plot_parameters,
    plot_uq,plot_field,save_figure,style)
from pbm_plotting.coordinates import physical,inverse,display,gaussian_marginal,prior_density
from pbm_plotting.density import marginal_on_grid,joint_density,probability_levels


class SharedPlots(unittest.TestCase):
    def tearDown(self):plt.close('all')

    def test_nonuniform_distribution_and_field_geometry(self):
        edges=np.array([0.,1.,3.,6.]);counts=np.array([2.,8.,9.]);before=counts.copy()
        fig,ax=plot_distribution(counts,edges,representation='bin_integrals')
        np.testing.assert_array_equal(ax.patches[0].get_data().values,[2,4,3])
        np.testing.assert_array_equal(counts,before)
        values=np.array([[1.,2.,3.],[4.,5.,6.]])
        fig,ax=plot_field(values,[0,1,4],[0,2,3,6],orientation='rows_x')
        np.testing.assert_array_equal(ax._pbm_mesh.get_array(),values.T)
        coords=ax._pbm_mesh.get_coordinates()
        np.testing.assert_array_equal(coords[0,:,0],[0,1,4])
        np.testing.assert_array_equal(coords[:,0,1],[0,2,3,6])
        self.assertEqual(len(fig.axes),2)
        with self.assertRaisesRegex(ValueError,'orientation'):
            plot_field(values,[0,1,4],[0,2,3,6],orientation='guess')

    def test_covariance_and_optional_fit_uncertainty(self):
        C=np.array([[1.,.6],[.6,4.]])
        blocks=[dict(times=[0,1],values=[[1,2]],data=[[1.2,1.8]],covariance=C,identity='experiment/concentration')]
        before=copy.deepcopy(blocks)
        fig,axes=plot_fit(blocks)
        self.assertEqual(axes[0,0]._pbm_measurement_identity,'experiment/concentration')
        np.testing.assert_array_equal(axes[0,0]._pbm_covariance,C)
        self.assertTrue(any('correlated' in t.get_text() for t in axes[0,0].texts))
        np.testing.assert_array_equal(blocks[0]['covariance'],before[0]['covariance'])
        self.assertEqual(len(axes[0,0].collections),1) # errorbar, no invented posterior band
        with self.assertRaisesRegex(ValueError,'band_label'):
            plot_observable([0,1],[1,2],band=([0,1],[2,3]))

    def test_coordinate_jacobians_and_exact_gaussian_pushforward(self):
        m=dict(transform='log',scale=2.,plotting_coordinate='physical')
        z=np.array([-3.,0.,2.]);theta=2*np.exp(z)
        np.testing.assert_allclose(physical(z,m),theta)
        np.testing.assert_allclose(inverse(theta,m)[0],z)
        # Known lognormal marginal: Gaussian in z has mapped median .2 and
        # mean .2*exp(.5^2/2), rather than the mapped MAP .2.
        mu=np.log(.1);sd=.5;x=np.exp(np.linspace(mu-8*sd,mu+8*sd,20001))*2
        d=gaussian_marginal(x,mu,sd**2,m)
        self.assertAlmostEqual(np.trapz(d,x),1.,places=7)
        self.assertAlmostEqual(np.trapz(x*d,x),.2*np.exp(.125),places=7)
        logmeta=dict(m,plotting_coordinate='log')
        np.testing.assert_array_equal(display(z,logmeta),z+np.log(2.))
        expected=np.exp(-.5*((np.log(x/.2))/.5)**2)/(x*.5*np.sqrt(2*np.pi))
        np.testing.assert_allclose(d,expected,rtol=2e-14)
        shifted=dict(transform='shifted_log',bound=5.,upper_bound=True,scale=3.)
        np.testing.assert_allclose(inverse(physical(z,shifted),shifted)[0],z)
        logit=dict(transform='logit',lower=1.,upper=5.,plotting_coordinate='physical')
        self.assertEqual(float(physical(np.array(0.),logit)),3.)
        zz=np.linspace(-10,10,50001);xx=physical(zz,logit)
        self.assertAlmostEqual(np.trapz(gaussian_marginal(xx,0.,1.,logit),xx),1.,places=7)
        prior=dict(m,family='uniform',density_coordinates='inference',arguments=[-2.,2.])
        p=prior_density(prior,theta)
        np.testing.assert_allclose(p,[0,1/(4*2),1/(4*theta[-1])])

    def test_gaussian_and_draw_contours_have_distinct_meaning(self):
        metas=[dict(name=k,transform='identity',plotting_coordinate='inference') for k in ['a','b']]
        mean=[0.,0.];cov=np.array([[1.,.8],[.8,2.]])
        fig,axes=plot_parameters(metadata=metas,gaussian=dict(center=mean,covariance=cov,coordinates='inference'),legend=False)
        self.assertEqual(axes[1,0].get_xlabel(),'z[a]')
        for line,p in zip(axes[1,0].lines,[.95,.68]):
            xy=np.column_stack([line.get_xdata(),line.get_ydata()])
            radius=np.einsum('ni,ij,nj->n',xy,np.linalg.inv(cov),xy)
            np.testing.assert_allclose(radius,-2*np.log(1-p),atol=1e-13)
        with self.assertRaises(np.linalg.LinAlgError):
            plot_parameters(metadata=metas,gaussian=dict(center=mean,covariance=[[1,2],[2,1]],coordinates='inference'))
        rng=np.random.default_rng(52);draws=rng.multivariate_normal(mean,cov,size=1200);saved=draws.copy()
        with self.assertWarns(RuntimeWarning):
            fig,_=plot_parameters(draws,metadata=metas,diagnostics=dict(passed=False),kde_points=51)
        self.assertFalse(fig._pbm_parameter_diagnostics['passed'])
        np.testing.assert_array_equal(draws,saved)
        x=np.linspace(-6,6,501);X,Y=np.meshgrid(x,x);Z=np.exp(-.5*(X*X+Y*Y))/(2*np.pi)
        np.testing.assert_allclose(probability_levels(Z),np.array([.05,.32])/(2*np.pi),rtol=.005)
        with self.assertRaisesRegex(ValueError,'no filtering'):
            plot_parameters(np.array([[1,np.nan],[2,3]]),metadata=metas)

    def test_uq_output_bands_are_not_mean_errors(self):
        times=[0,1];mean=np.array([[2.,3.]]);q=np.array([[[1,4],[2,5]]])
        fig,ax=plot_uq(times,mean,quantiles=q,probabilities=[.025,.975])
        self.assertIn('output quantiles',ax.get_legend_handles_labels()[1][1])
        np.testing.assert_array_equal(mean,[[2,3]])
        with self.assertRaisesRegex(ValueError,'explicitly labelled'):
            plot_uq(times,mean,sd=mean,quantiles=q,probabilities=[.025,.975])
        with self.assertRaisesRegex(ValueError,'band_label'):
            plot_uq(times,mean,sd=mean)
        with self.assertWarns(RuntimeWarning):
            fig,ax=plot_uq(times,mean,diagnostics=dict(passed=False,failed=0))
        self.assertTrue(any('validation' in t.get_text() for t in ax.texts))

    def test_scoped_style_and_physical_export_size(self):
        original=plt.rcParams.copy()
        with tempfile.TemporaryDirectory() as folder:
            fig,ax=plot_observable([0,1],[0,1],figsize=(3.35,2.8),xlabel='Time',ylabel='Number',font_size=10)
            size=fig.get_size_inches().copy();font=ax.xaxis.label.get_fontsize()
            for dpi in [150,300]:
                path=Path(folder)/f'figure{dpi}.png';save_figure(fig,path,dpi=dpi)
                with Image.open(path) as image:
                    self.assertEqual(image.size,tuple((size*dpi).astype(int)))
                    np.testing.assert_allclose(image.info['dpi'],[dpi,dpi],atol=.02)
                self.assertEqual(ax.xaxis.label.get_fontsize(),font)
            save_figure(fig,Path(folder)/'wider.png',width=7,height=3.2,font_size=11,layout='constrained')
            with Image.open(Path(folder)/'wider.png') as image:self.assertEqual(image.size,(2100,960))
            self.assertEqual(len(list(Path(folder).iterdir())),3)
        self.assertEqual(dict(plt.rcParams),dict(original))
        with patch('pbm_plotting.style.shutil.which',return_value=None):
            with self.assertRaisesRegex(RuntimeError,'latex and dvipng'):
                plot_observable([0,1],[0,1],preset='publication')

    def test_retained_protein_plotting_does_not_require_current_source(self):
        sys.path.insert(0,str(ROOT/'paper/_support'))
        import protein_plot,protein_provenance
        folder=ROOT/'paper/protein/data/cnt'
        with patch.object(protein_provenance,'numerical_sources',side_effect=AssertionError('no numerical reuse check during plotting')):
            manifest=protein_plot.verify_model(folder)
        self.assertTrue(manifest['complete'])
        # A genuine MAP-only subset remains plottable without any posterior or
        # uncertainty files. Corruption of a requested input still rejects it.
        with tempfile.TemporaryDirectory() as temporary:
            target=Path(temporary)/'cnt';target.mkdir()
            names=['map.json','trajectory.csv','distribution_curves.csv']
            for name in names:shutil.copyfile(folder/name,target/name)
            # The production driver's complete flag means POSTERIOR complete;
            # it deliberately remains false after a completed MAP-only run.
            subset=dict(manifest,complete=False,files={k:manifest['files'][k] for k in names})
            (target/'manifest.json').write_text(json.dumps(subset))
            self.assertFalse(protein_plot.verify_model(target,posterior=False,bands=False)['complete'])
            with (target/'map.json').open('a') as stream:stream.write(' ')
            with self.assertRaisesRegex(ValueError,'Changed'):
                protein_plot.verify_model(target,posterior=False,bands=False)

if __name__=='__main__':unittest.main(verbosity=2)
