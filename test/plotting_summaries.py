"""Independent analytical checks for plot-only probability and work summaries.

Run in the NumPy/Matplotlib plotting environment; no Julia runtime dependency.
"""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'docs/benchmarks'))
from plot_population_studies import pareto_mask
from plot_protein_posterior import probability_levels,joint_density,marginal_density
from unittest.mock import patch
import matplotlib.pyplot as plt
import plot_transport_comparisons
from plot_style import relative_percent

class Summaries(unittest.TestCase):
    def test_relative_percent_and_zero_reference(self):
        # A 0.0038 fractional difference is 0.38%, regardless of physical scale.
        np.testing.assert_allclose(relative_percent([1.0038,1003.8],[1.,1000.]),[.38,.38])
        self.assertTrue(np.isnan(relative_percent([1.],[0.]))[0])
        self.assertEqual(relative_percent([1.],[1.],display_floor=True)[0],1e-14)

    def test_retained_aggregation_cancellation(self):
        path=Path(__file__).resolve().parents[1]/'docs/src/assets/qamar_moments/method_comparison.csv'
        data=np.genfromtxt(path,delimiter=',',names=True,dtype=None,encoding='utf8')
        for method,k,intervals in (('GammaEQMOM',1,((9.5,10.),(39.5,40.))),
                                   ('QMOM',2,((5.,5.5),(28.,28.5)))):
            q=data[(data['scenario']=='aggregation_breakage')&(data['method']==method)&(data['order']==k)]
            for a,b in intervals:
                pair=q[(q['time']==a)|(q['time']==b)]
                self.assertEqual(len(pair),2)
                signed=pair['moment']-pair['reference']
                self.assertLess(signed[0]*signed[1],0.)

    def test_dominated_points(self):
        np.testing.assert_array_equal(pareto_mask([1,2,3,4,2],[4,3,3,1,5]),[True,True,False,True,False])

    def test_gaussian_enclosed_probability(self):
        x=np.linspace(-6,6,601)
        X,Y=np.meshgrid(x,x)
        density=np.exp(-.5*(X*X+Y*Y))/(2*np.pi)
        levels=probability_levels(density)
        # An isotropic Gaussian encloses p at density (1-p)/(2*pi).
        np.testing.assert_allclose(levels,np.array([.05,.32])/(2*np.pi),rtol=.004)

    def test_thin_ridge_and_support(self):
        rng=np.random.default_rng(7)
        x=rng.normal(size=1500)
        y=3*x+rng.normal(scale=1e-3,size=len(x))
        X,Y,density=joint_density(x,y,(-np.inf,np.inf),(-np.inf,np.inf),points=101)
        dx=X[0,1]-X[0,0];dy=Y[1,0]-Y[0,0]
        self.assertAlmostEqual(density.sum()*dx*dy,1.,places=3)
        self.assertLess(np.max(Y[density>density.max()*.5]-3*X[density>density.max()*.5]),.003)
        X,Y,density=joint_density(x,y,(-1,1),(-2,2),points=101)
        self.assertTrue(np.all(density[(abs(X)>1)|(abs(Y)>2)]==0))

    def test_exported_outlet_integrals(self):
        folder=Path(__file__).resolve().parents[1]/'docs/src/assets/continuous_precipitation'
        read=lambda name:np.genfromtxt(folder/name,delimiter=',',names=True,dtype=None,encoding='utf8')
        references=read('reference_outlets.csv')
        performance=read('performance.csv')
        outlets=read('outlets.csv')
        for config in performance:
            q=outlets[np.logical_and.reduce([outlets[name]==config[name]
                for name in ('mesh','size_method','axial_method','nx','nz')])]
            reference=references  # one common uniform-size WENO5 reference for both mesh families
            re=np.r_[reference['x_left'],reference['x_right'][-1]]
            n=int(config['nx'])
            edges=(np.geomspace(re[0],re[-1],n+1) if config['mesh']=='log'
                   else np.linspace(re[0],re[-1],n+1))
            # Independent formulation through interpolated cumulative bin counts:
            # integrate the absolute increments of the CDF difference.
            partition=np.unique(np.r_[edges,re])
            cdf=np.interp(partition,edges,np.r_[0,np.cumsum(q['number_density']*np.diff(edges))])
            refcdf=np.interp(partition,re,np.r_[0,np.cumsum(reference['number_density']*np.diff(re))])
            error=np.abs(np.diff(cdf-refcdf)).sum()/refcdf[-1]
            self.assertAlmostEqual(error,config['normalized_L1'],places=10)
            self.assertAlmostEqual(np.sum(q['volume_density']*np.diff(edges)),1.,places=12)

    def test_compact_comparison_exports(self):
        folder=Path(__file__).resolve().parents[1]/'docs/src/assets/constant_growth'
        errors=np.genfromtxt(folder/'errors.csv',delimiter=',',names=True,dtype=None,encoding='utf8')
        rows=errors[(errors['scheme']=='hoc')&(errors['bins']==200)]
        self.assertEqual(len(rows),2)
        for row in rows:
            data=np.genfromtxt(folder/f"hoc_midpoint_{row['courant']}_200.csv",delimiter=',',names=True)
            edges=np.linspace(0.,1.,201)
            left=np.maximum(edges[:-1],.4)
            right=edges[1:]
            nodes,weights=np.polynomial.legendre.leggauss(24)
            # Independent quadrature of the translated Gaussian, not the CDF
            # expression used by Julia's exact-bin initializer.
            reference=np.array([0. if b<=a else
                (b-a)/2*np.dot(weights,100*np.exp(-.5*((
                    (a+b)/2+(b-a)/2*nodes-.6)/.03)**2))*200
                for a,b in zip(left,right)])
            np.testing.assert_allclose(data['exact'],reference,atol=2e-12)
            self.assertAlmostEqual(np.sqrt(np.mean((data['density']-reference)**2)),row['rmse'],places=10)
            self.assertAlmostEqual(data['density'].min(),row['minimum_density'],places=12)
            self.assertAlmostEqual(-np.minimum(data['density'],0.).sum()/200,
                                   row['negative_number'],places=12)
        smooth=np.genfromtxt(folder/'smooth_errors.csv',delimiter=',',names=True,dtype=None,encoding='utf8')
        hoc=smooth[smooth['scheme']=='hoc']
        self.assertEqual(len(hoc),5)
        measured=np.log(hoc['rmse'][:-1]/hoc['rmse'][1:])/np.log(2)
        np.testing.assert_allclose(measured,2.,atol=.03)  # complete midpoint update at fixed CFL

    def test_tubular_size_resolution_exports(self):
        folder=Path(__file__).resolve().parents[1]/'docs/src/assets/continuous_precipitation'
        summary=np.genfromtxt(folder/'size_resolution.csv',delimiter=',',names=True)
        outlets=np.genfromtxt(folder/'size_resolution_outlets.csv',delimiter=',',names=True)
        np.testing.assert_array_equal(summary['nx'],[144,288,576,1152])
        reference=outlets[outlets['nx']==1152]
        refedges=np.r_[reference['x_left'],reference['x_right'][-1]]
        refnumber=np.sum(reference['number_density']*np.diff(refedges))
        self.assertAlmostEqual(np.diff(refedges).max()*1e6,899/1152,places=10)
        for row in summary:
            q=outlets[outlets['nx']==row['nx']]
            edges=np.r_[q['x_left'],q['x_right'][-1]]
            counts=q['number_density']*np.diff(edges)
            centers=(edges[:-1]+edges[1:])/2
            self.assertAlmostEqual(counts.sum()/row['number_per_m3'],1.,places=12)
            self.assertAlmostEqual((counts*centers**3).sum()/row['M3'],1.,places=12)
            partition=np.unique(np.r_[edges,refedges])
            cdf=np.interp(partition,edges,np.r_[0,np.cumsum(counts)])
            refcdf=np.interp(partition,refedges,np.r_[0,np.cumsum(reference['number_density']*np.diff(refedges))])
            self.assertAlmostEqual(np.abs(np.diff(cdf-refcdf)).sum()/refnumber,
                                   row['normalized_L1_to_finest'],places=11)
        self.assertEqual(summary['normalized_L1_to_finest'][-1],0.)

    def test_compact_visible_in_all_growth_figures(self):
        # Inspect the figures produced by the real plotter, including legends.
        # A CSV alone cannot catch a method accidentally omitted from plotting.
        figures = {}
        def capture(fig, folder, name):
            figures[name] = fig
        with patch.object(plot_transport_comparisons, 'save_figure', capture):
            plot_transport_comparisons.constant_growth()
        self.assertEqual(len(figures), 5)
        for fig in figures.values():
            lines = [line for ax in fig.axes for line in ax.lines]
            compact = [line for line in lines if line.get_label() == 'HOC FV + midpoint']
            self.assertTrue(compact)
            self.assertTrue(all(len(line.get_xdata()) > 1 for line in compact))
            self.assertTrue(all(line.get_linestyle() == '--' for line in compact))
            self.assertFalse(any('Lax' in line.get_label() for line in lines))
            plt.close(fig)

    def test_bounded_marginal(self):
        values=np.linspace(1.001,1.2,1000)
        x,density=marginal_density(values,(1,10),points=2001)
        self.assertGreaterEqual(x.min(),1)
        self.assertAlmostEqual(np.trapz(density,x),1.,places=3)

if __name__=='__main__':unittest.main()
