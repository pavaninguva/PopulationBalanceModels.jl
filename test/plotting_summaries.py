"""Independent analytical checks for plot-only probability and work summaries.

Run in the NumPy/Matplotlib plotting environment; no Julia runtime dependency.
"""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'docs/benchmarks'))
from plot_new_case_studies import pareto_mask
from plot_protein_posterior import probability_levels,joint_density,marginal_density

class Summaries(unittest.TestCase):
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
            left=np.maximum(edges[:-1]-.4,0.)
            right=np.maximum(edges[1:]-.4,0.)
            reference=(np.exp(-left/.01)-np.exp(-right/.01))*200
            np.testing.assert_allclose(data['exact'],reference,atol=2e-12)
            self.assertAlmostEqual(np.sqrt(np.mean((data['density']-reference)**2)),row['rmse'],places=10)
            self.assertLess(row['minimum_density'],0.)  # signed lobes were not clipped
        smooth=np.genfromtxt(folder/'smooth_errors.csv',delimiter=',',names=True,dtype=None,encoding='utf8')
        hoc=smooth[smooth['scheme']=='hoc']
        self.assertEqual(len(hoc),5)
        measured=np.log(hoc['rmse'][:-1]/hoc['rmse'][1:])/np.log(2)
        np.testing.assert_allclose(measured,2.,atol=.03)  # complete midpoint update at fixed CFL

    def test_bounded_marginal(self):
        values=np.linspace(1.001,1.2,1000)
        x,density=marginal_density(values,(1,10),points=2001)
        self.assertGreaterEqual(x.min(),1)
        self.assertAlmostEqual(np.trapz(density,x),1.,places=3)

if __name__=='__main__':unittest.main()
