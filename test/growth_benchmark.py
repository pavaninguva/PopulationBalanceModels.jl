"""Gaussian-only case-study selection, independent norms and plot wiring.

Uses the bundled small figure selection; no archived production folders or
Julia solves are required by these checks.
"""
import copy
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'paper'))
sys.path.insert(0,str(ROOT/'docs/benchmarks'))
from _support.growth_results import (METHODS,load_results,refinement,profile,
                                     observed_orders,validate,write_selection)
import plot_transport_comparisons


class GaussianBenchmark(unittest.TestCase):
    def setUp(self):
        self.data=load_results()

    def tearDown(self):
        plt.close('all')

    def test_single_physical_problem_and_complete_pairings(self):
        data=self.data
        self.assertEqual(set(data['metrics']['profile']),{'gaussian'})
        self.assertEqual(data['configuration']['bins'],[40,80,160,320,640])
        self.assertEqual(data['configuration']['profile_bins'],200)
        self.assertEqual(len(data['metrics']),48)
        self.assertEqual(len(data['profiles']),2600)
        for method in METHODS:
            rows=refinement(data,method)
            np.testing.assert_array_equal(rows['bins'],[40,80,160,320,640])
            np.testing.assert_array_equal(rows['tf'],np.full(5,4.0))
            expected=1. if method=='inguva2022' else .5
            np.testing.assert_array_equal(rows['courant'],np.full(5,expected))
        changed=copy.deepcopy(data)
        changed['metrics']['profile'][0]='smooth'
        with self.assertRaisesRegex(ValueError,'Gaussian runs only'):
            validate(changed)

    def test_profiles_and_norms_against_independent_gaussian_quadrature(self):
        nodes,weights=np.polynomial.legendre.leggauss(32)
        for method in METHODS:
            for c in (.5,1.):
                if method=='inguva2022' and c!=1.:continue
                curve=profile(self.data,method,c)
                widths=curve['right']-curve['left']
                exact=[]
                for a,b in zip(curve['left'],curve['right']):
                    lo,hi=max(a,.4),min(b,1.4)
                    exact.append(0. if hi<=lo else (hi-lo)/2*np.dot(weights,
                        100*np.exp(-.5*(((lo+hi)/2+(hi-lo)/2*nodes-.6)/.03)**2)))
                exact=np.asarray(exact)
                np.testing.assert_allclose(curve['exact_number'],exact,atol=2e-14,rtol=2e-13)
                np.testing.assert_allclose(curve['number']/widths,curve['density'],rtol=2e-14)
                metrics=self.data['metrics']
                row=metrics[(metrics['method']==method)&(metrics['bins']==200)&(metrics['courant']==c)][0]
                error=curve['number']/widths-exact/widths
                self.assertAlmostEqual(math.sqrt(float(np.dot(widths,error**2)/widths.sum())),row['rmse'],delta=2e-12)
                self.assertAlmostEqual(float(np.dot(widths,np.abs(error))),row['l1'],delta=2e-12)

    def test_orders_have_known_scaling_and_roundoff_semantics(self):
        data=copy.deepcopy(self.data)
        for method in METHODS:
            mask=data['metrics']['method']==method
            n=data['metrics']['bins'][mask]
            data['metrics']['rmse'][mask]=3./n**2
            data['metrics']['l1'][mask]=5./n**3
        orders=observed_orders(data)
        for method in METHODS:
            rows=[r for r in orders if r['method']==method]
            self.assertTrue(np.isnan(rows[0]['observed_rmse_order']))
            np.testing.assert_allclose([r['observed_rmse_order'] for r in rows[1:]],2.,atol=1e-13)
            np.testing.assert_allclose([r['observed_l1_order'] for r in rows[1:]],3.,atol=1e-13)
        scaled=copy.deepcopy(data)
        scaled['benchmark']['height']*=1e6
        scaled['metrics']['rmse']*=1e6
        scaled['metrics']['l1']*=1e6
        other=observed_orders(scaled)
        np.testing.assert_allclose([r['observed_rmse_order'] for r in orders],
                                   [r['observed_rmse_order'] for r in other],equal_nan=True)
        actual=[r for r in observed_orders(self.data) if r['method']=='inguva2022']
        self.assertTrue(all(np.isnan(r['observed_rmse_order']) and np.isnan(r['observed_l1_order']) for r in actual))

    def test_selection_roundtrip_and_corruption_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            write_selection(self.data,folder)
            repeated=load_results(folder)
            np.testing.assert_array_equal(repeated['metrics'],self.data['metrics'])
            np.testing.assert_array_equal(repeated['profiles'],self.data['profiles'])
            with (folder/'gaussian_metrics.csv').open('a') as stream:stream.write('\n')
            with self.assertRaisesRegex(ValueError,'Changed Gaussian figure input'):
                load_results(folder)

    def test_both_plotters_use_the_gaussian_metrics_for_work_precision(self):
        spec=importlib.util.spec_from_file_location('paper_gaussian_plot',ROOT/'paper/growth/plot.py')
        paper=importlib.util.module_from_spec(spec);spec.loader.exec_module(paper)
        paper_figures={};manual_figures={}
        with patch.object(paper,'export',side_effect=lambda fig,number,name,info:paper_figures.setdefault(name,fig)):
            paper.growth()
        with patch.object(plot_transport_comparisons,'save_figure',side_effect=lambda fig,folder,name:manual_figures.setdefault(name,fig)):
            plot_transport_comparisons.constant_growth()
        self.assertEqual(set(paper_figures),{'growth_profiles','growth_work_precision'})
        self.assertEqual(set(manual_figures),{'solutions','performance','observed_orders','timings'})
        for figures,name in ((paper_figures,'growth_work_precision'),(manual_figures,'timings')):
            axes=figures[name].axes
            for i,method in enumerate(METHODS):
                rows=refinement(self.data,method)
                np.testing.assert_array_equal(axes[0].lines[i].get_xdata(),rows['bins'])
                np.testing.assert_array_equal(axes[0].lines[i].get_ydata(),rows['seconds'])
                np.testing.assert_array_equal(axes[1].lines[i].get_xdata(),rows['seconds'])
                np.testing.assert_array_equal(axes[1].lines[i].get_ydata(),rows['rmse'])


if __name__=='__main__':unittest.main(verbosity=2)
