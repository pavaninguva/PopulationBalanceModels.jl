"""Focused statistical transformations and read-only publication contracts."""
import copy
import importlib.util
import json
import tempfile
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'paper/_support'))
import protein_provenance as provenance
from protein_plot import BASE, prior_density
from protein_uncertainty import physical, display, gaussian_marginal, gaussian_draws


class PublicationContracts(unittest.TestCase):
    def test_png_export_keeps_no_vector_derivatives(self):
        from PIL import Image
        import matplotlib.pyplot as plt
        from plotting_style import save_png
        with tempfile.TemporaryDirectory() as temporary:
            fig,ax=plt.subplots(figsize=(2,1))
            ax.plot([0,1],[0,1])
            path=Path(temporary)/'figure.png'
            save_png(fig,path)
            plt.close(fig)
            self.assertEqual([p.name for p in Path(temporary).iterdir()],['figure.png'])
            with Image.open(path) as image:
                self.assertEqual(image.size,(600,300))
                np.testing.assert_allclose(image.info['dpi'],[300,300],atol=.01)

    def test_reviewed_reuse_preserves_original_analysis_identity(self):
        fields = dict(passed=True, model='cnt', original_target_hash='target',
                      inputs={'map.json': 'map'}, settings={'bins': 100},
                      parameter_names=['logAJ'], likelihood='same density',
                      projects={'Project.toml': 'project'}, dependencies={'ForwardDiff': '1'})
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'cnt'
            (directory/'history').mkdir(parents=True)
            original = directory/'compatibility.json'
            original.write_text(json.dumps(fields))
            digest = provenance.sha(original)
            preserved = directory/'history'/f'compatibility_{digest}.json'
            preserved.write_bytes(original.read_bytes())
            current = dict(fields, source_review={'current_sources': 'reviewed change'})
            original.write_text(json.dumps(current))
            with patch.object(provenance,'ANALYSIS',Path(temporary)), \
                 patch.object(provenance,'verify_compatibility',return_value=current):
                report = {'compatibility_sha256': digest}
                result = provenance.verify_analysis_compatibility(directory,{},report)
                self.assertTrue(result['reused_under_review'])
                self.assertEqual(result['producing_compatibility_sha256'],digest)
                self.assertEqual(provenance.sha(preserved),digest)
                changed_project = copy.deepcopy(current)
                changed_project['projects']['Project.toml'] = 'test-only Statistics addition'
                with patch.object(provenance,'verify_compatibility',return_value=changed_project):
                    with self.assertRaisesRegex(ValueError,'target/environment changed: projects'):
                        provenance.verify_analysis_compatibility(directory,{},report)
                changed_project['source_review']['test_project_review'] = dict(
                    baseline_sha256='project',current_sha256='test-only Statistics addition')
                with patch.object(provenance,'verify_compatibility',return_value=changed_project):
                    self.assertTrue(provenance.verify_analysis_compatibility(directory,{},report)['reused_under_review'])
                changed_project['projects']['paper/Manifest.toml'] = 'changed runtime'
                with patch.object(provenance,'verify_compatibility',return_value=changed_project):
                    with self.assertRaisesRegex(ValueError,'target/environment changed: projects'):
                        provenance.verify_analysis_compatibility(directory,{},report)
                for field in ('inputs','settings','parameter_names','likelihood','dependencies'):
                    changed = dict(current)
                    changed[field] = 'changed'
                    with patch.object(provenance,'verify_compatibility',return_value=changed):
                        with self.assertRaisesRegex(ValueError,'target/environment changed'):
                            provenance.verify_analysis_compatibility(directory,{},report)
                preserved.unlink()
                with self.assertRaisesRegex(ValueError,'Missing original'):
                    provenance.verify_analysis_compatibility(directory,{},report)

    def test_gaussian_change_of_variables(self):
        metas=json.loads((BASE/'cnt_grd/priors.json').read_text())
        # Independent integrals use known Normal/lognormal/logit-normal changes
        # of variables. Their medians and CDF integrals are analytic.
        for meta in metas:
            z=np.linspace(-7,7,100001)
            x=display(z,meta)
            density=gaussian_marginal(x,0,1,meta)
            self.assertAlmostEqual(float(np.trapz(density,x)),1.,places=7)
            below=x<=display(np.array([0.]),meta)[0]
            self.assertAlmostEqual(float(np.trapz(density[below],x[below])),.5,places=7)
            if meta['transform']=='log':
                np.testing.assert_allclose(physical(np.array([-3.,2.]),meta),np.exp([-3.,2.]))
            elif meta['transform']=='logit':
                a,b=meta['lower'],meta['upper']
                np.testing.assert_allclose(physical(np.array([0.]),meta),[(a+b)/2])
                np.testing.assert_allclose(physical(np.array([-1000.,1000.]),meta),[a,b])

    def test_support_violations_are_reported_without_filtering(self):
        meta=dict(name='AG',transform='log',plotting_coordinate='log',
                  density_coordinates='inference',family='uniform',arguments=[-1.,1.])
        report=dict(center=[0.],covariance=[[4.]],priors=[meta])
        z,theta,shown,diagnostic=gaussian_draws(report,count=10000,seed=19)
        self.assertEqual(len(z),10000)
        self.assertGreater(diagnostic['prior_support_violations']['AG'],5000)
        np.testing.assert_array_equal(shown,z)
        np.testing.assert_allclose(theta,np.exp(z))
        # Display-space uniform log prior integrates once, not once per crop.
        x=np.linspace(-.5,.5,101)
        self.assertAlmostEqual(float(np.trapz(prior_density(meta,x),x)),.5)

    def test_compatibility_guard_rejects_stale_sources_and_inputs(self):
        folder=BASE/'cnt'
        manifest=json.loads((folder/'manifest.json').read_text())
        receipt=provenance.verify_compatibility(folder,manifest)
        with patch.object(provenance,'numerical_sources',return_value={}):
            with self.assertRaisesRegex(ValueError,'Sources changed'):
                provenance.verify_compatibility(folder,manifest)
        changed=copy.deepcopy(manifest)
        changed['target_hash']='changed'
        with self.assertRaisesRegex(ValueError,'another target'):
            provenance.verify_compatibility(folder,changed)
        original_sha=provenance.sha
        def changed_manifest(path):
            return 'changed' if Path(path)==folder/'manifest.json' else original_sha(path)
        with patch.object(provenance,'sha',side_effect=changed_manifest):
            with self.assertRaisesRegex(ValueError,'Changed compatibility input'):
                provenance.verify_compatibility(folder,manifest)
        self.assertGreaterEqual(len(receipt['checks']),9)

    def test_laplace_outputs_keep_rejections_explicit(self):
        for model,n in (('cnt',4),('cnt_grd',5)):
            report=json.loads((provenance.ANALYSIS/model/'laplace.json').read_text())
            self.assertEqual(len(report['center']),n)
            self.assertTrue(report['no_sampling_or_latent_propagation'])
            self.assertIn(report['status'],('accepted','rejected'))
            if report['status']=='rejected':
                self.assertNotIn('covariance',report)
                self.assertNotIn('hessian',report)
                self.assertFalse(report['diagnostics']['valid'])
            else:
                h=np.asarray(report['hessian']);c=np.asarray(report['covariance'])
                np.testing.assert_allclose(h@c,np.eye(n),atol=1e-7)

    def test_gauss_newton_outputs_are_joint_and_distinct_from_laplace(self):
        for model,n in (('cnt',4),('cnt_grd',5)):
            folder=provenance.ANALYSIS/model
            report=json.loads((folder/'gauss_newton.json').read_text())
            receipt=json.loads((folder/'gauss_newton_comparison_plot.json').read_text())
            self.assertEqual(receipt['approximation_sha256'],provenance.sha(folder/'gauss_newton.json'))
            self.assertEqual(receipt['posterior_sha256'],provenance.sha(BASE/model/'posterior_draws.csv'))
            self.assertEqual(report['approximation'],'gauss_newton')
            self.assertEqual(len(report['center']),n)
            self.assertTrue(report['no_sampling_or_latent_propagation'])
            self.assertNotIn('hessian',report)
            if report['status']=='accepted':
                precision=np.asarray(report['precision']);covariance=np.asarray(report['covariance'])
                jacobian=np.asarray(report['jacobian']);prior=np.asarray(report['prior_hessian'])
                # Independent column solves and posterior-metric eigenvalues.
                independent=np.column_stack([np.linalg.solve(precision,np.eye(n)[:,i]) for i in range(n)])
                np.testing.assert_allclose(covariance,independent,rtol=2e-7)
                np.testing.assert_allclose(precision,jacobian.T@jacobian+prior,rtol=1e-12,atol=1e-8)
                d=report['diagnostics']
                self.assertLessEqual(d['information_error'],d['information_tolerance'])
                self.assertLessEqual(d['gradient_verification'],d['derivative_tolerance'])
                table=np.genfromtxt(folder/'gauss_newton_marginal_comparison.csv',delimiter=',',names=True,dtype=None,encoding='utf8')
                draws=np.genfromtxt(BASE/model/'posterior_draws.csv',delimiter=',',names=True,dtype=None,encoding='utf8')
                from statistics import NormalDist
                for i,meta in enumerate(report['priors']):
                    z=np.asarray(report['center'][i])+np.sqrt(covariance[i,i])*np.array([NormalDist().inv_cdf(.025),0,NormalDist().inv_cdf(.975)])
                    expected=physical(z,meta)
                    for key,value in zip(('GaussNewton_q025','GaussNewton_median','GaussNewton_q975'),expected):
                        self.assertAlmostEqual(table[key][i],value,places=10)
                    self.assertAlmostEqual(table['MCMC_median'][i],np.median(draws[meta['name']]),places=10)
            else:
                self.assertNotIn('covariance',report)

    def test_architecture_dependency_roles(self):
        path=ROOT/'paper/architecture/generate_pbm_architecture.py'
        spec=importlib.util.spec_from_file_location('architecture',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        project=module.check_architecture()
        self.assertIn('NonlinearSolve',project['weakdeps'])
        self.assertNotIn('NonlinearSolve',project['deps'])


if __name__=='__main__':
    unittest.main(verbosity=2)
