"""Independent data, posterior, probability, balance and publication contracts."""
import csv
import argparse
import hashlib
import json
import sys
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
BASE=ROOT/'paper/results/04_protein_crystallization/production'
sys.path.insert(0,str(ROOT/'paper'))
from plot_protein_isothermal import prior_density

def rows(folder,name):
    with (folder/name).open(newline='') as f: return list(csv.DictReader(f))
def table(folder,name):
    return np.atleast_1d(np.genfromtxt(folder/name,delimiter=',',names=True,dtype=None,encoding='utf8'))
def record(folder,name): return json.loads((folder/name).read_text())

class ProteinResults(unittest.TestCase):
    def test_producer_sources_match_checkout(self):
        for name in ('cnt','cnt_grd'):
            manifest=record(BASE/name,'manifest.json')
            for path,digest in manifest['source_files'].items():
                with self.subTest(model=name,path=path):
                    self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),digest,path)

    def test_original_data_split_and_fixed_uncertainties(self):
        folder=ROOT/'examples/data/protein_crystallization/tang2018_isothermal'
        origin=record(folder/'provenance','integration.json')
        for f,digest in origin['data_hashes'].items():
            self.assertEqual(hashlib.sha256((folder/f).read_bytes()).hexdigest(),digest)
        data=table(BASE/'shared','concentration.csv');psd=table(BASE/'shared','psd.csv')
        self.assertEqual(len(data),16)
        self.assertEqual(sum(data['split']=='calibration'),11);self.assertEqual(sum(data['split']=='validation'),4)
        self.assertEqual(data['time_min'][0],0.);self.assertEqual(data['observed_g_L'][0],19.965841161400512)
        np.testing.assert_array_equal(data['sigma_g_L'],.05*data['observed_g_L'])
        original=rows(folder,'concentration.csv')
        for a,b in zip(original,data):
            self.assertEqual(a['observation_id'],b['id'])
            self.assertEqual(float(a['model_time_min']),b['time_min'])
            self.assertEqual(float(a['concentration_g_L']),b['observed_g_L'])
        original=[r for r in rows(folder,'psd_cdf.csv') if r['window']=='isothermal_core']
        self.assertEqual(len(original),len(psd))
        for a in original:
            b=psd[(psd['time_min']==float(a['time_min']))&(psd['size_um']==float(a['size_um']))]
            self.assertEqual(len(b),1)
            self.assertEqual(float(a['conditional_cdf']),b['conditional_cdf'][0])
            self.assertEqual(float(a['conditional_density_per_um']),b['conditional_density_per_um'][0])
        o=record(BASE/'shared','observation_model.json')
        self.assertEqual(o['CDF_absolute_sd'],.03);self.assertEqual(o['CDF_correlation_length_um'],2.)
        self.assertEqual(o['CDF_ordinates_um'],list(range(3,25)));self.assertEqual(o['window_um'],[2.5,25.])
        self.assertFalse(o['quantiles_in_likelihood'])
        qs=table(BASE/'shared','quantiles.csv')
        for t in (74.,180.,420.,660.):
            q=psd[psd['time_min']==t];x,F,f=q['size_um'],q['conditional_cdf'],q['conditional_density_per_um']
            self.assertAlmostEqual(np.trapz(f,x),1.,places=12)
            np.testing.assert_allclose(np.r_[0,np.cumsum(np.diff(x)*(f[1:]+f[:-1])/2)],F,atol=1e-12)
            r=qs[qs['time_min']==t]
            np.testing.assert_allclose(np.interp(r['probability'],F,x),r['observed_um'],atol=1e-12)
            self.assertTrue(all(q['split']==('validation' if t==660 else 'calibration')))

    def test_targets_manifests_and_sampling(self):
        common=[]
        for name,n in (('cnt',4),('cnt_grd',5)):
            folder=BASE/name;m=record(folder,'manifest.json')
            self.assertTrue(m['complete']);self.assertEqual(m['mode'],'production')
            for path,digest in m['files'].items():
                self.assertEqual(hashlib.sha256((folder/path).read_bytes()).hexdigest(),digest,path)
            prior=record(folder,'priors.json');common.append(prior[:4]);self.assertEqual(len(prior),n)
            raw=table(folder,'posterior_draws.csv');names=raw.dtype.names
            self.assertFalse('sigma_c' in names or 'sigma_F' in names)
            self.assertEqual(sum(k.startswith('z') and k[1:].isdigit() for k in names),n)
            d=record(folder,'sampling_diagnostics.json');self.assertTrue(d['passed'])
            self.assertGreaterEqual(d['chains'],2);self.assertEqual(len(raw),d['chains']*d['samples'])
            self.assertLessEqual(max(d['rhat']),1.01)
            self.assertGreaterEqual(min(d['ess_bulk']),400);self.assertGreaterEqual(min(d['ess_tail']),400)
            self.assertEqual(d['divergences'],0);self.assertEqual(d['max_depth_hits'],0)
            self.assertGreaterEqual(min(d['ebfmi']),.3)
            mode=record(folder,'map.json');self.assertTrue(mode['converged'])
            self.assertLessEqual(mode['gradient_norm'],.001);self.assertEqual(len(mode['starts']),3)
            self.assertAlmostEqual(mode['logposterior'],mode['logprior']+mode['loglikelihood_whitened_CDF'],places=10)
            frozen=record(folder,'frozen_model.json');self.assertFalse(frozen['validation_used'])
            metrics=record(folder,'validation_metrics.json');self.assertLessEqual(frozen['frozen_at'],metrics['evaluated_at'])
            if name=='cnt': self.assertTrue(np.all(raw['grd_strength']==0))
            else: self.assertTrue(np.all(raw['grd_strength']>0))
            self.assertTrue(np.all((raw['logAJ']>0)&(raw['logAJ']<100)&(raw['g']>1)&(raw['g']<3)))
        self.assertEqual(common[0],common[1])

    def test_physical_invariants_and_distribution_diagnostics(self):
        for name in ('cnt','cnt_grd'):
            folder=BASE/name;trajectory=table(folder,'trajectory.csv')
            self.assertLess(np.max(np.abs(trajectory['material_error_g_L'])),1e-9)
            np.testing.assert_allclose(trajectory['concentration_g_L']+trajectory['solid_concentration_g_L']-19.965841161400512,
                trajectory['material_error_g_L'],atol=1e-12)
            self.assertGreaterEqual(np.min(trajectory['minimum_scaled_count']),0.)
            self.assertEqual(trajectory['number_per_m3'][0],0.)
            self.assertLess(np.max(trajectory['upper_bin_fraction']),1e-8)
            theta=record(folder,'map.json')['parameters']['grd_strength']
            np.testing.assert_allclose(trajectory['D_G_um2_min'],theta*trajectory['G_um_min']**2,rtol=1e-13,atol=0)
            curves=table(folder,'distribution_curves.csv');observed=table(folder,'psd_predictions.csv')
            metrics=table(folder,'distribution_metrics.csv');quantiles=table(folder,'quantile_predictions.csv')
            for t in (74.,180.,420.,660.):
                q=curves[curves['time_min']==t];o=observed[observed['time_min']==t]
                edges=np.r_[q['left_um'],q['right_um'][-1]];density=q['conditional_density_per_um']
                cdf=np.r_[0,np.cumsum(density*np.diff(edges))]
                self.assertAlmostEqual(cdf[-1],1,places=12)
                np.testing.assert_allclose(np.interp(o['size_um'],edges,cdf),o['predicted_cdf'],atol=2e-12)
                k=quantiles[quantiles['time_min']==t]
                np.testing.assert_allclose(np.interp(k['probability'],cdf,edges),k['predicted_um'],atol=2e-10)
                F=o['observed_cdf'];partition=np.unique(np.r_[cdf,F]);widths=np.diff(partition)
                mid=(partition[:-1]+partition[1:])/2
                u=np.vstack((mid-widths/(2*np.sqrt(3)),mid+widths/(2*np.sqrt(3))))
                qm=np.interp(u,cdf,edges);qo=np.interp(u,F,o['size_um'])
                w2=np.sqrt(np.dot(widths,np.mean((qm-qo)**2,axis=0)))
                r=metrics[metrics['time_min']==t][0]
                self.assertAlmostEqual(w2,r['W2_um'],delta=2e-8)
                self.assertAlmostEqual(np.sqrt(np.mean((o['predicted_cdf']-F)**2)),r['CDF_RMSE'],places=12)
            conc=table(folder,'concentration_predictions.csv');saved=record(folder,'validation_metrics.json')
            for split in ('calibration','validation'):
                r=conc[conc['split']==split]
                self.assertAlmostEqual(np.sqrt(np.mean(r['residual_g_L']**2)),saved[f'{split}_concentration_RMSE_g_L'],places=12)
            np.testing.assert_allclose(conc['standardized_residual'],conc['residual_g_L']/(.05*conc['observed_g_L']),atol=1e-13)
            innovations=table(folder,'likelihood_contributions.csv')
            self.assertEqual(len(innovations),77);self.assertTrue(np.all(innovations['time_min']<=420))
            self.assertAlmostEqual(np.sum(innovations['loglikelihood']),
                record(folder,'map.json')['loglikelihood_whitened_CDF'],delta=1e-9)
            schedule=record(folder,'observation_schedule_check.json')
            self.assertLess(schedule['maximum_CDF_change'],1e-4)
            self.assertLess(schedule['maximum_concentration_change_g_L'],1e-4)
            moment_rates=table(folder,'moment_rate_checks.csv')
            meshes=np.unique(moment_rates['bins'])
            self.assertEqual(len(meshes),2)
            coarse=moment_rates[moment_rates['bins']==meshes[0]]
            fine=moment_rates[moment_rates['bins']==meshes[1]]
            self.assertTrue(np.all(np.abs(fine['relative_rate_difference'])<np.abs(coarse['relative_rate_difference'])))
            np.testing.assert_allclose(moment_rates['bulk_rate_g_L_min']+
                1e-6*1240*.81*moment_rates['discrete_M3_rate'],0.,atol=2e-13)
            # The early empty conditioning window is not assigned fake quantiles.
            self.assertTrue(np.isnan(trajectory['D50_um'][0]))

    def test_latent_draws_summaries_and_refinement(self):
        estimates=table(BASE,'parameter_estimates.csv')
        for name in ('cnt','cnt_grd'):
            folder=BASE/name;raw=table(folder,'posterior_draws.csv');meta=record(folder,'priors.json')
            for p in meta:
                r=estimates[(estimates['model']==name)&(estimates['parameter']==p['name'])][0]
                np.testing.assert_allclose(np.quantile(raw[p['name']],[.025,.5,.975]),[r['q025'],r['median'],r['q975']],rtol=1e-13)
            latent=table(folder,'latent_predictions.csv');curves=table(folder,'latent_distribution_curves.csv')
            np.testing.assert_allclose(latent['concentration_g_L']+latent['solid_concentration_g_L']-19.965841161400512,
                latent['material_error_g_L'],atol=1e-12)
            ids=table(folder,'latent_prediction_indices.csv');self.assertGreaterEqual(len(ids),100)
            self.assertEqual(len(np.unique(latent['draw'])),len(ids))
            diagnostics=table(folder,'latent_draw_diagnostics.csv')
            self.assertEqual(len(diagnostics),len(ids))
            self.assertLess(np.max(diagnostics['maximum_material_error_g_L']),1e-9)
            self.assertGreaterEqual(np.min(diagnostics['minimum_scaled_count']),0.)
            self.assertLess(np.max(diagnostics['maximum_upper_bin_fraction']),1e-8)
            verified=record(folder,'target_verification.json')
            self.assertTrue(verified['passed'])
            self.assertEqual(verified['target_hash'],record(folder,'manifest.json')['target_hash'])
            self.assertEqual(verified['posterior_draws_sha256'],hashlib.sha256((folder/'posterior_draws.csv').read_bytes()).hexdigest())
            self.assertEqual(len(verified['points']),2*record(folder,'sampling_diagnostics.json')['chains'])
            for t in (74.,180.,420.,660.):
                q=curves[curves['time_min']==t]
                for draw in np.unique(q['draw']):
                    r=q[q['draw']==draw]
                    self.assertAlmostEqual(np.dot(r['right_um']-r['left_um'],r['conditional_density_per_um']),1.,places=11)
            checks=table(folder,'refinement.csv')
            self.assertEqual(set(checks['check']),{'time','size','domain'})
            self.assertTrue(np.all(np.isfinite(checks['maximum_CDF_change'])))
            self.assertLess(checks[checks['check']=='time']['maximum_CDF_change'][0],.003)
            self.assertLess(checks[checks['check']=='domain']['maximum_CDF_change'][0],.003)
            self.assertLess(checks[checks['check']=='size']['maximum_CDF_change'][0],.006)
            self.assertLess(checks[checks['check']=='size']['maximum_concentration_change_g_L'][0],.03)
            if name=='cnt_grd':
                finer=table(folder,'additional_size_refinement.csv')
                self.assertEqual(finer['coarse_bins'][0],200)
                self.assertEqual(finer['fine_bins'][0],400)
                self.assertLess(finer['maximum_CDF_change'][0],.001)
            posterior=table(folder,'posterior_refinement.csv')
            self.assertTrue(np.all(np.isfinite(posterior['maximum_CDF_change'])))
            self.assertLess(np.max(posterior['maximum_CDF_change']),.01)
        fixed=estimates[(estimates['model']=='cnt')&(estimates['parameter']=='grd_strength')][0]
        self.assertEqual(fixed['MAP'],0.);self.assertTrue(np.isnan(fixed['median']))

    def test_analytical_prior_plot_coordinates(self):
        metas=record(BASE/'shared','priors.json')
        for m in metas:
            if m['name']=='AG':
                x=np.linspace(np.log(.01),np.log(1500),10001)
                np.testing.assert_allclose(prior_density(m,x),1/np.log(150000),rtol=1e-13)
            elif m['name']=='grd_strength':
                x=np.linspace(-25,np.log(400),20001)
                np.testing.assert_allclose(prior_density(m,np.array([np.log(10.)])),[np.exp(-1)],rtol=1e-13)
            elif m['family']=='uniform': x=np.linspace(*m['arguments'],10001)
            else: x=np.geomspace(np.exp(m['arguments'][0]-8*m['arguments'][1]),np.exp(m['arguments'][0]+8*m['arguments'][1]),20001)
            self.assertAlmostEqual(np.trapz(prior_density(m,x),x),1.,delta=2e-6)
        # A cropped broad prior must retain its original density/normalizer.
        logAJ=metas[0];self.assertAlmostEqual(np.trapz(prior_density(logAJ,np.linspace(25,35,101)),np.linspace(25,35,101)),.1,places=12)

    def test_five_publication_figures(self):
        import pymupdf
        from PIL import Image
        folder=ROOT/'paper/figures/04_protein_crystallization/production'
        for name in ('protein_data','protein_fit_cnt','protein_fit_cnt_grd','protein_corner_cnt','protein_corner_cnt_grd'):
            with Image.open(folder/f'{name}.png') as image:
                self.assertEqual(image.width,2100);np.testing.assert_allclose(image.info['dpi'],[300,300],atol=.01)
            with pymupdf.open(folder/f'{name}.pdf') as pdf:
                self.assertEqual(len(pdf),1);self.assertAlmostEqual(pdf[0].rect.width,504,places=2)
                for font in pdf[0].get_fonts(full=True): self.assertTrue(pdf.extract_font(font[0])[3],font)
            for ext in ('png','svg'):
                self.assertEqual(hashlib.sha256((folder/f'{name}.{ext}').read_bytes()).hexdigest(),
                    hashlib.sha256((ROOT/f'docs/src/assets/protein_isothermal/{name}.{ext}').read_bytes()).hexdigest())

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retained',action='store_true',help=
        'Check stored observations, outputs and posterior diagnostics without asserting that the current checkout is the producer.')
    args,remaining=parser.parse_known_args()
    if args.retained:
        print('Retained-result check: producer/current-checkout equivalence is NOT asserted.',flush=True)
        names=[name for name in unittest.defaultTestLoader.getTestCaseNames(ProteinResults)
               if name!='test_producer_sources_match_checkout']
        suite=unittest.TestSuite(ProteinResults(name) for name in names)
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(not result.wasSuccessful())
    unittest.main(argv=[sys.argv[0],*remaining])
