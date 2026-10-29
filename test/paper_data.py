"""Independent numerical/data-contract checks, without screenshot assertions."""
import argparse
import csv
import gzip
import math
from pathlib import Path
import shutil
import tempfile
import tomllib
import unittest
from unittest import mock
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"paper"))
from _support import figures as plots

CHECK_CHECKOUT=True

class GrowthData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base,cls.info,cls.runs = plots.load_case(1,check_checkout=CHECK_CHECKOUT)

    def test_bin_integrals_and_width_weighted_norms(self):
        self.assertEqual(self.info['configuration']['profile_family'],'gaussian')
        for name,(folder,record) in self.runs.items():
            data = plots.table(folder/"profile.csv")
            metric = plots.rows(folder/"metrics.csv")[0]
            widths = data["right"]-data["left"]
            np.testing.assert_allclose(data["density"]*widths,data["number"],rtol=2e-14,atol=1e-30)
            error = data["density"]-data["exact_density"]
            self.assertAlmostEqual(float(metric["rmse"]),float(np.sqrt(np.dot(widths,error**2)/widths.sum())),delta=1e-12)
            self.assertAlmostEqual(float(metric["l1"]),float(np.dot(widths,np.abs(error))),delta=1e-12)
            self.assertAlmostEqual(float(metric["negative_number"]),float(-np.minimum(data["number"],0).sum()),delta=1e-13)
            if name.startswith("gaussian_"):
                settings=record['settings']
                self.assertEqual(len(data),settings['bins'])
                self.assertEqual(settings['standard_deviation'],.03)
                shift=settings['growth']*settings['tf']
                a=np.maximum(data['left']-shift,0)
                b=np.minimum(data['right']-shift,1)
                # Independent normal-CDF integration; small tails use an
                # absolute population tolerance rather than density clipping.
                z=lambda x:(x-settings['center'])/(math.sqrt(2)*settings['standard_deviation'])
                erf=np.vectorize(math.erf)
                scale=settings['height']*settings['standard_deviation']*math.sqrt(math.pi/2)
                exact=np.where(b>a,scale*(erf(z(b))-erf(z(a))),0.)
                np.testing.assert_allclose(data['exact_number'],exact,rtol=2e-12,atol=2e-14)

    def test_observed_orders_and_characteristic_resolution(self):
        data = plots.table(self.base/"observed_orders.csv")
        for method in plots.METHODS:
            part = data[data["method"]==method]
            for error,order in (("rmse","observed_rmse_order"),("l1","observed_l1_order")):
                for i in range(1,len(part)):
                    a,b = part[i-1],part[i]
                    if min(a[error],b[error])<=b["roundoff_limit"]:
                        self.assertTrue(np.isnan(b[order]))
                    else:
                        expected = math.log(a[error]/b[error])/math.log(b["bins"]/a["bins"])
                        self.assertAlmostEqual(b[order],expected,places=12)

    def test_time_methods_courants_and_steps(self):
        methods={'upwind':'ForwardEuler','muscl':'SSPRK33','weno3':'SSPRK33',
                 'weno5_ssprk33':'SSPRK33','weno5':'RK4',
                 'hoc':'HOCFiniteVolume','inguva2022':'Inguva2022'}
        self.assertEqual(set(plots.METHODS),set(methods))
        for folder,record in self.runs.values():
            settings=record['settings']
            metric=plots.rows(folder/'metrics.csv')[0]
            name=settings['method']
            self.assertTrue(settings['api_time_method'].endswith(methods[name]))
            self.assertEqual(float(metric['courant']),settings['courant'])
            expected=settings['tf']*settings['growth']*settings['bins']/settings['courant']
            steps=int(metric['steps'])
            self.assertGreaterEqual(steps,round(expected))
            self.assertLessEqual(steps,round(expected)+1)
            if name=='inguva2022':
                self.assertEqual(settings['courant'],1.)
                self.assertEqual(steps,round(expected))

    def test_common_partition_norm_and_unit_conversion(self):
        a,b = np.array([0.,.5,1.]),np.array([0.,.25,1.])
        # Three subinterval contributions are 0.25, 0.25 and 1.5;
        # the reference inventory is 2.5.
        self.assertAlmostEqual(plots.density_l1(a,np.array([2.,0.]),b,np.array([1.,3.])),.8)
        self.assertAlmostEqual(plots.density_l1(1000*a,np.array([2.,0.])/1000,
            1000*b,np.array([1.,3.])/1000),.8)

    def test_plot_input_rejects_wrong_mode_and_changed_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            paper = Path(temporary)/"paper"
            paper.mkdir()
            base = (paper.parent/".cache/paper/checks"/plots.CASES[0] if plots.MODE == "smoke"
                    else paper/plots.CASES[0]/"data")
            base.mkdir(parents=True)
            for key,relative in self.info["runs"].items():
                shutil.copytree(self.base/relative,base/relative)
            shutil.copy(self.base/"index.toml",base/"index.toml")
            for item in self.info.get("derived_files",[]):
                shutil.copy(self.base/item["path"],base/item["path"])
            with mock.patch.object(plots,"ROOT",paper.parent), mock.patch.object(plots,"PAPER",paper),mock.patch.object(plots,"source_digest",return_value=self.info["source_digest"]),mock.patch.object(plots,"digest",return_value=self.info["workflow_digest"]):
                plots.load_case(1)
                with mock.patch.object(plots,"source_digest",return_value="different-checkout"):
                    with self.assertRaisesRegex(ValueError,"Numerical source differs"):
                        plots.load_case(1)
                    plots.load_case(1,check_checkout=False)
                index = base/"index.toml"
                original = index.read_text()
                wrong = "smoke" if plots.MODE=="production" else "production"
                index.write_text(original.replace(f'mode = "{plots.MODE}"',f'mode = "{wrong}"'))
                with self.assertRaisesRegex(ValueError,"Incorrect result mode"):
                    plots.load_case(1)
                index.write_text(original)
                folder = next(iter(self.info["runs"].values()))
                provenance = base/folder/"provenance.toml"
                original_provenance = provenance.read_text()
                count = tomllib.loads(original_provenance)["environment"]["julia_threads"]
                provenance.write_text(original_provenance.replace(f"julia_threads = {count}",f"julia_threads = {count+1}",1))
                with self.assertRaisesRegex(ValueError,"Mixed compute environments"):
                    plots.load_case(1)
                provenance.write_text(original_provenance)
                with open(base/folder/"profile.csv","a") as stream:
                    stream.write("changed\n")
                with self.assertRaisesRegex(ValueError,"Changed numerical data"):
                    plots.load_case(1)
                with self.assertRaisesRegex(ValueError,"Changed numerical data"):
                    plots.load_case(1,check_checkout=False)

class MillingData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if plots.MODE == "smoke":
            raise unittest.SkipTest("Milling paper figures use completed production inputs")
        cls.base,cls.info,cls.runs = plots.load_case(2,check_checkout=CHECK_CHECKOUT)

    def test_original_producers_and_single_forward_selection(self):
        import hashlib,json
        forward,uq = self.runs['forward'][1],self.runs['uq'][1]
        self.assertEqual(uq['settings']['mc_samples'],12000)
        self.assertEqual(self.info['configuration']['forward_statistic'],'pivot_CDF_d50_um')
        data=plots.table(self.base/'forward.csv')
        self.assertEqual(set(data['bins']),{100})
        self.assertEqual(set(data['rpm']),{3350,4923})
        self.assertTrue(np.all(data['minimum_inventory']>=0))
        self.assertLess(np.max(np.abs(data['material_defect_kg'])),1e-9)
        for speed in (3350,4923):
            part=data[data['rpm']==speed]
            self.assertTrue(np.all(np.diff(part['time_s'])>0))
            self.assertTrue(np.all(part['pivot_CDF_d50_um']>0))
        # The reference coordinates are original figure readings, not generated
        # from the package trajectory or moved to agree with it.
        reference=self.base.parents[2]/'examples/data/milling_uncertainty/barrasso_figure7.csv'
        if reference.exists():
            self.assertEqual((self.base/'reference.csv').read_bytes(),reference.read_bytes())
        for kind,name in (('forward','forward_producer.json'),('uq','uq_producer.json')):
            source=plots.ROOT/self.info[kind+'_source']/'provenance.json'
            if source.exists(): self.assertEqual((self.base/name).read_bytes(),source.read_bytes())
        self.assertEqual(set(self.info['configuration']['figures']),{'milling_forward','milling_uq'})

    def test_uq_shared_times_and_unchanged_output_bands(self):
        nominal=plots.table(self.base/'nominal.csv')
        self.assertTrue(np.all(np.diff(nominal['time_s'])>0))
        self.assertTrue({29.999,30.,30.001,31.}.issubset(set(nominal['time_s'])))
        for name in ('mc','pce1','pce2'):
            d=plots.table(self.base/f'{name}_summary.csv')
            np.testing.assert_array_equal(d['time_s'],nominal['time_s'])
            self.assertTrue(np.all(np.isfinite(d['mean_um'])))
            self.assertTrue(np.all(d['sd_um']>=0))
            np.testing.assert_allclose(d['gaussian_low_um'],d['mean_um']-1.96*d['sd_um'],rtol=2e-14)
            np.testing.assert_allclose(d['gaussian_high_um'],d['mean_um']+1.96*d['sd_um'],rtol=2e-14)
            self.assertTrue(np.all(d['q025_um']<=d['median_um']))
            self.assertTrue(np.all(d['median_um']<=d['q975_um']))
            source=plots.ROOT/self.info['uq_source']/f'{name}_summary.csv'
            if source.exists(): self.assertEqual((self.base/f'{name}_summary.csv').read_bytes(),source.read_bytes())

    def test_changed_curve_or_producer_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'data'
            shutil.copytree(self.base,target)
            original=(target/'mc_summary.csv').read_bytes()
            with open(target/'mc_summary.csv','a') as stream: stream.write('changed\n')
            with self.assertRaisesRegex(ValueError,'Changed numerical data'):
                plots.load_case(2,check_checkout=False,input_path=target)
            (target/'mc_summary.csv').write_bytes(original)
            with open(target/'uq_producer.json','a') as stream: stream.write('\n')
            with self.assertRaisesRegex(ValueError,'Changed numerical data'):
                plots.load_case(2,check_checkout=False,input_path=target)

class PrecipitationData(unittest.TestCase):
    def test_separate_refinement_coordinates_and_reference_controls(self):
        base,info,runs = plots.load_case(3,check_checkout=CHECK_CHECKOUT)
        cfg = info['configuration']
        if plots.MODE == 'production':
            self.assertEqual((cfg['reference_nx'],cfg['reference_nz']),(1000,500))
            self.assertEqual(cfg['size_bins'],[50,100,200,400])
            self.assertEqual(cfg['axial_cells'],[50,100,200,400])
        reference = runs['reference'][1]['settings']
        temporal = runs['time_refinement'][1]['settings']
        self.assertEqual((reference['nx'],reference['nz']),(temporal['nx'],temporal['nz']))
        self.assertEqual(reference['physical_parameters'],temporal['physical_parameters'])
        for name in ('relative_tolerance','population_absolute_tolerance','bulk_absolute_tolerance'):
            self.assertAlmostEqual(reference[name]/temporal[name],cfg['temporal_refinement_factor'])
        self.assertEqual(reference['maximum_step']/temporal['maximum_step'],2.)
        for name,(_,record) in runs.items():
            s = record['settings']
            if name.startswith('size_Nx'):
                self.assertEqual(s['nz'],cfg['reference_nz'])
            if name.startswith('axial_Nz'):
                self.assertEqual(s['nx'],cfg['reference_nx'])
        # An unresolved early outlet must remain distinguishable from an
        # accurate relative error. Reconstruct the flag from the independently
        # exported inventory and the declared, dimensional tolerance scale.
        account = plots.table(runs['reference'][0]/'accounting.csv')
        nominal = reference['nx']*reference['population_absolute_tolerance']*reference['population_scaling']
        for row in plots.rows(base/'refinement.csv'):
            i = reference['times'].index(float(row['time_s']))
            number = account['outlet_number_per_m3'][i]
            above = number > nominal
            self.assertEqual(row['outlet_reference_above_nominal_absolute_scale'].lower()=='true',above)
            self.assertEqual(float(row['nominal_outlet_absolute_tolerance_scale_per_m3']),nominal)
            self.assertAlmostEqual(float(row['reference_outlet_number_per_m3']),number,
                delta=3e-14*max(1,abs(number)))
            if above:
                self.assertEqual(float(row['interpretable_outlet_L1']),float(row['normalized_outlet_L1']))
            else:
                self.assertTrue(np.isnan(float(row['interpretable_outlet_L1'])))

    def test_geometry_scaling_normalization_and_accounting(self):
        base,info,runs = plots.load_case(3,check_checkout=CHECK_CHECKOUT)
        for name,(folder,provenance) in runs.items():
            if provenance["summary"].get("role")=="verification":
                continue
            settings = provenance["settings"]
            summary = provenance["summary"]
            self.assertEqual(summary["status"],"completed")
            self.assertEqual(summary["completed_time"],settings["tf"])
            self.assertEqual(summary["steps"],summary["checked_steps"])
            self.assertGreaterEqual(summary["growth_jacobian_updates"],1)
            self.assertGreaterEqual(summary["preconditioner_setups"],summary["growth_jacobian_updates"])
            self.assertEqual(settings["maximum_bdf_order"],info['configuration']['maximum_bdf_order'])
            self.assertEqual(settings["late_maximum_bdf_order"],info['configuration']['late_maximum_bdf_order'])
            restart = settings['late_maximum_bdf_order'] != 0
            caps = [settings['maximum_bdf_order'],settings['late_maximum_bdf_order']] if restart else [settings['maximum_bdf_order']]
            self.assertEqual(summary["restarts"],int(restart))
            self.assertEqual(summary["phase_end_times"],[settings['bdf_restart_time'],200.] if restart else [200.])
            self.assertEqual(summary["phase_maximum_orders"],caps)
            self.assertTrue(all(1<=actual<=cap for actual,cap in zip(summary["phase_maximum_observed_orders"],caps)))
            self.assertEqual(sum(summary["phase_steps"]),summary["steps"])
            self.assertGreaterEqual(summary["minimum_scaled_population"],-1e-8)
            self.assertGreaterEqual(summary["minimum_solute"],-1e-8)
            x = plots.table(folder/"size_edges.csv")["size_m"]
            z = plots.table(folder/"axial_edges.csv")["z_m"]
            np.testing.assert_allclose(np.diff(np.log(x)),np.diff(np.log(x))[0],rtol=2e-12)
            self.assertGreater(np.ptp(np.diff(x)),0)
            self.assertEqual(settings["population_scaling"],1e12)
            self.assertEqual(settings["size_reconstruction"],"geometry-aware WENO5")
            account = plots.table(folder/"accounting.csv")
            np.testing.assert_array_equal(account["time_s"],settings["times"])
            parameters = settings["physical_parameters"]
            np.testing.assert_allclose(account["inlet_kg_s"],parameters["Q"]*parameters["cfeed"],rtol=2e-13)
            np.testing.assert_allclose(account["column_total_kg"],parameters["area"]*parameters["porosity"]*
                (account["dissolved_inventory_kg_m2"]+account["solid_inventory_kg_m2"]),rtol=3e-14)
            for i,t in enumerate(settings["times"],1):
                with gzip.open(folder/f"population_{i}.csv.gz","rt") as stream:
                    numbers = np.loadtxt(stream,skiprows=1).reshape((settings["nx"],settings["nz"]),order="F")
                centers = (x[:-1]+x[1:])/2
                material = np.sum(centers[:,None]**3*numbers*np.diff(z)[None,:])
                self.assertAlmostEqual(account["number_per_m2"][i-1],
                    float(np.sum(numbers*np.diff(z)[None,:])),delta=3e-14*max(1,abs(account["number_per_m2"][i-1])))
                expected = settings["physical_parameters"]["rho"]*settings["physical_parameters"]["kv"]*material
                self.assertAlmostEqual(account["solid_inventory_kg_m2"][i-1],expected,delta=1e-11*max(1,abs(expected)))
            scale = np.abs(account["material_inlet_kg_m2_s"])+np.abs(account["material_outlet_kg_m2_s"])
            self.assertTrue(np.all(np.abs(account["material_rate_defect_kg_m2_s"])<1e-11*np.maximum(scale,1e-10)))
            outlet = plots.table(folder/"outlet.csv")
            np.testing.assert_allclose(outlet["number_density_m4"]*np.diff(x),outlet["number_per_m3"],rtol=2e-14,atol=1e-20)
            self.assertAlmostEqual(float(np.sum(outlet["volume_density_per_m"]*np.diff(x))),1.,places=12)
            material = parameters["rho"]*parameters["kv"]*np.dot(((x[:-1]+x[1:])/2)**3,outlet["number_per_m3"])
            expected_outflow = parameters["Q"]*(account["outlet_solute_kg_m3"][-1]+material)
            self.assertAlmostEqual(account["outlet_kg_s"][-1],expected_outflow,delta=2e-13*max(abs(expected_outflow),1e-15))
        reference = plots.table(base/"refinement.csv")
        self.assertEqual(set(reference["reference_nx"]),{info["configuration"]["reference_nx"]})
        self.assertEqual(set(reference["reference_nz"]),{info["configuration"]["reference_nz"]})
        truth = plots.table(runs["reference"][0]/"outlet.csv")
        reference_edges = np.r_[truth["left_m"],truth["right_m"][-1]]
        for comparison in reference:
            if comparison["time_s"] != 200.:
                continue
            values = plots.table(runs[comparison["comparison"]][0]/"outlet.csv")
            edges = np.r_[values["left_m"],values["right_m"][-1]]
            independent = plots.density_l1(edges,values["number_density_m4"],reference_edges,truth["number_density_m4"])
            self.assertAlmostEqual(independent,comparison["normalized_outlet_L1"],delta=2e-13)

    def test_whole_field_error_and_literature_provenance(self):
        base,info,runs = plots.load_case(3,check_checkout=CHECK_CHECKOUT)
        path,producer = runs['reference']
        nx,nz = producer['settings']['nx'],producer['settings']['nz']
        x = plots.table(path/'size_edges.csv')['size_m']
        z = plots.table(path/'axial_edges.csv')['z_m']
        with gzip.open(path/'population_4.csv.gz','rt') as stream:
            truth = np.loadtxt(stream,skiprows=1).reshape((nx,nz),order='F')/np.diff(x)[:,None]
        for row in plots.rows(base/'refinement.csv'):
            if float(row['time_s']) != 200.:
                continue
            path,record = runs[row['comparison']]
            a = plots.table(path/'size_edges.csv')['size_m']
            b = plots.table(path/'axial_edges.csv')['z_m']
            with gzip.open(path/'population_4.csv.gz','rt') as stream:
                density = np.loadtxt(stream,skiprows=1).reshape((record['settings']['nx'],record['settings']['nz']),order='F')/np.diff(a)[:,None]
            common_x,common_z = np.unique(np.r_[x,a]),np.unique(np.r_[z,b])
            mx,mz = (common_x[:-1]+common_x[1:])/2,(common_z[:-1]+common_z[1:])/2
            ix,iz = np.searchsorted(x,mx)-1,np.searchsorted(z,mz)-1
            ia,ib = np.searchsorted(a,mx)-1,np.searchsorted(b,mz)-1
            weights = np.diff(common_x)[:,None]*np.diff(common_z)[None,:]
            ref = truth[np.ix_(ix,iz)]
            error = np.sum(np.abs(density[np.ix_(ia,ib)]-ref)*weights)/np.sum(ref*weights)
            self.assertAlmostEqual(error,float(row['normalized_field_L1']),delta=3e-13)
        if plots.MODE == 'production':
            import json,hashlib
            meta = json.loads((base/'literature/provenance.json').read_text())
            digitized = plots.table(base/'literature/zhang_figure9c_digitized.csv')
            self.assertEqual(meta['digitization']['figure'],'9(c)')
            self.assertEqual(len(digitized),17)
            source = base.parents[2]/'local_papers/Zhang2024.pdf'
            if source.exists():
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),meta['digitization']['pdf_sha256'])
            # Pixel-row calibration is independent of any package prediction.
            expected = (1098-digitized['pixel_row'])*1.2e15/496
            np.testing.assert_allclose(digitized['number_density_m4'],expected,rtol=3e-14)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--retained",action="store_true",help="Validate stored numerical data and internal provenance without claiming current-checkout equivalence")
    parser.add_argument("--case",choices=["all","1","2","3"],default="all")
    args = parser.parse_args()
    CHECK_CHECKOUT=not args.retained
    if args.retained:
        print("Retained-result check: producer/current-checkout equivalence is NOT asserted.",flush=True)
    plots.MODE = "smoke" if args.smoke else "production"
    suite = unittest.TestSuite()
    for i,case in enumerate((GrowthData,MillingData,PrecipitationData),1):
        if args.case in ("all",str(i)):
            suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(case))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
