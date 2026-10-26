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
import plot_all as plots

CHECK_CHECKOUT=True

class GrowthData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base,cls.info,cls.runs = plots.load_case(1,check_checkout=CHECK_CHECKOUT)

    def test_bin_integrals_and_width_weighted_norms(self):
        for name,(folder,_) in self.runs.items():
            data = plots.table(folder/"profile.csv")
            metric = plots.rows(folder/"metrics.csv")[0]
            widths = data["right"]-data["left"]
            np.testing.assert_allclose(data["density"]*widths,data["number"],rtol=2e-14,atol=1e-30)
            error = data["density"]-data["exact_density"]
            self.assertAlmostEqual(float(metric["rmse"]),float(np.sqrt(np.dot(widths,error**2)/widths.sum())),delta=1e-12)
            self.assertAlmostEqual(float(metric["l1"]),float(np.dot(widths,np.abs(error))),delta=1e-12)
            self.assertAlmostEqual(float(metric["negative_number"]),float(-np.minimum(data["number"],0).sum()),delta=1e-13)
            if name.startswith("front_"):
                self.assertEqual(len(data),100)
                a = np.maximum(data["left"]-.4,0)
                b = np.maximum(data["right"]-.4,0)
                # Independent CDF integration; cancellation in negligible tails
                # is covered by an absolute population tolerance.
                exact = np.exp(-a/.01)-np.exp(-b/.01)
                np.testing.assert_allclose(data["exact_number"],exact,rtol=2e-12,atol=5e-15)

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

    def test_common_partition_norm_and_unit_conversion(self):
        a,b = np.array([0.,.5,1.]),np.array([0.,.25,1.])
        # Three subinterval contributions are 0.25, 0.25 and 1.5;
        # the reference inventory is 2.5.
        self.assertAlmostEqual(plots.density_l1(a,np.array([2.,0.]),b,np.array([1.,3.])),.8)
        self.assertAlmostEqual(plots.density_l1(1000*a,np.array([2.,0.])/1000,
            1000*b,np.array([1.,3.])/1000),.8)

    def test_plot_input_rejects_wrong_mode_and_changed_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            paper = Path(temporary)
            base = paper/"results"/plots.CASES[0]/plots.MODE
            base.mkdir(parents=True)
            for key,relative in self.info["runs"].items():
                shutil.copytree(self.base/relative,base/relative)
            shutil.copy(self.base/"index.toml",base/"index.toml")
            for item in self.info.get("derived_files",[]):
                shutil.copy(self.base/item["path"],base/item["path"])
            with mock.patch.object(plots,"PAPER",paper),mock.patch.object(plots,"source_digest",return_value=self.info["source_digest"]),mock.patch.object(plots,"digest",return_value=self.info["workflow_digest"]):
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

class MomentData(unittest.TestCase):
    def test_analytical_reference_layouts_and_shape_bias(self):
        base,info,runs = plots.load_case(2,check_checkout=CHECK_CHECKOUT)
        for method in plots.MOMENT_METHODS:
            data = plots.table(runs[method+"_dt1"][0]/"moments.csv")
            arrays = [data[data["order"]==k] for k in range(4)]
            for k,d in enumerate(arrays):
                times = d["time_s"]
                steady = math.sqrt(.002)
                r = (1-steady)/(1+steady)
                number = steady*(1+r*np.exp(-steady*times))/(1-r*np.exp(-steady*times))
                exact = number*(.001/number)**(k/3)*math.gamma(1+k/3)
                np.testing.assert_allclose(d["exact"],exact,rtol=3e-14)
                np.testing.assert_allclose(d["signed_difference"],d["moment"]-d["exact"],atol=1e-16)
                np.testing.assert_allclose(d["absolute_relative_percent"],100*np.abs(d["moment"]-d["exact"])/d["exact"],rtol=2e-8,atol=3e-12)
            if method=="lognormal":
                reconstructed = arrays[0]["moment"]*(arrays[2]["moment"]/arrays[1]["moment"])**3
                np.testing.assert_allclose(arrays[3]["moment"],reconstructed,rtol=3e-14)
                self.assertGreater(arrays[3]["moment"][0]/.001,1.02)
                self.assertTrue(all(str(v).lower()=="false" for v in arrays[3]["tracked"]))
            else:
                self.assertLess(np.max(np.abs(arrays[0]["moment"]/arrays[0]["exact"]-1)),1e-5)
                self.assertLess(np.max(np.abs(arrays[3]["moment"]/.001-1)),1e-6)
        adaptive = plots.table(runs["adaptive3_dt1"][0]/"moments.csv")
        self.assertEqual(set(adaptive["active_rank"]),{3})

class PrecipitationData(unittest.TestCase):
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
            self.assertEqual(settings["maximum_bdf_order"],3)
            self.assertEqual(settings["late_maximum_bdf_order"],2)
            self.assertEqual(summary["restarts"],1)
            self.assertEqual(summary["phase_end_times"],[5.,200.])
            self.assertEqual(summary["phase_maximum_orders"],[3,2])
            self.assertTrue(all(1<=actual<=cap for actual,cap in zip(summary["phase_maximum_observed_orders"],[3,2])))
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
    for i,case in enumerate((GrowthData,MomentData,PrecipitationData),1):
        if args.case in ("all",str(i)):
            suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(case))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
