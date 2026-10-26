"""Extract public numerical data and explicitly calibrated Figure 9(c) points.

The source PDFs remain outside version control. The raster coordinates below
refer to the original 1801 x 1197 embedded Figure 9, not a rescaled screenshot.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import pymupdf

PAPER = Path(__file__).resolve().parents[3]/"paper"

def extract(pdf,si):
    folder = PAPER / "precipitation/data/literature"
    raw = folder / "cadet" / "reference.h5"
    download = json.loads((raw.parent / "download.json").read_text())
    expected = next(f["sha256"] for f in download["files"] if f["file"] == "reference.h5")
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == expected
    with h5py.File(raw) as h:
        size_order = int(h["input/model/unit_001/reaction/CRY_GROWTH_SCHEME_ORDER"][()])
        axial_order = int(h["input/model/unit_001/discretization/weno/WENO_ORDER"][()])
        if (size_order,axial_order) != (1,2):
            raise ValueError("Recheck the public CADET scheme labels against this input")
        edges = h["input/model/unit_001/reaction/CRY_BINS"][:]
        centers = (edges[:-1]+edges[1:])/2
        density = h["output/solution/unit_001/SOLUTION_OUTLET"][-1, 1:-1]
        t = float(h["output/solution/SOLUTION_TIMES"][-1])
        assert t == 200 and len(density) == 200 and np.all(density >= 0)
        numbers = density*np.diff(edges)
        volume_density = numbers*centers**3/(sum(numbers*centers**3)*np.diff(edges))
        np.savetxt(folder/"cadet_reference.csv", np.column_stack((edges[:-1], edges[1:], density, volume_density)),
                   delimiter=",", header="left_m,right_m,number_density_m4,volume_density_per_m", comments="")
        info = dict(source="public CADET verification numerical output; not the Figure 9 WENO35 run",
            extraction_environment=dict(python=sys.version,numpy=np.__version__,h5py=h5py.__version__,pymupdf=pymupdf.VersionBind),
            download=download, time_s=t, nx=200, nz=int(h["input/model/unit_001/discretization/NCOL"][()]),
            cadet_version=h["meta/CADET_VERSION"][()].decode(),
            cadet_commit=h["meta/CADET_COMMIT"][()].decode(),
            size_scheme="Upwind; CRY_GROWTH_SCHEME_ORDER=1", axial_scheme="WENO23; WENO_ORDER=2",
            relative_tolerance=float(h["input/solver/time_integrator/RELTOL"][()]),
            absolute_tolerance=float(h["input/solver/time_integrator/ABSTOL"][()]),
            flow_m3_s=float(h["input/model/connections/switch_000/CONNECTIONS"][4]),
            number_per_m3=float(numbers.sum()), solute_kg_m3=float(h["output/solution/unit_001/SOLUTION_OUTLET"][-1, 0]),
            reference_csv_sha256=hashlib.sha256((folder/"cadet_reference.csv").read_bytes()).hexdigest())
    document = pymupdf.open(pdf)
    page = document[13]
    image = next(v for v in page.get_images() if v[2:4] == (1801,1197))
    pix = pymupdf.Pixmap(document,image[0])
    pixels = np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,pix.n)[:,:,:3].astype(float)
    # Calibration: x=1,10,100 um at columns 130,356,582; n=0 and
    # 1.2e15 m^-4 at rows 1098 and 602. Conservative +/-2-pixel uncertainty.
    sizes = np.array([1.5,2,2.5,3,4,5,6,7,8,10,12,15,20,25,30,35,40.])
    points = []
    for size in sizes:
        col = round(130+226*np.log10(size))
        patch = pixels[550:1099,col-1:col+2]
        blue = (patch[:,:,2]-patch[:,:,0]>70)&(patch[:,:,1]-patch[:,:,0]>25)&(patch[:,:,2]-patch[:,:,1]>10)
        rows = np.where(blue)[0]+550
        if not len(rows):
            raise ValueError(f"no unambiguous blue curve at {size} um")
        row = np.median(rows[rows <= rows.min()+4])
        value = (1098-row)*1.2e15/496
        points.append((size*1e-6,value,size*1e-6*(10**(2/226)-1),2*1.2e15/496,col,row))
    np.savetxt(folder/"zhang_figure9c_digitized.csv",points,delimiter=",",
        header="size_m,number_density_m4,size_uncertainty_m,density_uncertainty_m4,pixel_column,pixel_row",comments="")
    info["digitization"] = dict(doi="10.1016/j.compchemeng.2024.108612",figure="9(c)",page=14,
        pdf_sha256=hashlib.sha256(Path(pdf).read_bytes()).hexdigest(),image_size=[1801,1197],
        nx=200,nz=200,size_scheme="WENO35",axial_scheme="WENO35",time_s=200,
        calibration="log10(x_um)=(column-130)/226; n=(1098-row)*1.2e15/496",
        extraction="blue number-density curve; 1.5--40 um only; uppermost blue trace before the number/volume-curve intersection",
        uncertainty="conservative +/-2 pixels for calibration, antialiasing and line thickness; not experimental noise",
        excluded="ambiguous intersections and small-density tails; no full-domain digitization error is claimed")
    info["physical_parameter_source"] = dict(
        section="6.7.1, Case 7A",tables="Supporting Information S8 and S10",
        supplementary_pdf_sha256=hashlib.sha256(Path(si).read_bytes()).hexdigest(),
        package_flow_m3_s=1.67e-7,
        convention="Interstitial axial velocity Q/(cross-sectional area times porosity)")
    (folder/"provenance.json").write_text(json.dumps(info,indent=2)+"\n")
    print("Extracted",len(density),"public numerical bins and",len(points),"Figure 9(c) points")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf",type=Path,default=PAPER.parent/"local_papers"/"Zhang2024.pdf")
    parser.add_argument("--si",type=Path,default=PAPER.parent/"local_papers"/"Zhang2024_SI.pdf")
    args = parser.parse_args()
    extract(args.pdf,args.si)
