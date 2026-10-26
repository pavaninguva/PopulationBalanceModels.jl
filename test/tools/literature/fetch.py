"""Download pinned public CADET verification data, without executing upstream code."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

REVISION = "090606762fc657510a9e31c547890bf9a17ffb59"
BASE = f"https://raw.githubusercontent.com/cadet/CADET-Verification/{REVISION}/"
FILES = {
    "reference.h5": "data/CADET-Core_reference/crystallization/ref_DPFR_PBM_primarySecondaryNucleationAndGrowth.h5",
    "settings_crystallization.py": "src/benchmark_models/settings_crystallization.py",
    "LICENSE": "LICENSE",
}

def main():
    folder = Path(__file__).resolve().parents[3]/"paper/precipitation/data/literature/cadet"
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for name, source in FILES.items():
        url = BASE + source
        data = urlopen(Request(url, headers={"User-Agent": "PopulationBalanceModels-paper"}), timeout=60).read()
        (folder / name).write_bytes(data)
        records.append(dict(file=name, url=url, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
        print(name, len(data), flush=True)
    (folder / "download.json").write_text(json.dumps(dict(repository="cadet/CADET-Verification", revision=REVISION, files=records), indent=2)+"\n")

if __name__ == "__main__":
    main()
