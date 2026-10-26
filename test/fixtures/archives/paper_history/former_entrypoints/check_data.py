"""Validate the first three paper datasets without running simulations."""
from pathlib import Path
import runpy
if __name__=='__main__':
    runpy.run_path(str(Path(__file__).resolve().parent/'support/check_data.py'),run_name='__main__')
