"""Check the retained protein figures/data; implementation lives with case 4."""
from pathlib import Path
import runpy,sys
folder=Path(__file__).resolve().parent/'case_studies/04_protein_crystallization'
sys.path.insert(0,str(folder))
if __name__=='__main__': runpy.run_path(str(folder/'check.py'),run_name='__main__')
