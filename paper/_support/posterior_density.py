"""Compatibility imports for the shared, display-only KDE implementation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"python"))
from pbm_plotting.density import (marginal_density, marginal_on_grid,
    joint_density, probability_levels)
