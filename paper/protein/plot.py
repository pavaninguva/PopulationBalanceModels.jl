"""Draw the protein paper figures from retained data; never fit or sample."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'_support'))
import protein_plot
import protein_uncertainty

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-only', action='store_true')
    parser.add_argument('--tables', action='store_true')
    analysis = parser.add_mutually_exclusive_group()
    analysis.add_argument('--gauss-newton', action='store_true')
    analysis.add_argument('--laplace', action='store_true')
    args = parser.parse_args()
    if args.gauss_newton or args.laplace:
        if args.data_only:
            parser.error('Choose data-only figures or an uncertainty comparison')
        protein_uncertainty.main(['--approximation', 'gauss_newton' if args.gauss_newton else 'laplace']
                                + (['--tables'] if args.tables else []))
    else:
        protein_plot.plot(data_only=args.data_only, figures_only=not args.tables)
