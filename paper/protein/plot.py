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
    parser.add_argument('--input',type=Path,help='Numerical result root, containing shared/, cnt/ and cnt_grd/')
    parser.add_argument('--output',type=Path,help='Figure directory; numerical exports are never modified')
    analysis = parser.add_mutually_exclusive_group()
    analysis.add_argument('--gauss-newton', action='store_true')
    analysis.add_argument('--laplace', action='store_true')
    args = parser.parse_args()
    if args.input is not None:
        protein_plot.BASE=args.input.resolve()
        protein_plot.TABLES=protein_plot.BASE/'analysis/tables'
    if args.output is not None:
        protein_plot.OUT=args.output.resolve()
    if args.gauss_newton or args.laplace:
        if args.input is not None or args.output is not None:
            parser.error('Use retained active results for the separate uncertainty analyses')
        if args.data_only:
            parser.error('Choose data-only figures or an uncertainty comparison')
        protein_uncertainty.main(['--approximation', 'gauss_newton' if args.gauss_newton else 'laplace']
                                + (['--tables'] if args.tables else []))
    else:
        protein_plot.plot(data_only=args.data_only, figures_only=not args.tables)
