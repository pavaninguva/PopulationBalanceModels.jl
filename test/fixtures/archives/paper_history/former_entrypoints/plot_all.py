"""Convenience dispatcher; each case also has an independently runnable plot.py."""
import argparse
from pathlib import Path
import subprocess
import sys

PAPER=Path(__file__).resolve().parent
CASES=('01_growth_time_methods','02_aggregation_fragmentation',
       '03_tubular_precipitation','04_protein_crystallization')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('all','1','2','3','4'),default='all')
    parser.add_argument('--retained',action='store_true',help='Cases 1–3: redraw validated exports without asserting current producer equality')
    parser.add_argument('--smoke',action='store_true',help='Ignored quick-check outputs, never manuscript data')
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--data-only',action='store_true')
    parser.add_argument('--figures-only',action='store_true',help='Compatibility flag: plotting is read-only by default')
    parser.add_argument('--tables',action='store_true',help='Case 4: export derived tables to analysis/')
    comparison=parser.add_mutually_exclusive_group()
    comparison.add_argument('--laplace-comparison',action='store_true')
    comparison.add_argument('--gauss-newton-comparison',action='store_true')
    args=parser.parse_args()
    protein_only=args.data_only or args.tables or args.laplace_comparison or args.gauss_newton_comparison
    if protein_only and args.case!='4': parser.error('Protein-only flags require --case 4')
    failed=[]
    for number,case in enumerate(CASES,1):
        if args.case not in ('all',str(number)): continue
        folder=PAPER/'case_studies'/case
        options=[];script=folder/'plot.py'
        if number<4:
            options=[flag for enabled,flag in ((args.retained,'--retained'),(args.smoke,'--smoke'),(args.preview,'--preview')) if enabled]
        else:
            if args.smoke or args.preview:
                print('Protein quick checks do not generate calibrated paper figures.')
                continue
            if args.laplace_comparison or args.gauss_newton_comparison:
                script=folder/'uncertainty.py'
                options=['--approximation','gauss_newton' if args.gauss_newton_comparison else 'laplace']
            else:
                options=[flag for enabled,flag in ((args.data_only,'--data-only'),(args.tables,'--tables')) if enabled]
        result=subprocess.run([sys.executable,str(script),*options])
        if result.returncode: failed.append(case)
    if failed: parser.exit(1,'Incomplete figure cases: '+', '.join(failed)+'\n')


if __name__=='__main__': main()
