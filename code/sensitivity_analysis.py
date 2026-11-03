"""Run the two-capability experiments, figures and synchronized reports."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
if (ROOT / '.runtime').is_dir():
    sys.path.insert(0, str(ROOT / '.runtime'))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-only', action='store_true')
    args = parser.parse_args()
    from experiments import run_experiments
    out, summary = run_experiments()
    if not args.data_only:
        from plot_figures import draw_all
        draw_all(out, summary)
        from write_report import write_reports
        write_reports(summary)
    print('All requested numerical checks passed.', flush=True)

if __name__ == '__main__':
    main()
