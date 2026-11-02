"""Regenerate the report figures from an existing alignment result folder."""
import argparse,os,subprocess,sys
from pathlib import Path

def main():
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=root/'registered_output')
    p.add_argument('--project',type=Path,default=root.parent)
    p.add_argument('--output',type=Path,default=root/'comparison_figures')
    args=p.parse_args()
    if not (args.results/'manifest.json').is_file():p.error('Results folder must contain manifest.json')
    if not (args.project/'cutout-CDS_P_DSS2_color.jpg').is_file():p.error('Project folder must contain the original sky image')
    args.output.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env.update(TELESCOPE_RESULTS_DIR=str(args.results.resolve()),TELESCOPE_FIGURE_DIR=str(args.output.resolve()),TELESCOPE_PROJECT_DIR=str(args.project.resolve()))
    for name in ['make_direct_overlay.py','inspect_local_details.py','make_local_comparison.py','finalize_comparison.py']:
        print('Generating figures:',name,flush=True)
        subprocess.run([sys.executable,str(root/'figure_tools'/name)],env=env,check=True)
    print('Report comparison:',args.output/'local_details/Local_Regions_1_and_2_Alignment_Comparison.png')

if __name__=='__main__':main()
