"""Replay the paper's frozen final configurations without retuning or resizing."""
from pathlib import Path
import argparse,json,sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=ROOT/'runs/reproduction_v1')
parser.add_argument('--execute',action='store_true')
args=parser.parse_args()
evidence=json.loads((ROOT/'paper/numeric-evidence.json').read_text(encoding='utf-8'))
if not args.execute:
    print(json.dumps(dict(status='NOT_RUN',matrix_runs=45,language_runs=9,
        frozen_steps=evidence['profile']['steps'],gpu_budget_seconds=28800,root=str(args.root)),indent=2))
    raise SystemExit(0)
if args.root.exists():
    raise ValueError('Use a fresh reproduction root; resume existing stages through optcourse study --root')
from optcourse.study import Study,recipes
from optcourse.data import ByteDataset
from optcourse.comparison import compare_study
dataset=ByteDataset(ROOT/'data/shakespeare_byte')
if dataset.fingerprint!=evidence['dataset_fingerprint']:
    raise ValueError('Data does not match the frozen paper split')
study=Study(args.root)
study.state.update(profile=evidence['profile'],matrix_selection=evidence['matrix_selection'],
    language_selection=evidence['language_selection'],dataset_fingerprint=dataset.fingerprint,
    frozen_reproduction=True)
study.save()
study.run_rows(recipes('matrix','final',selected=study.state['matrix_selection']))
study.matrix_diagnostics()
study.final();study.evaluate()
result=compare_study(study.path,study.root/'reports')
print(json.dumps(dict(status=study.state['status'],completed_final_runs=result['completed_final_runs'],
    gpu_seconds=study.budget.used_seconds),indent=2))
