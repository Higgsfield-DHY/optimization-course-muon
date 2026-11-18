"""No training or downloads happen unless explicitly requested by a command."""

import argparse
import json
from pathlib import Path
import sys

from .config import load_config, public_config


def plan(config, execution_requested=False):
    task = config["run"]["task"]
    return dict(status="NOT_RUN", execution_requested=execution_requested,
                steps=config[task]["steps" if task == "matrix" else "max_steps"],
                data_prepared=Path(config["language"]["data_dir"], "manifest.json").is_file()
                    if task == "language" else None,
                config=public_config(config))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Muon course experiments; commands are dry by default")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True)
        sub.add_argument("--seed", type=int)
        if name == "run":
            sub.add_argument("--execute", action="store_true")
            sub.add_argument("--resume", type=Path)
            sub.add_argument("--budget-seconds",type=float)
            sub.add_argument("--budget-ledger",type=Path)
    batch = commands.add_parser("batch")
    batch.add_argument("--configs", nargs="+", required=True)
    batch.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    batch.add_argument("--execute", action="store_true")
    batch.add_argument("--budget-seconds",type=float)
    batch.add_argument("--budget-ledger",type=Path)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--output", type=Path)
    data = commands.add_parser("prepare-data")
    source = data.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--download-shakespeare", action="store_true")
    data.add_argument("--output", required=True, type=Path)
    report = commands.add_parser("summarize")
    report.add_argument("--runs", type=Path, default=Path("runs"))
    report.add_argument("--output", type=Path, default=Path("reports"))
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--run", type=Path, required=True)
    evaluation.add_argument("--split", choices=("val", "test"), default="test")
    evaluation.add_argument("--checkpoint", choices=("best", "latest"), default="best")
    evaluation.add_argument('--execute',action='store_true')
    evaluation.add_argument('--budget-seconds',type=float,default=28800)
    evaluation.add_argument('--budget-ledger',type=Path)
    study = commands.add_parser('study')
    study.add_argument('--stage',choices=('all','matrix','pilot','tuning','final','evaluate','report'),default='all')
    study.add_argument('--root',type=Path,default=Path('runs/course_v1'))
    study.add_argument('--execute',action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'study':
            from .study import study_plan, execute_study
            result = execute_study(args.stage,args.root) if args.execute else study_plan()
        elif args.command == "doctor":
            from .tracking import environment_info
            result = environment_info()
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        elif args.command == "prepare-data":
            from .data import prepare_data, download_shakespeare, SHAKESPEARE_URL
            input_path = args.input or download_shakespeare(args.output.parent / "raw")
            result = prepare_data(input_path, args.output,
                                  source_url=SHAKESPEARE_URL if args.download_shakespeare else None)
        elif args.command == "summarize":
            from .report import summarize
            result = summarize(args.runs, args.output)
        elif args.command == "evaluate":
            from .evaluation import evaluate_run
            if not args.execute:
                result=dict(status='NOT_RUN',run=str(args.run),split=args.split,checkpoint=args.checkpoint)
            else:
                from .budget import BudgetLedger,owning_study_ledger
                metadata=json.loads((args.run/'metadata.json').read_text(encoding='utf-8'))
                owner=owning_study_ledger(metadata['config'],args.run)
                if owner is not None and args.budget_seconds!=28800:
                    raise ValueError('Course GPU budget is fixed at 28800 seconds')
                if owner is not None and args.budget_ledger is not None and args.budget_ledger.resolve()!=owner.resolve():
                    raise ValueError('Evaluation must use its owning study ledger')
                ledger=BudgetLedger(owner or args.budget_ledger or Path('runs/gpu_budget.json'),args.budget_seconds)
                with ledger.session('evaluate_'+str(args.run)):
                    result=evaluate_run(args.run,args.split,args.checkpoint,budget=ledger)
        else:
            if args.command == "batch":
                jobs = [load_config(path, seed) for path in args.configs for seed in args.seeds]
            else:
                jobs = [load_config(args.config, args.seed)]
            if args.command == "plan" or not getattr(args, "execute", False):
                result = [plan(config) for config in jobs]
                if args.command != "batch":
                    result = result[0]
            else:
                from .experiments import run_experiment
                from .budget import BudgetLedger,owning_study_ledger
                result = []
                for config in jobs:
                    owner=owning_study_ledger(config,getattr(args,'resume',None))
                    if owner is not None and args.budget_seconds not in (None,28800):
                        raise ValueError('Course GPU budget is fixed at 28800 seconds')
                    if owner is not None and args.budget_ledger is not None and args.budget_ledger.resolve()!=owner.resolve():
                        raise ValueError('Course run must use its owning study ledger')
                    seconds=args.budget_seconds if args.budget_seconds is not None else (28800 if owner else None)
                    ledger=BudgetLedger(owner or args.budget_ledger or Path('runs/gpu_budget.json'),seconds) if seconds is not None else None
                    if ledger is not None:
                        if ledger.remaining_seconds<=0:
                            break
                        with ledger.session(config['run']['name']):
                            result.append(str(run_experiment(config,getattr(args,'resume',None),budget=ledger)))
                    else:
                        result.append(str(run_experiment(config,getattr(args,'resume',None))))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
