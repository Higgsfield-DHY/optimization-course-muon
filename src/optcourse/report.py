"""Summarize completed, distinct-seed runs without selecting a test-set winner."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

from .config import fingerprint
from .tracking import METRICS, write_json


def summarize(runs_root, output_root, make_plots=True):
    grouped = defaultdict(list)
    status_counts, excluded = Counter(), []
    for path in sorted(Path(runs_root).glob("*/metadata.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        status = meta.get("status", "UNKNOWN")
        status_counts[status] += 1
        if meta.get("status") != "COMPLETED":
            excluded.append(dict(path=str(path.parent.resolve()), status=status,
                                 seed=meta.get("config", {}).get("run", {}).get("seed"),
                                 failure_reason=meta.get("failure_reason")))
            continue
        with (path.parent / "metrics.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"Completed run has no metrics: {path.parent}")
        prepared_identity = meta.get("prepared_dataset_fingerprint")
        if meta["config"]["run"]["task"] == "language" and not prepared_identity:
            raise ValueError(f"Completed language run lacks prepared dataset identity: {path.parent}")
        identity = fingerprint(meta["config"], ignore_seed=True) + str(prepared_identity)
        key = hashlib.sha256(identity.encode()).hexdigest()[:12]
        grouped[key].append((meta, rows, path.parent))
    if not grouped:
        raise ValueError(f"No completed experiment runs found; statuses={dict(status_counts)}; no report created")
    groups = []
    for key, items in grouped.items():
        seeds = [item[0]["config"]["run"]["seed"] for item in items]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seed in group {key}; select distinct experiment runs")
        group = dict(group=key, name=items[0][0]["config"]["run"]["name"],
                     task=items[0][0]["config"]["run"]["task"], n=len(items))
        for metric in METRICS:
            if metric == "step":
                continue
            values = []
            for _, rows, _ in items:
                available = [float(row[metric]) for row in rows if row.get(metric)]
                if available:
                    values.append(available[-1])
            group[metric + "_mean"] = statistics.mean(values) if len(values) == len(items) else None
            group[metric + "_std"] = statistics.stdev(values) if len(values) > 1 and len(values) == len(items) else None
        groups.append(group)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)
    if make_plots:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        for key, items in grouped.items():
            metric = "objective" if items[0][0]["config"]["run"]["task"] == "matrix" else "validation_loss"
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
            for meta, rows, _ in items:
                useful = [row for row in rows if row.get(metric)]
                for axis, x in zip(axes, ("step", "elapsed_seconds")):
                    axis.plot([float(row[x]) for row in useful], [float(row[metric]) for row in useful],
                              label=f"seed {meta['config']['run']['seed']}")
                    axis.set(xlabel=x, ylabel=metric)
                    axis.grid(alpha=.2)
                    axis.legend()
            fig.savefig(output_root / f"{key}.png", dpi=180)
            fig.savefig(output_root / f"{key}.pdf")
            plt.close(fig)
    result = dict(completed_runs=sum(len(items) for items in grouped.values()), groups=groups,
                  discovered_runs=sum(status_counts.values()), status_counts=dict(status_counts),
                  excluded_count=len(excluded), excluded_runs=excluded,
                  note="Distinct seeds only; sample standard deviation is unavailable for n=1; no test-set selection")
    with (output_root / "excluded_runs.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "status", "seed", "failure_reason"])
        writer.writeheader()
        writer.writerows(excluded)
    write_json(output_root / "summary.json", result)
    return result
