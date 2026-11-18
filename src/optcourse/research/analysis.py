"""Aggregate intervention and training outcomes by independent training seed."""

from collections import Counter, defaultdict
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from .report import read_csv


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def distribution(values):
    values = list(values)
    return {
        "n": len(values),
        "mean": mean(values) if values else None,
        "sd": stdev(values) if len(values) > 1 else None,
        "values": values,
    }


def verify_geometry(root):
    from .geometry import two_mode_boundary

    ext = root / "synthetic_extended"
    phase = defaultdict(dict)
    for row in read_csv(ext / "phase_fixed_momentum.csv"):
        phase[(float(row["gradient_ratio"]), float(row["curvature_ratio"]))][row["variant"]] = row
    errors = 0
    for (rho, kappa), rows in phase.items():
        r, s = (float(rows[k]["direction_ratio"]) for k in ("ns5", "ns1"))
        actual = float(rows["ns5"]["optimal_gain"]) - float(rows["ns1"]["optimal_gain"])
        predicted_sign = np.sign((r-s)*(two_mode_boundary(rho, r, s)-kappa))
        errors += abs(actual) > 1e-10 and np.sign(actual) != predicted_sign

    swaps = defaultdict(dict)
    for row in read_csv(ext / "same_momentum_swaps.csv"):
        if float(row["epsilon"]) == 1e-5 and float(row["exponent"]) > 0 and row["variant"] in ("ns3", "ns5"):
            swaps[(int(row["seed"]), float(row["exponent"]))][row["favored"], row["variant"]] = float(row["optimal_gain"])
    gains = {method: [outcomes[method, method]/outcomes[method, other]
                      for outcomes in swaps.values()] for method, other in (("ns3", "ns5"), ("ns5", "ns3"))}
    reversed_count = sum(all(outcomes[k,k] > outcomes[k,j] for k,j in (("ns3","ns5"),("ns5","ns3")))
                         for outcomes in swaps.values())

    trajectories = defaultdict(dict)
    for row in read_csv(root / "synthetic/final.csv"):
        condition, orientation, noise, seed = (row[key] for key in ("condition","orientation","noise_ratio","seed"))
        trajectories[float(condition), orientation, float(noise)][(row["variant"], int(seed))] = float(row["objective"])
    trajectory_rows = []
    for (condition, orientation, noise), scores in sorted(trajectories.items()):
        methods = {key: distribution(scores[key,seed] for seed in range(10))
                   for key in ("ns0","ns1","ns3","ns5","polar")}
        trajectory_rows.append({"condition": condition, "orientation": orientation, "noise": noise,
                                "methods": methods, "winner": min(methods, key=lambda k: methods[k]["mean"])})
    return {
        "phase_states": len(phase), "boundary_errors": int(errors),
        "error_by_spectrum": read(ext / "statistics.json"),
        "swap_pairs": len(swaps), "swap_reversals": reversed_count,
        "swap_gain_ratio": {key: {"median": float(np.median(value)), "min": min(value), "max": max(value)}
                            for key, value in gains.items()},
        "trajectories": trajectory_rows,
    }


def replay_changes(root, formal):
    runs = {(row["method"], row["seed"]): row for row in formal
            if row["cohort"] == "main" and row["status"] == "COMPLETED"}
    result = []
    for seed in range(3):
        if not all((method,seed) in runs for method in ("adaptive","replay")):
            continue
        adaptive, replay = [read(root / "language" / runs[method,seed]["id"] / "selections.json")
                            for method in ("adaptive","replay")]
        if [row["step"] for row in adaptive] != [row["step"] for row in replay]:
            raise ValueError(f"Selection cadence differs for seed {seed}")
        steps = [row["step"] for row in adaptive]
        end = runs["adaptive",seed]["steps"]
        weights = np.diff(steps+[end+1])
        a, r = [[int(row["choice"][2:]) for row in run] for run in (adaptive,replay)]
        for block in np.array_split(np.arange(len(steps)),4):
            left = Counter()
            right = Counter()
            for i in block:
                left[a[i]] += int(weights[i])
                right[r[i]] += int(weights[i])
            if left != right:
                raise ValueError(f"Replay did not preserve realized frequencies for seed {seed}")
        changed = [i for i,(x,y) in enumerate(zip(a,r)) if x!=y]
        result.append(dict(seed=seed,steps=steps,adaptive=a,replay=r,changed_decisions=len(changed),
                           changed_updates=sum(int(weights[i]) for i in changed),
                           first_changed_step=steps[changed[0]] if changed else None))
    return result


def training_results(root):
    formal = read(root / "formal_manifest.json")
    test_path = root / "test_manifest.json"
    tested = {row["id"]: row for row in read(test_path) if row["status"] == "COMPLETED"} if test_path.is_file() else {}
    members = defaultdict(list)
    curves = defaultdict(dict)
    policy = defaultdict(list)
    for row in formal:
        cohort, corpus, scale, method = (row[key] for key in ("cohort","corpus","scale","method"))
        key = f"{cohort}/{corpus}/{scale}"
        sample = dict(seed=row["seed"], status=row["status"])
        if row["status"] == "COMPLETED":
            run = root / "language" / row["id"]
            config = row["metadata"]["config"]
            metrics = read(run / "metrics.json")
            tail = next(item for item in metrics if item["step"] == row["steps"] and item["validation_loss"] is not None)
            sample.update(
                id=row["id"], lr=row["lr"], aux_lr=config["aux_lr"], terminal_validation=tail["validation_loss"],
                best_step=row["metadata"]["best_step"], best_validation=row["metadata"]["best_validation"],
                test_loss=tested.get(row["id"],{}).get("loss"), elapsed_seconds=row["metadata"]["elapsed_seconds"],
                peak_memory_mib=row["metadata"]["peak_memory_MiB"],
            )
            curves[key].setdefault(method,{})[row["seed"]] = [item for item in metrics if item["validation_loss"] is not None]
            if method in ("adaptive","replay","spectral"):
                selected = read(run / "selections.json")
                choices = [int(item["choice"][2:]) for item in selected]
                points = [item["step"] for item in selected]
                exposure = Counter({5:points[0]-1})
                for choice, duration in zip(choices,np.diff(points+[row["steps"]+1])):
                    exposure[choice]+=int(duration)
                policy[key,method].append(dict(seed=row["seed"],decisions=dict(Counter(choices)),
                    exposure=dict(exposure),probe_seconds=sum(item["probe_seconds"] for item in selected)))
        members[key,method].append(sample)
    groups = defaultdict(dict)
    for (key,method),rows in members.items():
        completed = [row for row in rows if row["status"] == "COMPLETED"]
        result = {"expected": len(rows), "completed": len(completed), "seeds": sorted(rows,key=lambda row:row["seed"])}
        for metric in ("terminal_validation","best_validation","test_loss","elapsed_seconds","peak_memory_mib"):
            result[metric] = distribution(row[metric] for row in completed if row[metric] is not None)
        groups[key][method] = result

    contrasts = {}
    for key, group in groups.items():
        if key.startswith("main/") and "criterion/wikitext103/small" in groups:
            group = group | groups["criterion/wikitext103/small"]
        records = {}
        for left,right in (("ns5","adamw"),("ns3","ns5"),("adaptive","ns5"),("adaptive","replay"),
                           ("adaptive","spectral"),("adam_rms","canonical5")):
            if left not in group or right not in group:
                continue
            paired = [{row["seed"]:row for row in group[name]["seeds"] if row["status"] == "COMPLETED"}
                      for name in (left,right)]
            metrics = {}
            for metric in ("terminal_validation","test_loss"):
                seeds = sorted(s for s in paired[0].keys() & paired[1].keys()
                               if paired[0][s][metric] is not None and paired[1][s][metric] is not None)
                metrics[metric] = distribution(paired[0][s][metric]-paired[1][s][metric] for s in seeds) | {"seeds": seeds}
            records[f"{left}-{right}"] = metrics
        contrasts[key] = records
    return dict(groups=dict(groups),curves=dict(curves),contrasts=contrasts,
                training_status=dict(Counter(row["status"] for row in formal)),test_completed=len(tested),
                replay=replay_changes(root,formal),
                policy={key+"/"+method:sorted(rows,key=lambda row:row["seed"]) for (key,method),rows in policy.items()})


def probe_results(root):
    formal = read(root / "formal_manifest.json")
    path = root / "probe_manifest.json"
    manifest = read(path) if path.is_file() else {}
    expected = {
        f"{row['id']}_step-{step}"
        for row in formal if row["status"]=="COMPLETED" and row["metadata"]["config"].get("save_anchors")
        for step in row["metadata"]["config"]["anchor_steps"]
    }
    observed = sorted(expected & {key for key,row in manifest.items() if row["status"]=="COMPLETED"})
    states = defaultdict(dict)
    ref_states = defaultdict(dict)
    spectral = []
    for key in observed:
        result = read(root / "probes" / key / "result.json")
        for row in result["rows"]:
            identity = (row["cohort"],row["corpus"],row["scale"],row["seed"],row["run_id"],
                        row["anchor_step"],row["repeat"],row["multiplier"])
            ref_states[identity][row["variant"]] = row
            if row["variant"] in ("ns1","ns3","ns5"):
                states[identity][row["variant"]] = row
        for layer in result["spectral"]:
            spectral.append({"run_id":result["run_id"],"step":result["anchor_step"],**layer})

    scores = {
        "polar_error": lambda row: -row["polar_error"],
        "alignment": lambda row: row["linear_gain"],
        "quadratic": lambda row: row["predicted_gain"],
        "batch_b_step": lambda row: row["actual_b_unit_gain"],
    }
    decision_rows = []
    calibration = []
    references = []
    scale_pairs = defaultdict(dict)
    for identity, mapping in states.items():
        cohort,corpus,scale,seed,run,step,repeat,multiplier = identity
        if len(mapping) != 3:
            raise ValueError(f"Probe candidate set changed: {identity}")
        keys = ("ns1","ns3","ns5")
        outcomes = {key: mapping[key]["actual_gain"] for key in keys}
        chosen = max(keys,key=lambda key:outcomes[key])
        dimensions = dict(cohort=cohort,corpus=corpus,scale=scale,seed=seed,run=run,step=step,
                          repeat=repeat,multiplier=multiplier)
        for left,right in combinations(keys,2):
            x,y = mapping[left],mapping[right]
            scale_pairs[identity[:-1]+(left,right)][multiplier] = dict(
                actual=outcomes[left]-outcomes[right],
                linear=x["linear_gain"]-y["linear_gain"],
                quadratic=x["predicted_gain"]-y["predicted_gain"],
            )
        for name,score in scores.items():
            if name == "batch_b_step" and multiplier != 1.:
                continue
            predicted = {key: score(mapping[key]) for key in keys}
            pick = max(keys,key=lambda key:predicted[key])
            agreements = []
            for left,right in combinations(keys,2):
                true_diff = outcomes[left]-outcomes[right]
                pred_diff = predicted[left]-predicted[right]
                if abs(true_diff)>1e-7:
                    agreements.append(0.5 if abs(pred_diff)<=1e-10 else float(true_diff*pred_diff>0))
            decision_rows.append(dimensions | dict(predictor=name,choice=pick,winner=chosen,
                concordance=mean(agreements) if agreements else None,n_comparisons=len(agreements),
                regret=outcomes[chosen]-outcomes[pick],top_hit=float(outcomes[chosen]-outcomes[pick]<=1e-7)))
        if multiplier == 1.:
            for left,right in combinations(keys,2):
                x,y = mapping[left],mapping[right]
                linear = x["linear_gain"]-y["linear_gain"]
                quadratic = x["predicted_gain"]-y["predicted_gain"]
                local = x["actual_b_unit_gain"]-y["actual_b_unit_gain"]
                calibration.append(dimensions | dict(pair=f"{left}-{right}",linear_error=abs(linear-local),
                    quadratic_error=abs(quadratic-local),linear=linear,curvature=quadratic-linear,same_batch_gain=local,
                    cross_batch_gain=x["actual_gain"]-y["actual_gain"]))
            reference = ref_states[identity]
            observations = {key: value["actual_gain"] for key,value in reference.items()}
            references.append(dimensions | dict(gains=observations,
                best_empirical_delta=max(outcomes.values())-observations["polar"]))

    def aggregated(fields,rows,metrics):
        grouped = defaultdict(lambda: defaultdict(list))
        for row in rows:
            group = tuple(row[k] for k in fields)
            grouped[group][row["seed"]].append(row)
        result = []
        for group,seed_rows in sorted(grouped.items()):
            summary = dict(zip(fields,group)) | {"n_seeds":len(seed_rows),"n_states":sum(map(len,seed_rows.values()))}
            for metric in metrics:
                values=[]
                for seed in sorted(seed_rows):
                    eligible=[row[metric] for row in seed_rows[seed] if row[metric] is not None]
                    if eligible:values.append(mean(eligible))
                summary[metric]=distribution(values)
            result.append(summary)
        return result

    crossover = []
    for identity,steps in scale_pairs.items():
        if set(steps)!={0.5,1.,2.}:
            raise ValueError(f"Probe step multipliers changed: {identity}")
        cohort,corpus,scale,seed,run,step,repeat,left,right=identity
        x,y = steps[0.5],steps[2.]
        linear_unit=steps[1.]["linear"]
        curvature_half=linear_unit-steps[1.]["quadratic"]
        threshold=linear_unit/curvature_half if abs(curvature_half)>1e-14 else None
        crossover.append(dict(seed=seed,cohort=cohort,corpus=corpus,scale=scale,run=run,step=step,repeat=repeat,
            pair=f"{left}-{right}",linear_gain=linear_unit,curvature_half=curvature_half,
            threshold=threshold,observed_flip=x["actual"]*y["actual"]<-1e-14,
            predicted_flip=x["quadratic"]*y["quadratic"]<-1e-14))
    return dict(expected_anchors=len(expected),completed_anchors=len(observed),registered_states=len(states),
                ranking=aggregated(("multiplier","predictor"),decision_rows,("concordance","regret","top_hit")),
                by_cohort=aggregated(("cohort","corpus","scale","multiplier","predictor"),decision_rows,
                                    ("concordance","regret","top_hit")),
                calibration=aggregated((),calibration,("linear_error","quadratic_error")),
                decision_rows=decision_rows,calibration_rows=calibration,references=references,spectral=spectral,
                crossover=crossover)


def profile_results(root):
    groups = {}
    for scale in ("small","medium"):
        path = root / f"profile_{scale}.json"
        if path.exists():
            groups[scale]={}
            rows=read(path)["rows"]
            for method in dict.fromkeys(row["method"] for row in rows):
                subset=[row for row in rows if row["method"]==method]
                groups[scale][method]={name:float(np.median([row[name] for row in subset]))
                                       for name in ("model_seconds","clip_seconds","optimizer_seconds","total_seconds","peak_allocated_MiB")}
    return groups


def assemble(root):
    root = Path(root)
    schedule = root/"late_replay_manifest.json"
    update_spectra = root/"update_spectra.json"
    return dict(synthetic=verify_geometry(root),training=training_results(root),probes=probe_results(root),
                profile=profile_results(root),gpu_budget=read(root/"gpu_budget.json"),
                late_replay=read(schedule) if schedule.is_file() else [],
                update_spectra=read(update_spectra) if update_spectra.is_file() else [])
