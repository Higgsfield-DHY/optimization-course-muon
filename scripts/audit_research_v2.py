"""Check the registered run, checkpoint, paired-control and result identities."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from optcourse.research.analysis import assemble, read
from optcourse.research.assessment import checked_checkpoint
from optcourse.research.data import TokenDataset, sha256
from optcourse.research.training import compatible_source


def check_candidate_amplitudes(anchor, collect_spectra=False):
    state = torch.load(anchor, map_location="cpu", weights_only=True)
    config = state["config"]
    rate = config["lr"] * state["lr_factor"]
    hidden = set(state["hidden_names"])
    base = state["model"]
    candidates = state["candidates"]
    auxiliary = set(candidates["ns1"]) - hidden
    maximum = 0.0
    spectra = []
    depth = config["model"]["depth"]
    layers = {
        "blocks.0.attention.qkv.weight": "QKV",
        f"blocks.{depth//2}.mlp.0.weight": "MLP expansion",
        f"blocks.{depth-1}.attention.projection.weight": "Attention output",
        f"blocks.{depth-1}.mlp.2.weight": "MLP contraction",
    }
    for name in auxiliary:
        x = candidates["ns1"][name]
        assert torch.equal(x, candidates["ns3"][name]) and torch.equal(x, candidates["ns5"][name])
    for name in hidden:
        common_decay = base[name] * config["weight_decay"]
        has_momentum = bool(state["momenta"][name].count_nonzero())
        expected = 0.2 if has_momentum else 0.0
        for variant,values in candidates.items():
            direction = values[name].double() / rate - common_decay.double()
            rms = direction.square().mean().sqrt().item()
            maximum = max(maximum, abs(rms-expected))
            if collect_spectra and name in layers:
                singular=torch.linalg.svdvals(direction)
                stable=(singular.square().sum()/singular[0].square()).item() if singular[0]>0 else 0.0
                spectra.append(dict(run=anchor.parent.parent.name,seed=config["seed"],corpus=config["corpus"],
                    scale=config["scale"],cohort=config["cohort"],step=state["step"],layer=layers[name],variant=variant,
                    stable_rank=stable,stable_rank_fraction=stable/min(direction.shape)))
    assert maximum < 2e-6
    return (maximum,spectra) if collect_spectra else maximum


def audit(root, verify_checkpoints=True):
    data = assemble(root)
    formal = read(root / "formal_manifest.json")
    expected = read(root / "frozen_protocol.json")["expected_jobs"]
    key = lambda row: tuple(row[name] for name in ("cohort", "corpus", "scale", "method", "seed", "steps"))
    assert Counter(map(key, formal)) == Counter(map(key, expected))
    assert all(row["status"] == "COMPLETED" for row in formal)
    assert all(row["metadata"]["completed_steps"] == row["steps"] for row in formal)
    assert all(compatible_source(row["metadata"].get("source_hashes")) for row in formal)

    ledger = data["gpu_budget"]
    assert ledger["active"] is None
    assert math.isclose(sum(row["seconds"] for row in ledger["attempts"]), ledger["used_seconds"], rel_tol=1e-12)
    assert ledger["used_seconds"] <= ledger["limit_seconds"]
    charged = {item["session_id"] for item in ledger["attempts"] if item["outcome"] == "CLOSED"}

    datasets = {name: TokenDataset(ROOT / "data/research_v2/tokens" / name, verify=verify_checkpoints)
                for name in ("wikitext103", "tinystories")} if verify_checkpoints else {}
    test = {row["id"]: row for row in read(root / "test_manifest.json")}
    assert len(test) == len(formal) and all(row["status"] == "COMPLETED" for row in test.values())
    paired_controls = defaultdict(list)
    initial = defaultdict(list)
    for row in formal:
        identifier = row["id"]
        group = row["metadata"]["config"]
        result = test[identifier]
        assert result["dataset_fingerprint"] == row["metadata"]["dataset_fingerprint"]
        assert result["config_fingerprint"] == row["metadata"]["config_fingerprint"]
        assert result["budget_provenance"]["session_id"] in charged
        assert row["metadata"]["budget_provenance"]["session_id"] in charged
        assert math.isfinite(result["loss"])
        run = root / "language" / identifier
        metrics = read(run / "metrics.json")
        val = [point for point in metrics if point["validation_loss"] is not None]
        best = min(val, key=lambda point:(point["validation_loss"],point["step"]))
        assert best["step"] == result["best_step"]
        assert best["validation_loss"] == result["best_validation"]
        assert group["batch_size"] * group["accumulation"] * group["model"]["context"] == 8192
        assert group["aux_lr"] == 0.001
        assert val[-1]["tokens"] == row["steps"] * 8192
        assert result["targets"] == {"wikitext103": 287644, "tinystories": 2369481}[row["corpus"]]
        if verify_checkpoints:
            _, receipt = checked_checkpoint(run, datasets[row["corpus"]], row["metadata"])
            assert all(result[field] == value for field, value in receipt.items())
        if row["cohort"] in ("main", "criterion", "ablation"):
            initial[row["seed"]].append(val[0]["validation_loss"])
        if row["cohort"] in ("main", "criterion") and row["method"] in ("ns1","ns3","ns5","adaptive","replay","spectral"):
            paired_controls[row["seed"]].append((group["lr"],group["aux_lr"],group["weight_decay"],group["warmup_steps"],
                                                group["steps"],row["metadata"]["dataset_fingerprint"]))
    for values in paired_controls.values():
        assert len(values)==6 and len(set(values))==1
    for values in initial.values():
        assert len(values)==10 and max(values)-min(values)<1e-5

    assert data["probes"]["completed_anchors"] == data["probes"]["expected_anchors"] == 36
    probe_manifest=read(root/"probe_manifest.json")
    amplitude_deviation=[]
    spectra=[]
    torch.set_num_threads(2)
    for identifier, probe in probe_manifest.items():
        assert probe["status"]=="COMPLETED" and probe["rows"]==30
        receipt=read(root/"probes"/identifier/"result.json")
        run_id=receipt["run_id"]
        anchor=root/"language"/run_id/"anchors"/f"step-{receipt['anchor_step']}.pt"
        if verify_checkpoints:
            assert receipt["source_anchor_sha256"]==sha256(anchor)
            deviation,rows=check_candidate_amplitudes(anchor,collect_spectra=True)
            amplitude_deviation.append(deviation)
            spectra.extend(rows)
        for row in receipt["rows"]:
            assert row["b_batch_sha256"]!=row["c_batch_sha256"]
            assert math.isfinite(row["actual_gain"])
    assert len(data["training"]["replay"])==3
    assert data["synthetic"]["boundary_errors"]==0
    assert data["synthetic"]["swap_reversals"]==data["synthetic"]["swap_pairs"]==30
    assert set(data["profile"])=={"small","medium"}
    followup = data["late_replay"]
    for row in followup:
        assert Counter(row["source_schedule"])==Counter(row["replay_schedule"])
        assert row["source_schedule"][:7]==row["replay_schedule"][:7]
        assert row["source_schedule"][9:]==row["replay_schedule"][9:]
        if row["status"]=="COMPLETED":
            assert row["changed_updates"]==200
            assert row["test"]["targets"]==287644
            assert row["test"]["budget_provenance"]["session_id"] in charged
            if verify_checkpoints:
                _,receipt=checked_checkpoint(root/"language"/row["id"],datasets["wikitext103"],row["metadata"])
                assert all(row["test"][field]==value for field,value in receipt.items())
    if verify_checkpoints:
        assert len(spectra)==36*4*3
        (root/"update_spectra.json").write_text(json.dumps(spectra,indent=2),encoding="utf-8")
    return dict(formal_runs=len(formal),test_runs=len(test),frozen_anchors=len(probe_manifest),
                same_momentum_reversals=data["synthetic"]["swap_reversals"],paired_depth_controls=len(paired_controls),
                gpu_seconds=ledger["used_seconds"],checkpoints_verified=verify_checkpoints,
                maximum_hidden_rms_deviation=max(amplitude_deviation) if amplitude_deviation else None,
                late_schedule_runs=sum(row["status"]=="COMPLETED" for row in followup))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=ROOT/"runs/research_v2")
    parser.add_argument("--archived-metrics",action="store_true")
    args=parser.parse_args()
    print(json.dumps(audit(args.root,not args.archived_metrics),indent=2))
