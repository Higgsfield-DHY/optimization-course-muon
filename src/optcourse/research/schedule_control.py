"""Reposition the two equal-length late-stage depth choices after the primary replay."""

import json
from pathlib import Path

from .assessment import evaluate_checkpoint
from .data import TokenDataset
from .protocol import ledger, run_one
from .training import atomic_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def schedule_control(root):
    root = Path(root)
    manifest_path = root / "late_replay_manifest.json"
    core = read(root / "formal_manifest.json")
    references = {row["seed"]: row for row in core
                  if row["cohort"] == "main" and row["method"] == "adaptive" and row["status"] == "COMPLETED"}
    if set(references) != {0, 1, 2}:
        raise ValueError("Late replay requires the three matched Probe-NS runs")
    test = {row["id"]: row for row in read(root / "test_manifest.json")}
    if any(row["status"] != "COMPLETED" for row in test.values()):
        raise ValueError("Complete the registered test set before the late replay")

    budget = ledger(root)
    dataset = TokenDataset(Path("data/research_v2/tokens/wikitext103"))
    entries = []
    for seed, reference in sorted(references.items()):
        base_dir = root / "language" / reference["id"]
        selections = read(base_dir / "selections.json")
        points = [row["step"] for row in selections]
        if points != list(range(100,1001,100)):
            raise ValueError("Late replay assumes the 100-update selection cadence")
        source = [int(row["choice"][2:]) for row in selections]
        schedule = source.copy()
        schedule[7], schedule[8] = schedule[8], schedule[7]
        unchanged = schedule == source
        run_id = f"late_replay_wikitext103_small_s{seed}"
        entry = dict(seed=seed, id=run_id, reference=reference["id"], source_schedule=source, replay_schedule=schedule,
                     changed_updates=0 if unchanged else 200, reference_test_loss=test[reference["id"]]["loss"],
                     status="SAME_SCHEDULE" if unchanged else "NOT_RUN")
        entries.append(entry)
        atomic_json(manifest_path, entries)
        if unchanged:
            continue
        budget.state = budget._read()
        # Measured primary replay cost, with room for held-out assessment.
        if budget.remaining_seconds < 1.25*reference["metadata"]["elapsed_seconds"] + 60:
            entry["status"]="GPU_BUDGET_LIMIT"
            atomic_json(manifest_path,entries)
            continue
        config = reference["metadata"]["config"] | dict(method="replay",cohort="late_replay",depth_schedule=schedule)
        config.pop("save_anchors",None)
        config.pop("anchor_steps",None)
        result = run_one(root,run_id,config,dataset)
        entry.update(status=result["status"], metadata=result)
        atomic_json(manifest_path,entries)
        if result["status"] != "COMPLETED":
            continue
        evaluated = evaluate_checkpoint(root/"language"/run_id,dataset,result,budget)
        entry.update(test=evaluated, test_delta=evaluated["loss"]-entry["reference_test_loss"],
                     validation_delta=result["best_validation"]-reference["metadata"]["best_validation"])
        atomic_json(manifest_path,entries)
    return entries
