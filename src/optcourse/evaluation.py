"""Deterministic held-out evaluation; no training or network operations."""

import hashlib
import json
import math
from pathlib import Path
from contextlib import nullcontext

import torch
from torch.nn import functional as F

from .config import fingerprint
from .data import ByteDataset
from .model import TinyTransformer
from .tracking import load_checkpoint, write_json


@torch.inference_mode()
def evaluate_split(model, dataset, split, block_size, batch_size, device, budget=None):
    if split not in {'val', 'test'} or block_size < 1 or batch_size < 1:
        raise ValueError('Invalid held-out evaluation settings')
    data = dataset.parts[split]
    if len(data) < 2:
        raise ValueError('Split has no predictable target bytes')
    was_training = model.training
    model.eval()
    total_loss, total_targets = 0., 0

    def consume(windows):
        nonlocal total_loss, total_targets
        with budget.operation('evaluation_batch') if budget is not None else nullcontext():
            rows = torch.stack(windows).long().to(device)
            logits = model(rows[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, 256), rows[:, 1:].reshape(-1), reduction='sum')
            if not torch.isfinite(loss).item():
                raise RuntimeError('Nonfinite evaluation loss')
            total_loss += loss.item()
            total_targets += rows[:, 1:].numel()

    try:
        pending = []
        for start in range(0, len(data)-1, block_size):
            window = data[start:min(start+block_size+1, len(data))]
            if pending and len(window) != len(pending[0]):
                consume(pending)
                pending = []
            pending.append(window)
            if len(pending) == batch_size:
                consume(pending)
                pending = []
        if pending:
            consume(pending)
        if budget is not None:
            budget.check()
    finally:
        model.train(was_training)
    loss = total_loss / total_targets
    return dict(loss=loss, byte_perplexity=math.exp(loss), predicted_bytes=total_targets,
                loss_unit='nats_per_byte', split=split,
                context_policy='nonoverlapping_blocks_reset_context')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024*1024):
            digest.update(block)
    return digest.hexdigest()


def evaluate_run(run, split='test', checkpoint='best', budget=None):
    run = Path(run)
    if split not in {'val', 'test'} or checkpoint not in {'best', 'latest'}:
        raise ValueError('Invalid evaluation split/checkpoint')
    metadata = json.loads((run/'metadata.json').read_text(encoding='utf-8'))
    config = metadata['config']
    from .budget import require_study_accounting
    owner=require_study_accounting(config,budget,run)
    if metadata['status'] != 'COMPLETED' or config['run']['task'] != 'language':
        raise ValueError('Evaluation requires a completed language run')
    if metadata['config_fingerprint'] != fingerprint(config):
        raise ValueError('Run configuration identity mismatch')
    dataset = ByteDataset(config['language']['data_dir'])
    payload = load_checkpoint(run/f'{checkpoint}.pt', config)
    if dataset.fingerprint != metadata.get('prepared_dataset_fingerprint') or dataset.fingerprint != payload.get('prepared_dataset_fingerprint'):
        raise ValueError('Checkpoint/run dataset identity mismatch')
    identity = dict(checkpoint_sha256=file_sha256(run/f'{checkpoint}.pt'),
                    config_fingerprint=fingerprint(config), prepared_dataset_fingerprint=dataset.fingerprint)
    destination = run/'evaluations'/f'{split}_{checkpoint}.json'
    def record_receipt(result):
        if split=='test':
            metadata.update(test_set_evaluated=True,test_evaluation_checkpoint=checkpoint,
                            test_evaluation_checkpoint_sha256=identity['checkpoint_sha256'])
            write_json(run/'metadata.json',metadata)
        return result
    if destination.exists():
        existing = json.loads(destination.read_text(encoding='utf-8'))
        if any(existing.get(key) != value for key, value in identity.items()):
            raise ValueError('Cached evaluation identity mismatch')
        if owner is not None:
            budget.validate_provenance(existing.get('accounting'))
        return record_receipt(existing)
    device = torch.device(config['run']['device'])
    model = TinyTransformer(config['language']).to(device)
    model.load_state_dict(payload['model'])
    result = evaluate_split(model, dataset, split, config['language']['block_size'],
                            config['language']['batch_size'], device, budget)
    result.update(identity, step=payload['step'], checkpoint=checkpoint)
    if owner is not None:
        result['accounting']=budget.provenance()
    destination.parent.mkdir(exist_ok=True)
    write_json(destination, result)
    return record_receipt(result)
