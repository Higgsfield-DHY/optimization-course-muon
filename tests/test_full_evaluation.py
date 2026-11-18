import importlib
import math

import pytest
import torch


class PositionModel(torch.nn.Module):
    def forward(self, tokens):
        # Different loss for each input byte catches unweighted tail averages.
        scores = torch.zeros(*tokens.shape, 256)
        scores[..., 0] = tokens.float()
        return scores


def test_full_split_counts_each_target_and_weights_short_tail():
    evaluation = importlib.import_module('optcourse.evaluation')
    dataset = type('Fixture', (), {'parts': {'val': torch.tensor([0, 1, 2, 3, 4, 5, 6])}})()
    model = PositionModel().train()
    result = evaluation.evaluate_split(model, dataset, 'val', 4, 2, 'cpu')
    logits = model(dataset.parts['val'][:-1].unsqueeze(0))
    expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), dataset.parts['val'][1:]).item()
    assert result['predicted_bytes'] == 6
    assert result['loss'] == pytest.approx(expected, rel=1e-6)
    assert result['byte_perplexity'] == pytest.approx(math.exp(expected), rel=1e-6)
    assert result['context_policy'] == 'nonoverlapping_blocks_reset_context'
    assert model.training
    assert all(p.grad is None for p in model.parameters())


def test_evaluation_restores_mode_after_failure():
    evaluation = importlib.import_module('optcourse.evaluation')
    class Broken(torch.nn.Module):
        def forward(self, tokens):
            raise RuntimeError('fixture')
    model = Broken().train()
    dataset = type('Fixture', (), {'parts': {'val': torch.arange(7)}})()
    with pytest.raises(RuntimeError, match='fixture'):
        evaluation.evaluate_split(model, dataset, 'val', 4, 2, 'cpu')
    assert model.training


def test_floor_normalization_matches_hand_computed_scalar_map():
    from optcourse.optim import orthogonalize
    matrix = torch.tensor([[.01]])
    actual = orthogonalize(matrix, steps=1, eps=.1, normalization='max')
    x = .1
    expected = 3.4445*x - 4.775*x**3 + 2.0315*x**5
    assert actual.item() == pytest.approx(expected, rel=1e-6)
    assert torch.equal(orthogonalize(torch.zeros(2, 3), normalization='max'), torch.zeros(2, 3))
    with pytest.raises(ValueError):
        orthogonalize(matrix, normalization='unknown')


def test_new_config_fields_are_strict_and_backward_compatible(tmp_path):
    from optcourse.config import load_config
    path = tmp_path/'config.toml'
    path.write_text('[run]\ntask="language"\nphase="final"\ncohort="course_v1"\n[optimizer]\nnormalization="max"\n[language]\neval_mode="full"\n', encoding='utf-8')
    config = load_config(path)
    assert config['run']['phase'] == 'final'
    assert config['language']['eval_mode'] == 'full'
    path.write_text(path.read_text().replace('"final"','"typo"'))
    with pytest.raises(ValueError, match='phase'):
        load_config(path)


def test_run_evaluation_checks_data_and_checkpoint_identity(tmp_path):
    evaluation = importlib.import_module('optcourse.evaluation')
    from optcourse.config import load_config
    from optcourse.data import prepare_data, ByteDataset
    from optcourse.model import TinyTransformer
    from optcourse.tracking import RunStore
    source = tmp_path/'input.txt'
    source.write_bytes(bytes(range(256))*4)
    prepare_data(source, tmp_path/'data')
    path = tmp_path/'config.toml'
    path.write_text('[run]\ntask="language"\ndevice="cpu"\n[language]\ndata_dir="data"\nn_layer=1\nn_head=2\nn_embd=8\nblock_size=4\neval_mode="full"\n', encoding='utf-8')
    config = load_config(path)
    dataset = ByteDataset(tmp_path/'data')
    model = TinyTransformer(config['language'])
    store = RunStore.create(tmp_path/'fixtures', config)
    store.update(prepared_dataset_fingerprint=dataset.fingerprint)
    store.save_checkpoint(dict(step=0,model=model.state_dict(),validation_loss=0.),best=True)
    store.finish('COMPLETED')
    result = evaluation.evaluate_run(store.path, 'val')
    assert result['predicted_bytes'] == len(dataset.parts['val'])-1
    assert evaluation.evaluate_run(store.path, 'val') == result
    payload = torch.load(store.path/'best.pt', weights_only=False)
    payload['prepared_dataset_fingerprint'] = 'mismatch'
    torch.save(payload, store.path/'best.pt')
    with pytest.raises(ValueError, match='dataset'):
        evaluation.evaluate_run(store.path, 'test')


def test_course_cache_cannot_be_adopted_without_accounting(tmp_path):
    import json
    from optcourse.config import load_config,fingerprint
    from optcourse.data import prepare_data,ByteDataset
    from optcourse.model import TinyTransformer
    from optcourse.tracking import RunStore
    from optcourse.evaluation import evaluate_run,file_sha256
    from optcourse.budget import BudgetLedger
    source=tmp_path/'text.txt';source.write_bytes(b'abcde'*100)
    prepare_data(source,tmp_path/'data')
    path=tmp_path/'config.toml'
    path.write_text('[run]\ntask="language"\ndevice="cuda"\ncohort="course_v1"\nphase="final"\n[language]\ndata_dir="data"\nn_layer=1\nn_embd=8\nn_head=2\nblock_size=4\n',encoding='utf-8')
    config=load_config(path);dataset=ByteDataset(tmp_path/'data')
    root=tmp_path/'study';root.mkdir();(root/'study_state.json').write_text('{}')
    store=RunStore.create(root/'language',config)
    store.update(prepared_dataset_fingerprint=dataset.fingerprint)
    store.save_checkpoint(dict(step=0,model=TinyTransformer(config['language']).state_dict()),best=True)
    store.finish('COMPLETED')
    cached=dict(loss=2.,checkpoint_sha256=file_sha256(store.path/'best.pt'),
        config_fingerprint=fingerprint(config),prepared_dataset_fingerprint=dataset.fingerprint)
    (store.path/'evaluations').mkdir()
    (store.path/'evaluations/test_best.json').write_text(json.dumps(cached))
    ledger=BudgetLedger(root/'gpu_budget.json')
    with ledger.session('try unaccounted cache'):
        with pytest.raises(ValueError,match='account'):
            evaluate_run(store.path,'test',budget=ledger)
