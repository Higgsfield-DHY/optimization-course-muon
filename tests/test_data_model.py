import hashlib
import importlib

import pytest
import torch


def prepared(tmp_path):
    data = importlib.import_module("optcourse.data")
    source = tmp_path / "input.txt"
    source.write_bytes(b"A" * 800 + b"B" * 100 + b"C" * 100)
    output = tmp_path / "prepared"
    data.prepare_data(source, output, ratios=(.8, .1, .1))
    return data, source, output


def test_data_splits_and_sampled_windows_do_not_cross_boundaries(tmp_path):
    data, source, output = prepared(tmp_path)
    assert b"".join((output / f"{name}.bin").read_bytes() for name in ("train", "val", "test")) == source.read_bytes()
    dataset = data.ByteDataset(output)
    for split, expected in [("train", 65), ("val", 66), ("test", 67)]:
        x, y = dataset.batch(split, 2, 8, torch.Generator().manual_seed(0), "cpu")
        assert x.shape == y.shape == (2, 8)
        assert (x == expected).all() and (y == expected).all()
    assert dataset.manifest["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_corrupted_data_and_short_splits_are_rejected(tmp_path):
    data, _, output = prepared(tmp_path)
    dataset = data.ByteDataset(output)
    with pytest.raises(ValueError):
        dataset.batch("val", 1, 100, torch.Generator(), "cpu")
    (output / "val.bin").write_bytes(b"X" * 100)
    with pytest.raises(ValueError, match="checksum"):
        data.ByteDataset(output)


def test_preparation_does_not_overwrite_existing_dataset(tmp_path):
    data, source, output = prepared(tmp_path)
    with pytest.raises(FileExistsError):
        data.prepare_data(source, output)


def test_prepared_identity_and_resume_detect_changed_split_assignments(tmp_path):
    data, source, first_dir = prepared(tmp_path)
    second_dir = tmp_path / "other_splits"
    data.prepare_data(source, second_dir, ratios=(.6, .2, .2))
    first, second = data.ByteDataset(first_dir), data.ByteDataset(second_dir)
    assert first.manifest["source_sha256"] == second.manifest["source_sha256"]
    assert first.fingerprint != second.fingerprint
    tracking = importlib.import_module("optcourse.tracking")
    experiment = importlib.import_module("optcourse.experiments")
    config = dict(run=dict(name="identity", task="language", seed=0, device="cpu"),
                  optimizer=dict(name="adamw"), language=dict(data_dir=str(first_dir)))
    store = tracking.RunStore.create(tmp_path / "fixtures", config)
    store.update(prepared_dataset_fingerprint=first.fingerprint)
    store.save_checkpoint(dict(step=0))
    store.finish("INTERRUPTED")
    with pytest.raises(ValueError, match="dataset"):
        experiment._open_store(config, store.path, second.fingerprint)


def test_model_logits_are_causal_and_embedding_is_tied_without_training():
    module = importlib.import_module("optcourse.model")
    torch.manual_seed(7)
    model = module.TinyTransformer(dict(n_layer=1, n_head=2, n_embd=8, block_size=8, dropout=0.)).eval()
    with torch.no_grad():
        first = model(torch.tensor([[1, 2, 3, 4]]))
        second = model(torch.tensor([[1, 2, 99, 98]]))
    assert first.shape == (1, 4, 256)
    assert model.head.weight is model.token_embedding.weight
    torch.testing.assert_close(first[:, :2], second[:, :2], atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError):
        model(torch.zeros(1, 9, dtype=torch.long))
