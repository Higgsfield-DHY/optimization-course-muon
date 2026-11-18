"""Regression for truthful final-test metadata; temporary CPU inference only."""
import json
import torch

torch.set_num_threads(1)


def test_final_test_receipt_updates_metadata(tmp_path):
    from optcourse.config import load_config,public_config,fingerprint
    from optcourse.data import prepare_data,ByteDataset
    from optcourse.model import TinyTransformer
    from optcourse.evaluation import evaluate_run
    source=tmp_path/'text.txt';source.write_bytes(b'abcde'*100)
    prepare_data(source,tmp_path/'data')
    path=tmp_path/'config.toml'
    path.write_text('[run]\ntask="language"\ndevice="cpu"\nphase="verification"\n[language]\ndata_dir="data"\nn_layer=1\nn_embd=8\nn_head=2\nblock_size=4\n',encoding='utf-8')
    config=load_config(path);dataset=ByteDataset(tmp_path/'data')
    run=tmp_path/'fixture';run.mkdir()
    meta=dict(status='COMPLETED',config=public_config(config),config_fingerprint=fingerprint(config),
        prepared_dataset_fingerprint=dataset.fingerprint,test_set_evaluated=False)
    (run/'metadata.json').write_text(json.dumps(meta),encoding='utf-8')
    torch.save(dict(step=0,model=TinyTransformer(config['language']).state_dict(),
        config_fingerprint=fingerprint(config),prepared_dataset_fingerprint=dataset.fingerprint),run/'best.pt')
    evaluate_run(run,'test')
    receipt=json.loads((run/'metadata.json').read_text(encoding='utf-8'))
    assert receipt['test_set_evaluated'] is True
    assert receipt['test_evaluation_checkpoint']=='best'
