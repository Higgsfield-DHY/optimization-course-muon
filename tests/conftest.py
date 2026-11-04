import torch


def pytest_sessionstart(session):
    # Small verification fixtures avoid large BLAS thread-launch overhead.
    torch.set_num_threads(1)


def pytest_addoption(parser):
    parser.addoption('--run-training', action='store_true', default=False)


def pytest_collection_modifyitems(config, items):
    if not config.getoption('--run-training'):
        import pytest
        for item in items:
            if 'training' in item.keywords:
                item.add_marker(pytest.mark.skip(reason='explicit --run-training required'))
