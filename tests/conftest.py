import pytest

from emcomm.config import EnvConfig
from emcomm.layouts import make_layout_splits


@pytest.fixture(scope="session")
def cfg() -> EnvConfig:
    return EnvConfig()


@pytest.fixture(scope="session")
def splits(cfg):
    return make_layout_splits(cfg)
