# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Codebase


def pytest_ignore_collect(*args, **kwargs):
    import importlib
    return importlib.util.find_spec('mmpose') is None


@pytest.fixture(autouse=True, scope='package')
def import_all_modules():
    codebase = Codebase.MMPOSE
    try:
        import_codebase(codebase)
    except ImportError:
        pytest.skip(f'{codebase} is not installed.', allow_module_level=True)
