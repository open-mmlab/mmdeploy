# Copyright (c) OpenMMLab. All rights reserved.
import pytest


def pytest_ignore_collect(collection_path, path, config):
    import importlib
    return importlib.util.find_spec('mmaction') is None


@pytest.fixture(autouse=True)
def init_test():
    # init default scope
    from mmdeploy.codebase import import_codebase
    from mmdeploy.utils import Codebase
    import_codebase(Codebase.MMACTION)
