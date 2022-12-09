# Copyright (c) OpenMMLab. All rights reserved.
import pytest


@pytest.fixture(autouse=True)
def init_test():
    # init default scope
    from mmdeploy.codebase import import_codebase
    from mmdeploy.utils import Codebase
    try:
        import_codebase(Codebase.MMDET3D)
    except Exception as e:
        pytest.skip(
            f'Can not start codebase test because {e}',
            allow_module_level=True)
