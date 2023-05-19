# Copyright (c) OpenMMLab. All rights reserved.
import pytest


@pytest.fixture(autouse=True)
def init_test():
    # init default scope
    from mmagic.utils import register_all_modules

    register_all_modules(True)
