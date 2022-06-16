# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.backend.torchscript import get_ops_path
from .trace import trace

__all__ = ['get_ops_path', 'trace']
