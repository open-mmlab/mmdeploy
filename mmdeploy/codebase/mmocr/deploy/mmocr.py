# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmengine.registry import Registry
from packaging import version
from torch.utils.data import DataLoader

from mmdeploy.codebase.base import CODEBASE, MMCodebase
from mmdeploy.utils import Codebase

MMOCR_TASK = Registry('mmocr_tasks')


@CODEBASE.register_module(Codebase.MMOCR.value)
class MMOCR(MMCodebase):
    """MMOCR codebase class."""

    task_registry = MMOCR_TASK

    @staticmethod
    def single_gpu_test(model: torch.nn.Module,
                        data_loader: DataLoader,
                        show: bool = False,
                        out_dir: Optional[str] = None,
                        **kwargs):
        """Run test with single gpu.

        Args:
            model (torch.nn.Module): Input model from nn.Module.
            data_loader (DataLoader): PyTorch data loader.
            show (bool): Specifying whether to show plotted results. Defaults
                to `False`.
            out_dir (str): A directory to save results, defaults to `None`.

        Returns:
            list: The prediction results.
        """
        import mmocr

        # fixed the bug when using `--show-dir` after mocr v0.4.1
        if version.parse(mmocr.__version__) < version.parse('0.4.1'):
            from mmdet.apis import single_gpu_test
        else:
            from mmocr.apis import single_gpu_test
        outputs = single_gpu_test(model, data_loader, show, out_dir, **kwargs)
        return outputs
