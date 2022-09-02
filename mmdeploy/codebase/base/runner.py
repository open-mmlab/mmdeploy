# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Union

import torch
from mmengine.device import get_device
from mmengine.logging import MMLogger
from mmengine.model import BaseModel
from mmengine.registry import LOOPS
from mmengine.runner import Runner, TestLoop, autocast


class DeployTestRunner(Runner):

    def __init__(self,
                 log_file: Optional[str] = None,
                 device: str = get_device(),
                 *args,
                 **kwargs):
        self._log_file = log_file
        self._device = device
        super().__init__(*args, **kwargs)

    def wrap_model(self, model_wrapper_cfg: Optional[Dict],
                   model: BaseModel) -> BaseModel:
        """Wrap the model to :obj:``MMDistributedDataParallel`` or other custom
        distributed data-parallel module wrappers.

        An example of ``model_wrapper_cfg``::

            model_wrapper_cfg = dict(
                broadcast_buffers=False,
                find_unused_parameters=False
            )

        Args:
            model_wrapper_cfg (dict, optional): Config to wrap model. If not
                specified, ``DistributedDataParallel`` will be used in
                distributed environment. Defaults to None.
            model (BaseModel): Model to be wrapped.

        Returns:
            BaseModel or DistributedDataParallel: BaseModel or subclass of
            ``DistributedDataParallel``.
        """
        return model.to(self._device)

    def build_logger(self,
                     log_level: Union[int, str] = 'INFO',
                     log_file: str = None,
                     **kwargs) -> MMLogger:
        """Build a global asscessable MMLogger.

        Args:
            log_level (int or str): The log level of MMLogger handlers.
                Defaults to 'INFO'.
            log_file (str, optional): Path of filename to save log.
                Defaults to None.
            **kwargs: Remaining parameters passed to ``MMLogger``.

        Returns:
            MMLogger: A MMLogger object build from ``logger``.
        """
        if log_file is None:
            log_file = self._log_file

        return super().build_logger(log_level, log_file, **kwargs)


@LOOPS.register_module()
class DeployTestLoop(TestLoop):
    """Loop for test.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 testing. Defaults to
            False.
    """

    def run(self, skip_preprocessor: bool = False) -> dict:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            if idx == 0:
                continue
            self.run_iter(idx, data_batch, skip_preprocessor)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics

    @torch.no_grad()
    def run_iter(self,
                 idx,
                 data_batch: Sequence[dict],
                 skip_preprocessor: bool = False) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # predictions should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            # if not skip_preprocessor:
            data_batch = self.runner.model.data_preprocessor(data_batch, False)
            outputs = self.runner.model._run_forward(
                data_batch, mode='predict')
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
