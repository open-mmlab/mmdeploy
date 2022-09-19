# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from mmengine.runner.amp import autocast

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmedit.engine.runner.multi_loops.MultiTestLoop.run_iter')
def multi_test_loop__run_iter(
        ctx,
        self,
        idx: int,
        data_batch: Sequence[dict]):
    """Rewrite `run_iter` of MultiTestLoop for default backend.

    Args:
        idx (int): The index of the current batch in the loop.
        data_batch (Sequence[dict]): Batch of data
            from dataloader.
    """
    self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
    # outputs should be sequence of BaseDataElement
    with autocast(enabled=self.fp16):
        predictions = self.runner.model.test_step(data_batch)
    self.evaluator.process(predictions, data_batch)
    self.runner.call_hook(
        'after_test_iter',
        batch_idx=idx,
        data_batch=data_batch,
        outputs=predictions)
