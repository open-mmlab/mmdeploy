# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Dict, Union

import tvm
from mmcv.utils import Registry
from tvm import IRModule, autotvm, relay
from tvm.target import Target

TVM_AUTO_TUNER = Registry('tvm_auto_tuner')


def build_tvm_auto_tuner(cfg):
    return TVM_AUTO_TUNER.build(cfg)


class TVMTunerBase:

    def __init__(self, target: Union[str, Target]) -> None:
        if isinstance(target, str):
            target = Target(target)
        self._target = target

    @abstractmethod
    def tune(self, mod: IRModule, params: Dict):
        raise NotImplementedError('tune method not implemented.')

    def build(self, mod: IRModule, params: Dict):
        """Build tuning library.

        Args:
            mod (IRModule): IRModule to build
            params (Dict): Parameter of the mod

        Returns:
            lib: The runtime factory for the graph executor
        """
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(
                mod, target=self._target, params=params)

        return lib


@TVM_AUTO_TUNER.register_module
class DefaultTuner(TVMTunerBase):

    def __init__(self, target: Union[str, Target]) -> None:
        super().__init__(target)

    def tune(self, mod: IRModule, params: Dict):
        """Tune model, This tuner does not need to tune."""
        pass


@TVM_AUTO_TUNER.register_module
class AutoTVMTuner(TVMTunerBase):

    def __init__(self, target: Union[str, Target], log_file: str) -> None:
        super().__init__(target)
        self._log_file = log_file

    def tune(self, mod: IRModule, params: Dict):
        pass

    def build(self, mod: IRModule, params: Dict):
        with autotvm.apply_history_best(self._log_file):
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build_module.build(
                    mod, target=self._target, params=params)

        return lib
