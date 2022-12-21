# Copyright (c) OpenMMLab. All rights reserved.
import os
from abc import abstractmethod
from typing import Any, Dict, Optional, Union

import tvm
from mmcv.utils import Registry
from tvm import IRModule, auto_scheduler, autotvm, relay
from tvm.target import Target

from mmdeploy.utils import get_root_logger

TVM_TUNER = Registry('tvm_tuner')
AUTOTVM_TUNER = Registry('autotvm_tuner')
AUTOTVM_BUILDER = Registry('autotvm_builder')
AUTOTVM_RUNNER = Registry('autotvm_runner')
AUTO_SCHEDULER_BUILDER = Registry('auto_scheduler_builder')
AUTO_SCHEDULER_RUNNER = Registry('auto_scheduler_runner')


def build_tvm_tuner(cfg: Dict):
    """Build the tvm tuner.

    Args:
        cfg (Dict): The build config

    Returns:
        Any: The tvm tuner instance
    """
    return TVM_TUNER.build(cfg)


def build_autotvm_tuner(cfg: Dict):
    """Build the autotvm tuner.

    Args:
        cfg (Dict): The build config

    Returns:
        Any: The autotvm tuner instance
    """
    return AUTOTVM_TUNER.build(cfg)


def build_autotvm_builder(cfg: Dict):
    """Build the autotvm builder.

    Args:
        cfg (Dict): The build config

    Returns:
        Any: The autotvm builder instance
    """
    return AUTOTVM_BUILDER.build(cfg)


def build_autotvm_runner(cfg: Dict):
    """Build the autotvm runner.

    Args:
        cfg (Dict): The build config

    Returns:
        Any: The autotvm runner instance
    """
    return AUTOTVM_RUNNER.build(cfg)


def build_auto_scheduler_builder(cfg: Dict):
    """Build the ansor builder.

    Args:
        cfg (Dict): The build config

    Returns:
        Any: The ansor builder instance
    """
    return AUTO_SCHEDULER_BUILDER.build(cfg)


def build_auto_scheduler_runner(cfg: Dict):
    """Build the ansor tuner.

    Args:
        cfg (Dict): The build config

    Returns:
        Any: The ansor tuner instance
    """
    return AUTO_SCHEDULER_RUNNER.build(cfg)


AUTOTVM_TUNER.register_module()(autotvm.tuner.XGBTuner)
AUTOTVM_TUNER.register_module()(autotvm.tuner.GATuner)
AUTOTVM_TUNER.register_module()(autotvm.tuner.GridSearchTuner)
AUTOTVM_TUNER.register_module()(autotvm.tuner.RandomTuner)

AUTOTVM_BUILDER.register_module()(autotvm.LocalBuilder)

AUTOTVM_RUNNER.register_module()(autotvm.LocalRunner)
AUTOTVM_RUNNER.register_module()(autotvm.RPCRunner)

AUTO_SCHEDULER_BUILDER.register_module()(auto_scheduler.LocalBuilder)

AUTO_SCHEDULER_RUNNER.register_module()(auto_scheduler.LocalRunner)
AUTO_SCHEDULER_RUNNER.register_module()(auto_scheduler.RPCRunner)


class TVMTunerBase:
    """The base class of TVM tuner.

    Args:
        target (Union[str, Target]): The target platform to be tuned.
        opt_level (int): The optimization level.
        use_vm (bool): Enable tvm virtual machine runtime.
    """

    def __init__(self,
                 target: Union[str, Target],
                 opt_level: int = 3,
                 use_vm: bool = False) -> None:
        if isinstance(target, str):
            target = Target(target)
        self._target = target
        self._opt_level = opt_level
        self._use_vm = use_vm

    @property
    def use_vm(self) -> bool:
        """Get use_vm.

        Returns:
            bool: use_vm
        """
        return self._use_vm

    @abstractmethod
    def tune(self, mod: IRModule, params: Dict):
        """Tune the graph.

        Args:
            mod (IRModule): The graph module.
            params (Dict): The graph parameters.
        """
        raise NotImplementedError('tune method not implemented.')

    def build(self, mod: IRModule, params: Dict):
        """Build tuning library.

        Args:
            mod (IRModule): IRModule to build
            params (Dict): Parameter of the mod

        Returns:
            lib: The runtime factory for the graph executor
        """
        with tvm.transform.PassContext(opt_level=self._opt_level):
            if self._use_vm:
                ret = relay.vm.compile(mod, target=self._target, params=params)
            else:
                ret = relay.build_module.build(
                    mod, target=self._target, params=params)

        return ret


@TVM_TUNER.register_module
class DefaultTuner(TVMTunerBase):
    """The Default tuner, do nothing when tuning.

    Args:
        target (Union[str, Target]): The target platform to be tuned.
        opt_level (int): The optimization level.
        use_vm (bool): Enable tvm virtual machine runtime.
    """

    def __init__(self,
                 target: Union[str, Target],
                 opt_level: int = 3,
                 use_vm: bool = False) -> None:
        super().__init__(target, opt_level, use_vm)

    def tune(self, mod: IRModule, params: Dict):
        """Tune model, Default tuner does nothing."""
        pass


@TVM_TUNER.register_module
class AutoTVMTuner(TVMTunerBase):

    def __init__(self,
                 target: Union[str, Target],
                 log_file: str,
                 n_trial: int,
                 tuner: Dict,
                 opt_level: int = 3,
                 use_vm: bool = False,
                 early_stopping: Optional[int] = None,
                 builder: Union[Dict,
                                Any] = dict(type='LocalBuilder', timeout=10),
                 runner: Union[Dict, Any] = dict(
                     type='LocalRunner',
                     number=20,
                     repeat=3,
                     timeout=4,
                     min_repeat_ms=150),
                 use_transfer_learning: bool = True) -> None:
        """The AutoTVM tuner.

        Args:
            target (Union[str, Target]): The target platform to tune.
            log_file (str): the log file path.
            n_trial (int): Maximum number of configs to try.
            tuner (Dict): The autotvm tuner config.
            opt_level (int, optional): The optimization level. Defaults to 3.
            use_vm (bool, optional): Enable tvm virtual machine.
                Defaults to False.
            early_stopping (Optional[int], optional): Early stop the tuning
                when not finding better configs in this number of trials.
            builder (Union[Dict, Any], optional): The builder config.
            runner (Union[Dict, Any], optional): The runner config.
            use_transfer_learning (bool, optional): Whether to use transfer
                learning. Defaults to True.
        """
        super().__init__(target, opt_level, use_vm)
        self._log_file = log_file
        self._n_trial = n_trial
        self._tuner = tuner
        self._early_stopping = early_stopping
        self._use_transfer_learning = use_transfer_learning

        if isinstance(builder, Dict):
            builder = build_autotvm_builder(builder)

        if isinstance(runner, Dict):
            runner = build_autotvm_runner(runner)

        self._measure_option = autotvm.measure_option(
            builder=builder, runner=runner)

    def tune(self, mod: IRModule, params: Dict):
        """Tune the graph.

        Args:
            mod (IRModule): The graph module.
            params (Dict): The graph parameters.
        """
        logger = get_root_logger()
        target = self._target
        logger.info('Create autotvm task.')
        tasks = autotvm.task.extract_from_program(
            mod['main'], target=target, params=params)

        # create tmp log file
        if os.path.exists(self._log_file):
            os.remove(self._log_file)
        tmp_log_file = self._log_file + '.tmp'
        if os.path.exists(tmp_log_file):
            os.remove(tmp_log_file)

        tuner_cfg = self._tuner
        for i, task in enumerate(reversed(tasks)):
            prefix = '[Task %3d/%3d] ' % (i + 1, len(tasks))

            tuner_cfg['task'] = task
            tuner_obj = build_autotvm_tuner(tuner_cfg)

            if self._use_transfer_learning:
                if os.path.isfile(tmp_log_file) and os.path.exists(
                        tmp_log_file):
                    tuner_obj.load_history(
                        autotvm.record.load_from_file(tmp_log_file))

            # do tuning
            tsk_trial = min(self._n_trial, len(task.config_space))
            tuner_obj.tune(
                n_trial=tsk_trial,
                early_stopping=self._early_stopping,
                measure_option=self._measure_option,
                callbacks=[
                    autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                    autotvm.callback.log_to_file(tmp_log_file),
                ],
            )

        # pick best records to a cache file
        autotvm.record.pick_best(tmp_log_file, self._log_file)
        if os.path.exists(tmp_log_file):
            os.remove(tmp_log_file)

    def build(self, mod: IRModule, params: Dict):
        """Build tuning library.

        Args:
            mod (IRModule): IRModule to build
            params (Dict): Parameter of the mod

        Returns:
            lib: The runtime factory for the graph executor
        """
        with autotvm.apply_history_best(self._log_file):
            with tvm.transform.PassContext(opt_level=self._opt_level):
                if self._use_vm:
                    ret = relay.vm.compile(
                        mod, target=self._target, params=params)
                else:
                    ret = relay.build_module.build(
                        mod, target=self._target, params=params)

        return ret


@TVM_TUNER.register_module
class AutoScheduleTuner(TVMTunerBase):

    def __init__(
        self,
        target: Union[str, Target],
        log_file: str,
        num_measure_trials: int,
        opt_level: int = 3,
        use_vm: bool = False,
        early_stopping: Optional[int] = None,
        builder: Union[Dict, Any] = dict(type='LocalBuilder', timeout=15),
        runner: Union[Dict, Any] = dict(
            type='LocalRunner', repeat=10, enable_cpu_cache_flush=True)
    ) -> None:
        """The Ansor tuner.

        Args:
            target (Union[str, Target]): The target platform to tune.
            log_file (str): the log file path.
            num_measure_trials (int): Maximum number of configs to try.
            opt_level (int, optional): The optimization level. Defaults to 3.
            use_vm (bool, optional): Enable tvm virtual machine.
                Defaults to False.
            early_stopping (Optional[int], optional): Early stop the tuning
                when not finding better configs in this number of trials.
            builder (Union[Dict, Any], optional): The builder config.
            runner (Union[Dict, Any], optional): The runner config.
        """
        super().__init__(target, opt_level, use_vm)
        self._log_file = log_file
        self._num_measure_trials = num_measure_trials
        self._early_stopping = early_stopping

        if isinstance(builder, Dict):
            builder = build_auto_scheduler_builder(builder)

        if isinstance(runner, Dict):
            # CUDA device need a different process for measurement
            if runner['type'] == 'LocalRunner':
                runner.pop('type')
                if Target(target).kind != 'llvm':
                    if 'enable_cpu_cache_flush' in runner:
                        runner['enable_cpu_cache_flush'] = False
                self._measure_ctx = auto_scheduler.LocalRPCMeasureContext(
                    **runner)
                runner = self._measure_ctx.runner
            else:
                runner = build_auto_scheduler_runner(runner)

        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=num_measure_trials,
            runner=runner,
            builder=builder,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )
        self._tune_option = tune_option

    def tune(self, mod: IRModule, params: Dict):
        """Tune the graph.

        Args:
            mod (IRModule): The graph module.
            params (Dict): The graph parameters.
        """
        logger = get_root_logger()
        target = self._target

        if os.path.exists(self._log_file):
            os.remove(self._log_file)

        logger.info('Create auto scheduler task.')
        tasks, task_weights = auto_scheduler.extract_tasks(
            mod['main'], params, target)

        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)

        logger.info('Begin tuning.')
        tuner.tune(self._tune_option)

    def build(self, mod: IRModule, params: Dict):
        """Build tuning library.

        Args:
            mod (IRModule): IRModule to build
            params (Dict): Parameter of the mod

        Returns:
            lib: The runtime factory for the graph executor
        """
        with auto_scheduler.ApplyHistoryBest(self._log_file):
            with tvm.transform.PassContext(
                    opt_level=self._opt_level,
                    config={'relay.backend.use_auto_scheduler': True}):
                if self._use_vm:
                    ret = relay.vm.compile(
                        mod, target=self._target, params=params)
                else:
                    ret = relay.build_module.build(
                        mod, target=self._target, params=params)

        return ret
