# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from mmdeploy.utils.constants import Backend
from ..base import IR_MANAGERS, BaseIRManager, BaseIRParam


@dataclass
class TorchScriptParam(BaseIRParam):
    """TorchScript IR param.

    Args:
        args (Any): The arguments of the model.
        work_dir (str): The working directory to save the output.
        file_name (str): The file name of the output. postfix can be omitted.
        input_names (List[str]): The names to assign to the input of the ir.
        output_names (List[str]): The names to assign to the output of the ir.
        dynamic_axes (Dict): Determine the dynamic axes of the inputs. It not
            given, all axes will be static.
        backend (str): The expected backend of the ir.
        rewrite_context (Dict): Provide information to the rewriter.
        const_args (Any): The constant args of the model.
        check_trace (bool): Check outputs after trace.
        check_tolerance (float): The tolerance of the check outputs.
    """

    # latent fields
    _default_postfix = '.pth'

    # class fields
    const_args: Any = None
    check_trace: bool = True
    check_tolerance: float = 1e-05


@IR_MANAGERS.register('torchscript', param=TorchScriptParam)
class TorchScriptManager(BaseIRManager):
    """TorchScript IR Manager."""

    @classmethod
    def export(cls,
               model: Any,
               args: Any,
               output_path: str,
               backend: Union[Backend, str] = 'default',
               rewrite_context: Dict = None,
               check_trace: bool = True,
               check_tolerance: float = 1e-05,
               const_args: Optional[Dict] = None):
        """A wrapper of `torch.jit.trace` with some enhancement.

            Examples:
                >>> from mmdeploy.ir.torchscript import export
                >>>
                >>> func = create_model()
                >>> inputs = get_input_tensor()
                >>>
                >>> jit_model = export(
                >>>     func,
                >>>     inputs,
                >>>     backend='torchscript',
                >>>     check_trace=False)
                >>>

            Args:
                func (torch.nn.Module): A Python function or `torch.nn.Module`
                    that will be run with `example_inputs`.
                inputs (torch.Tensor, Tuple): A tuple of example inputs that
                    will be passed to the function while tracing.
                output_path (str): The output path.
                backend (Backend|str): Which backend will the graph be used.
                    Different backend would generate different graph.
                const_args (Dict): The constant inputs of the model.
                rewrite_context (Dict): The information that would be used in
                    the context of exporting.
                check_trace (bool): Check if the same inputs run through traced
                    code produce the same outputs.
                check_tolerance (float): Floating-point comparison tolerance to
                    use in the checker procedure.

            Returns:
                torch.jit.TracedModule: The traced torch jit model.
        """
        from .trace import trace
        trace(
            model,
            args,
            output_path,
            backend=backend,
            rewrite_context=rewrite_context,
            check_trace=check_trace,
            check_tolerance=check_tolerance,
            const_args=const_args)

    @classmethod
    def export_from_param(cls, model: Any, param: TorchScriptParam):
        """Export model to given ir.

        Examples:
            >>> from mmdeploy.ir.torchscript import export_from_param
            >>>
            >>> model = create_model()
            >>> param = TorchScriptParam(...)
            >>>
            >>> export_from_param(model, param)

        Args:
            model (Any): The model to be exported.
            param (TorchScriptParam): The packed export parameter.
        """

        from mmdeploy.utils import get_root_logger
        logger = get_root_logger()

        # check param validation
        param.check()

        # get output path
        work_dir = param.work_dir
        if not isinstance(work_dir, str):
            logger.warning('Invalid work_dir. Use `./work_dir` as default.')
            work_dir = './work_dir'

        assert isinstance(param.file_name, str), ('Expect string file name, '
                                                  f'got {type(param.name)}')
        output_path = osp.join(param.work_dir, param.file_name)

        cls.export(
            model,
            param.args,
            output_path,
            backend=param.backend,
            rewrite_context=param.rewrite_context,
            check_trace=param.check_trace,
            check_tolerance=param.check_tolerance,
            const_args=param.const_args)

    @classmethod
    def is_available(cls) -> bool:
        """check if the export is available."""
        import importlib
        return importlib.util.find_spec('torch') is not None
