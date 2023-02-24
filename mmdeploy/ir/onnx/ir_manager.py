# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..base import IR_MANAGERS, BaseIRManager, BaseIRParam


@dataclass
class ONNXIRParam(BaseIRParam):
    """ONNX IR params.

    Args:
        args (Any): The arguments of the model.
        work_dir (str): The working directory to save the output.
        file_name (str): The file name of the output. postfix can be omitted.
        input_names (List[str]): The names to assign to the input of the ir.
        output_names (List[str]): The names to assign to the output of the ir.
        dynamic_axes (Dict): Determine the dynamic axes of the inputs. It not
            given, all axes will be static.
        do_constant_folding (bool): Perform constant folding to the exported
            model. Default to True.
        opset_version (int): The version of the ONNX opset. Default to 11.
        backend (str): The expected backend of the ir.
        rewrite_context (Dict): Provide information to the rewriter.
        verbose (bool): Show detail log of ONNX export.
        const_args (Any): The constant args of the model.
        optimize (bool): Perform optimization.
    """
    # latent fields
    _default_postfix = '.onnx'

    # class fields
    do_constant_folding: bool = True
    opset_version: int = 11
    verbose: bool = False
    const_args: Any = None
    optimize: bool = True

    def check(self):
        super().check()
        import torch
        assert isinstance(
            self.args,
            (torch.Tensor, Tuple,
             Dict)), ('Expect args type: (torch.Tensor, Sequence, Dict),',
                      f' get type: {type(self.args)}.')
        assert self.opset_version >= 7, 'opset version < 7 is not supported.'


@IR_MANAGERS.register('onnx', params=ONNXIRParam)
class ONNXManager(BaseIRManager):
    """ONNX IR Manager."""

    @classmethod
    def export(cls,
               model: Any,
               args: Any,
               output_path: str,
               input_names: Optional[List[str]] = None,
               output_names: Optional[List[str]] = None,
               opset_version: int = 11,
               dynamic_axes: Optional[Dict] = None,
               backend: str = 'default',
               rewrite_context: Optional[Dict] = None,
               verbose: bool = False,
               const_args: Optional[Dict] = None,
               optimize: bool = True):
        """export model to ONNX.

        Examples:
            >>> from mmdeploy.ir.onnx import export
            >>>
            >>> model = create_model()
            >>> args = get_input_tensor()
            >>>
            >>> export(
            >>>     model,
            >>>     args,
            >>>     'place/to/save/model.onnx',
            >>>     backend='tensorrt',
            >>>     input_names=['input'],
            >>>     output_names=['output'],
            >>>     dynamic_axes={'input': {
            >>>         0: 'batch',
            >>>         2: 'height',
            >>>         3: 'width'
            >>>     }})

        Args:
            model (Any): Exportable PyTorch Model
            args (Any): Arguments are used to trace the graph.
            output_path (str): The output path.
            input_names (List[str], optional): The name of the input in
                the graph. Defaults to None.
            output_names (List[str], optional): The name of the output
                in the graph. Defaults to None.
            opset_version (int): The ONNX opset version. Defaults to 11.
            dynamic_axes (Dict, optional): Dynamic axes of each inputs.
                If not given, all inputs share the fixed shapes of the args.
            verbose (bool): Show detail export logs. Defaults to False.
            const_args (Dict, optional): The non-exported inputs of the model.
            rewrite_context (Dict, optional): The information used by
                the rewriter.
            optimize (bool): Enable optimize export model.
        """
        from .export import export
        export(
            model,
            args,
            output_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            verbose=verbose,
            backend=backend,
            const_args=const_args,
            rewrite_context=rewrite_context,
            optimize=optimize)

    @classmethod
    def export_from_param(cls, model: Any, param: ONNXIRParam):
        """Export model to ONNX by ONNXIRParam.

        Examples:
            >>> from mmdeploy.ir.onnx import export_from_param
            >>>
            >>> model = create_model()
            >>> param = ONNXIRParam(...)
            >>>
            >>> export_from_param(model, param)

        Args:
            model (Any): The model to be exported.
            params (ONNXIRParam): The packed export parameter.
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
            input_names=param.input_names,
            output_names=param.output_names,
            opset_version=param.opset_version,
            dynamic_axes=param.dynamic_axes,
            verbose=param.verbose,
            backend=param.backend,
            const_args=param.const_args,
            rewrite_context=param.rewrite_context,
            optimize=param.optimize)

    @classmethod
    def is_available(cls) -> bool:
        """check if the export is available."""
        import importlib
        return importlib.util.find_spec('torch') is not None
