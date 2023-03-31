# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta
from typing import Optional, Sequence, Union

import mmengine
from mmengine.model import BaseModel
from torch import nn

from mmdeploy.utils import Backend, get_ir_config


class BaseBackendModel(BaseModel, metaclass=ABCMeta):
    """A backend model wraps the details to initialize and run a backend
    engine."""

    def __init__(self,
                 deploy_cfg: Optional[Union[str, mmengine.Config]] = None,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 *args,
                 **kwargs):
        """The default for building the base class.

        Args:
            deploy_cfg (str | mmengine.Config | None): The deploy config.
        """
        input_names = output_names = None
        if deploy_cfg is not None:
            ir_config = get_ir_config(deploy_cfg)
            output_names = ir_config.get('output_names', None)
            input_names = ir_config.get('input_names', None)
        # TODO use input_names instead in the future for multiple inputs
        self.input_name = input_names[0] if input_names else 'input'
        self.output_names = output_names if output_names else ['output']
        super(BaseBackendModel,
              self).__init__(data_preprocessor=data_preprocessor)

    @staticmethod
    def _build_wrapper(backend: Backend,
                       backend_files: Sequence[str],
                       device: str,
                       input_names: Optional[Sequence[str]] = None,
                       output_names: Optional[Sequence[str]] = None,
                       deploy_cfg: Optional[mmengine.Config] = None,
                       *args,
                       **kwargs):
        """The default methods to build backend wrappers.

        Args:
            backend (Backend): The backend enum type.
            backend_files (Sequence[str]): Paths to all required backend files(
                e.g. '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
            device (str): A string specifying device type.
            input_names (Sequence[str] | None): Names of model inputs in
                order. Defaults to `None`.
            output_names (Sequence[str] | None): Names of model outputs in
                order. Defaults to `None` and the wrapper will load the output
                names from the model.
            deploy_cfg: Deployment config file.
        """
        from mmdeploy.backend.base import get_backend_manager
        backend_mgr = get_backend_manager(backend.value)
        if backend_mgr is None:
            raise NotImplementedError(
                f'Unsupported backend type: {backend.value}')
        return backend_mgr.build_wrapper(backend_files, device, input_names,
                                         output_names, deploy_cfg, **kwargs)

    def destroy(self):
        if hasattr(self, 'wrapper') and hasattr(self.wrapper, 'destroy'):
            self.wrapper.destroy()
