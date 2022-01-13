# Copyright (c) OpenMMLab. All rights reserved.
import inspect

import mmcv
from torch import nn

from mmdeploy.utils.constants import Backend
from .rewriter_utils import RewriterRegistry, eval_with_import


class ModuleRewriter:
    """A module rewriter which maintains rewritten modules.

    The rewritten modules can be registered by calling
    register_rewrite_module().  By calling patch_model(), all the registered
    modules of model will be replaced.

    Examples:
        >>> @MODULE_REWRITER.register_rewrite_module(
        >>>     'mmedit.models.backbones.sr_backbones.SRCNN',
        >>>     backend='tensorrt')
        >>> class SRCNNWrapper(torch.nn.Module):
        >>>     # rewrite the module here
    """

    def __init__(self):
        self._registry = RewriterRegistry()

    def add_backend(self, backend: str):
        """Add a backend by calling the _registry.add_backend."""
        self._registry.add_backend(backend)

    def register_rewrite_module(self,
                                module_type: str,
                                backend: str = Backend.DEFAULT.value,
                                **kwargs):
        """The interface of module rewriter decorator.

        Args:
            module_type (str): The module type name to rewrite.
            backend (str): The inference engine name.

        Returns:
            nn.Module: THe rewritten model.
        """
        return self._registry.register_object(module_type, backend, **kwargs)

    def patch_model(self,
                    model: nn.Module,
                    cfg: mmcv.Config,
                    backend: str = Backend.DEFAULT.value,
                    recursive: bool = True,
                    **kwargs) -> nn.Module:
        """Replace the models that was registered.

        Args:
            model (torch.nn.Module): The model to patch.
            cfg (Dict): Config dictionary of deployment.
            backend (str): The inference engine name.
            recursive (bool): The flag to enable recursive patching.

        Returns:
            nn.Module: THe patched model.

        Examples:
            >>> from mmdeploy.core import patch_model
            >>> patched_model = patch_model(model, cfg=deploy_cfg,
            >>>                             backend=backend)
        """
        self._collect_record(backend)
        return self._replace_module(model, cfg, recursive, **kwargs)

    def _replace_one_module(self, module, cfg, **kwargs):
        """Build a rewritten model."""
        object_dict = self._records.get(type(module), None)
        if object_dict is None:
            return module

        module_class = object_dict['_object']

        # Pop arguments that are not supported
        input_args = kwargs.copy()
        supported_args = inspect.getfullargspec(module_class.__init__).args
        redundant_key_name = []
        for k in input_args:
            if k not in supported_args:
                redundant_key_name.append(k)
        for k in redundant_key_name:
            input_args.pop(k)

        return module_class(module, cfg, **input_args)

    def _replace_module(self, model: nn.Module, cfg: mmcv.Config,
                        recursive: bool, **kwargs):
        """DFS and replace target models."""

        def _replace_module_impl(model, cfg, **kwargs):
            if recursive and hasattr(model, 'named_children'):
                for name, module in model.named_children():
                    model._modules[name] = _replace_module_impl(
                        module, cfg, **kwargs)
            return self._replace_one_module(model, cfg, **kwargs)

        return _replace_module_impl(model, cfg, **kwargs)

    def _collect_record(self, backend: str):
        """Collect models in registry."""
        self._records = {}
        records = self._registry.get_records(backend)
        for name, kwargs in records:
            self._records[eval_with_import(name)] = kwargs
