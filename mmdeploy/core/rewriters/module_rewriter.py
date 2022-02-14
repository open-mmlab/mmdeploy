# Copyright (c) OpenMMLab. All rights reserved.
import inspect
from typing import Dict, List, Optional, Union

import mmcv
from torch import nn

from mmdeploy.utils.constants import IR, Backend
from .rewriter_utils import (Checker, RewriterRegistry, collect_env,
                             eval_with_import)


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

    def register_rewrite_module(
            self,
            module_type: str,
            backend: str = Backend.DEFAULT.value,
            ir: IR = IR.DEFAULT,
            extra_checkers: Optional[Union[Checker, List[Checker]]] = None,
            **kwargs):
        """The interface of module rewriter decorator.

        Args:
            module_type (str): The module type name to rewrite.
            backend (str): The rewriter will be activated on which backend.
            ir (IR): The rewriter will be activated on which IR.
            extra_checkers (Checker | List[Checker] | None): Other requirements
                defined by Checker.

        Returns:
            nn.Module: The rewritten model.
        """
        return self._registry.register_object(module_type, backend, ir,
                                              extra_checkers, **kwargs)

    def patch_model(self,
                    model: nn.Module,
                    cfg: mmcv.Config,
                    backend: str = Backend.DEFAULT.value,
                    ir: IR = IR.DEFAULT,
                    recursive: bool = True,
                    **kwargs) -> nn.Module:
        """Replace the models that was registered.

        Args:
            model (torch.nn.Module): The model to patch.
            cfg (Dict): Config dictionary of deployment.
            backend (str): The inference engine name.
            ir (IR): The intermeditate representation name.
            recursive (bool): The flag to enable recursive patching.

        Returns:
            nn.Module: THe patched model.

        Examples:
            >>> from mmdeploy.core import patch_model
            >>> patched_model = patch_model(model, cfg=deploy_cfg,
            >>>                             backend=backend)
        """
        # TODO: Make the type of parameter backend to Backend
        env = collect_env(Backend.get(backend), ir)
        self._collect_record(env)
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

    def _collect_record(self, env: Dict):
        """Collect models in registry."""
        self._records = {}
        records = self._registry.get_records(env)
        for name, kwargs in records:
            self._records[eval_with_import(name)] = kwargs
