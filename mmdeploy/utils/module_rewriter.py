from copy import deepcopy

from mmcv.utils import Registry

from .register_utils import eval_with_import


def build_rewrite_module(module, cfg, backend, registry, **kwargs):

    backend_dict = registry.module_eval_dict.get(type(module), None)
    if backend_dict is None:
        return module

    RewriteModuleClass = None
    for backend in [backend, 'default']:
        RewriteModuleClass = backend_dict.get(backend, None)
        if RewriteModuleClass is not None:
            break

    if RewriteModuleClass is None:
        return module

    return RewriteModuleClass(module, cfg, **kwargs)


class RewriteModuleRegistry(Registry):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._module_eval_dict = dict()

    def register_rewrite_module(self,
                                module_type,
                                backend='default',
                                **kwargs):
        register_name = module_type + '@' + backend
        return self.register_module(register_name)

    @property
    def module_eval_dict(self):
        return self._module_eval_dict

    def _register_module(self, module_class, module_name=None, force=False):
        super()._register_module(module_class, module_name, force)

        module_type, backend = module_name.split('@')
        module_type_cls = eval_with_import(module_type)
        if module_type_cls not in self._module_eval_dict:
            self._module_eval_dict[module_type_cls] = dict()

        assert (backend not in self._module_eval_dict[module_type_cls]
                ), '{} with backend:{} has already been registed.'.format(
                    module_type, backend)
        self._module_eval_dict[module_type_cls][backend] = self.module_dict[
            module_name]


# create register
MODULE_REWRITERS = RewriteModuleRegistry('module_rewriters',
                                         build_func=build_rewrite_module,
                                         scope='.')


def patch_model(model, cfg, backend='default', **kwargs):
    def _patch_impl(model, cfg, **kwargs):
        for name, module in model.named_children():
            model._modules[name] = _patch_impl(module, cfg, **kwargs)
        return MODULE_REWRITERS.build(module=model,
                                      cfg=cfg,
                                      backend=backend,
                                      **kwargs)

    return _patch_impl(deepcopy(model), cfg, **kwargs)
