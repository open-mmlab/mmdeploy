# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import Registry

from mmdeploy.utils.config_utils import Backend


def __build_backend_wrapper_class(backend: Backend, registry: Registry):
    return registry.module_dict[backend.value]


BACKEND_WRAPPER = Registry('backend', __build_backend_wrapper_class)


def get_backend_wrapper_class(backend: Backend) -> type:
    """Get the backend wrapper class from the registry.

    Args:
        backend (Backend): The backend enum type.

    Returns:
        type: The backend wrapper class
    """
    return BACKEND_WRAPPER.build(backend)


def get_backend_file_count(backend: Backend):
    backend_class = get_backend_wrapper_class(backend)
    return backend_class.get_backend_file_count()
