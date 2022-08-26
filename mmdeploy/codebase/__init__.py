# Copyright (c) OpenMMLab. All rights reserved.
import importlib

from mmdeploy.utils import Codebase
from .base import BaseTask, MMCodebase, get_codebase_class

extra_dependent_library = {
    Codebase.MMOCR: ['mmdet'],
    Codebase.MMROTATE: ['mmdet']
}


def import_codebase(codebase: Codebase):
    """Import a codebase package in `mmdeploy.codebase`

    The function will check if all dependent libraries are installed.
    For example, to import `mmdeploy.codebase.mmdet`, `mmdet` must be
    installed. To import `mmdeploy.codebase.mmocr`, `mmdet` and `mmocr`
    must be installed.

    Args:
        codebase (Codebase): The codebase to import.
    """
    codebase_name = codebase.value
    dependent_library = [codebase_name] + \
        extra_dependent_library.get(codebase, [])

    for lib in dependent_library:
        if not importlib.util.find_spec(lib):
            raise ImportError(
                f'{lib} has not been installed. '
                f'Import mmdeploy.codebase.{codebase_name} failed.')
        importlib.import_module(f'mmdeploy.codebase.{lib}')
        importlib.import_module(f'{lib}.models')
        if importlib.util.find_spec(f'{lib}.core'):
            # not all codebases have core submodule
            importlib.import_module(f'{lib}.core')
        importlib.import_module(f'{lib}.datasets')
        if importlib.util.find_spec(f'{lib}.structures'):
            # not all codebases have structures submodule
            importlib.import_module(f'{lib}.structures')
        if importlib.util.find_spec(f'{lib}.visualization'):
            # not all codebases have visualization submodule
            importlib.import_module(f'{lib}.visualization')
        if importlib.util.find_spec(f'{lib}.visualizer'):
            # not all codebases have visualizer submodule
            importlib.import_module(f'{lib}.visualizer')
        if importlib.util.find_spec(f'{lib}.engine'):
            # not all codebases have engine submodule
            importlib.import_module(f'{lib}.engine')


__all__ = ['MMCodebase', 'BaseTask', 'get_codebase_class']
