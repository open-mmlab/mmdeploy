import importlib
import logging

from .base import BaseTask, MMCodebase, get_codebase_class

if importlib.util.find_spec('mmcls'):
    importlib.import_module('mmdeploy.codebase.mmcls')
else:
    logging.debug('mmcls is not installed.')

if importlib.util.find_spec('mmdet'):
    importlib.import_module('mmdeploy.codebase.mmdet')
else:
    logging.debug('mmdet is not installed.')

if importlib.util.find_spec('mmseg'):
    importlib.import_module('mmdeploy.codebase.mmseg')
else:
    logging.debug('mmseg is not installed.')

if importlib.util.find_spec('mmocr'):
    importlib.import_module('mmdeploy.codebase.mmocr')
else:
    logging.debug('mmocr is not installed.')

if importlib.util.find_spec('mmedit'):
    importlib.import_module('mmdeploy.codebase.mmedit')
else:
    logging.debug('mmedit is not installed.')

__all__ = ['MMCodebase', 'BaseTask', 'get_codebase_class']
