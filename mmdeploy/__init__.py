import importlib
import logging

from .version import __version__  # noqa F401

importlib.import_module('mmdeploy.pytorch')

if importlib.util.find_spec('mmcv'):
    importlib.import_module('mmdeploy.mmcv')
else:
    logging.debug('mmcv is not installed.')
