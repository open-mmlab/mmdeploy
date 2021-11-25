import importlib
import logging

importlib.import_module('mmdeploy.pytorch')

if importlib.util.find_spec('mmcv'):
    importlib.import_module('mmdeploy.mmcv')
else:
    logging.debug('mmcv is not installed.')
