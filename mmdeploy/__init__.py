import importlib
import logging

importlib.import_module('mmdeploy.pytorch')

if importlib.util.find_spec('mmcv'):
    importlib.import_module('mmdeploy.mmcv')
else:
    logging.debug('mmcv is not installed.')

if importlib.util.find_spec('mmcls'):
    importlib.import_module('mmdeploy.mmcls')
else:
    logging.debug('mmcls is not installed.')

if importlib.util.find_spec('mmdet'):
    importlib.import_module('mmdeploy.mmdet')
else:
    logging.debug('mmdet is not installed.')

if importlib.util.find_spec('mmseg'):
    importlib.import_module('mmdeploy.mmseg')
else:
    logging.debug('mmseg is not installed.')

if importlib.util.find_spec('mmocr'):
    # remove already registered modules to avoid error
    from mmcv.cnn import UPSAMPLE_LAYERS
    if 'DeconvModule' in UPSAMPLE_LAYERS:
        UPSAMPLE_LAYERS.module_dict.pop('DeconvModule')
    if 'InterpConv' in UPSAMPLE_LAYERS:
        UPSAMPLE_LAYERS.module_dict.pop('InterpConv')
    importlib.import_module('mmdeploy.mmocr')
else:
    logging.debug('mmocr is not installed.')

if importlib.util.find_spec('mmedit'):
    importlib.import_module('mmdeploy.mmedit')
else:
    logging.debug('mmedit is not installed.')
