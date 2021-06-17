import importlib
import logging

if importlib.util.find_spec('mmcv'):
    from .mmcv import *  # noqa: F401,F403
else:
    logging.debug('mmcv is not installed.')
if importlib.util.find_spec('mmcls'):
    from .mmcls import *  # noqa: F401,F403
else:
    logging.debug('mmcls is not installed.')

if importlib.util.find_spec('mmdet'):
    from .mmdet import *  # noqa: F401,F403
else:
    logging.debug('mmdet is not installed.')

if importlib.util.find_spec('mmseg'):
    from .mmseg import *  # noqa: F401,F403
else:
    logging.debug('mmseg is not installed.')
