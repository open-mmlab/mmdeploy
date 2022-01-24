# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.utils import collect_env as collect_base_env
from mmcv.utils import get_git_hash

import mmdeploy
from mmdeploy.utils import get_root_logger


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMDeployment'] = f'{mmdeploy.__version__}+{get_git_hash()[:7]}'

    return env_info


def check_backend():
    try:
        import onnxruntime as ort
    except ImportError:
        ort_version = None
    else:
        ort_version = ort.__version__
    import mmdeploy.apis.onnxruntime as ort_apis
    logger = get_root_logger()
    logger.info(f'onnxruntime: {ort_version} ops_is_avaliable : '
                f'{ort_apis.is_available()}')

    try:
        import tensorrt as trt
    except ImportError:
        trt_version = None
    else:
        trt_version = trt.__version__
    import mmdeploy.apis.tensorrt as trt_apis
    logger.info(
        f'tensorrt: {trt_version} ops_is_avaliable : {trt_apis.is_available()}'
    )

    try:
        import ncnn
    except ImportError:
        ncnn_version = None
    else:
        ncnn_version = ncnn.__version__
    import mmdeploy.apis.ncnn as ncnn_apis
    logger.info(
        f'ncnn: {ncnn_version} ops_is_avaliable : {ncnn_apis.is_available()}')

    import mmdeploy.apis.pplnn as pplnn_apis
    logger.info(f'pplnn_is_avaliable: {pplnn_apis.is_available()}')

    import mmdeploy.apis.openvino as openvino_apis
    logger.info(f'openvino_is_avaliable: {openvino_apis.is_available()}')


def get_version(s):
    return ''.join([c for c in s if c.isdigit() or c == '.'])


def get_op(s):
    return ''.join([c for c in s if c in '><='])


def check_codebase():
    ops = {
        '>=': (lambda x, y: x >= y),
        '==': (lambda x, y: x == y),
        '<=': (lambda x, y: x <= y),
        '': (lambda x, y: True)
    }
    with open('../requirements/optional.txt', 'r') as file:
        try:
            import mmcls
        except ImportError:
            mmcls_version = None
        else:
            mmcls_version = mmcls.__version__
        requirement = file.readline()
        is_available = ops[get_op(requirement)](mmcls_version,
                                                get_version(requirement))
        logger.info(f'mmcls: {mmcls_version} is_avaliable : {is_available}')

        try:
            import mmdet
        except ImportError:
            mmdet_version = None
        else:
            mmdet_version = mmdet.__version__
        requirement = file.readline()
        is_available = ops[get_op(requirement)](mmdet_version,
                                                get_version(requirement))
        logger.info(f'mmdet: {mmdet_version} is_avaliable : {is_available}')

        try:
            import mmedit
        except ImportError:
            mmedit_version = None
        else:
            mmedit_version = mmedit.__version__
        requirement = file.readline()
        is_available = ops[get_op(requirement)](mmedit_version,
                                                get_version(requirement))
        logger.info(f'mmedit: {mmedit_version} is_avaliable : {is_available}')

        try:
            import mmocr
        except ImportError:
            mmocr_version = None
        else:
            mmocr_version = mmocr.__version__
        requirement = file.readline()
        is_available = ops[get_op(requirement)](mmocr_version,
                                                get_version(requirement))
        logger.info(f'mmocr: {mmocr_version} is_avaliable : {is_available}')

        try:
            import mmseg
        except ImportError:
            mmseg_version = None
        else:
            mmseg_version = mmseg.__version__
        requirement = file.readline()
        is_available = ops[get_op(requirement)](mmseg_version,
                                                get_version(requirement))
        logger.info(f'mmseg: {mmseg_version} is_avaliable : {is_available}')


if __name__ == '__main__':
    logger = get_root_logger()
    for name, val in collect_env().items():
        logger.info('{}: {}'.format(name, val))
    check_backend()
    check_codebase()
