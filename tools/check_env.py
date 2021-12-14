# Copyright (c) OpenMMLab. All rights reserved.
import logging

from mmcv.utils import collect_env as collect_base_env
from mmcv.utils import get_git_hash

import mmdeploy


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
    logging.info(f'onnxruntime: {ort_version} ops_is_avaliable : '
                 f'{ort_apis.is_available()}')

    try:
        import tensorrt as trt
    except ImportError:
        trt_version = None
    else:
        trt_version = trt.__version__
    import mmdeploy.apis.tensorrt as trt_apis
    logging.info(
        f'tensorrt: {trt_version} ops_is_avaliable : {trt_apis.is_available()}'
    )

    try:
        import ncnn
    except ImportError:
        ncnn_version = None
    else:
        ncnn_version = ncnn.__version__
    import mmdeploy.apis.ncnn as ncnn_apis
    logging.info(
        f'ncnn: {ncnn_version} ops_is_avaliable : {ncnn_apis.is_available()}')

    import mmdeploy.apis.pplnn as pplnn_apis
    logging.info(f'pplnn_is_avaliable: {pplnn_apis.is_available()}')

    import mmdeploy.apis.openvino as openvino_apis
    logging.info(f'openvino_is_avaliable: {openvino_apis.is_available()}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    for name, val in collect_env().items():
        logging.info('{}: {}'.format(name, val))
    check_backend()
