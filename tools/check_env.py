# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.utils import collect_env as collect_base_env
from mmcv.utils import get_git_hash

import mmdeploy
from mmdeploy.utils import (get_backend_version, get_codebase_version,
                            get_root_logger)


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMDeploy'] = f'{mmdeploy.__version__}+{get_git_hash()[:7]}'

    return env_info


def check_backend():
    backend_versions = get_backend_version()
    ort_version = backend_versions['onnxruntime']
    trt_version = backend_versions['tensorrt']
    ncnn_version = backend_versions['ncnn']

    import mmdeploy.apis.onnxruntime as ort_apis
    logger = get_root_logger()
    logger.info(f'onnxruntime: {ort_version}\tops_is_avaliable : '
                f'{ort_apis.is_custom_ops_available()}')

    import mmdeploy.apis.tensorrt as trt_apis
    logger.info(f'tensorrt: {trt_version}\tops_is_avaliable : '
                f'{trt_apis.is_custom_ops_available()}')

    import mmdeploy.apis.ncnn as ncnn_apis
    logger.info(f'ncnn: {ncnn_version}\tops_is_avaliable : '
                f'{ncnn_apis.is_custom_ops_available()}')

    import mmdeploy.apis.pplnn as pplnn_apis
    logger.info(f'pplnn_is_avaliable: {pplnn_apis.is_available()}')

    import mmdeploy.apis.openvino as openvino_apis
    logger.info(f'openvino_is_avaliable: {openvino_apis.is_available()}')

    import mmdeploy.apis.snpe as snpe_apis
    logger.info(f'snpe_is_available: {snpe_apis.is_available()}')

    import mmdeploy.apis.ascend as ascend_apis
    logger.info(f'ascend_is_available: {ascend_apis.is_available()}')

    import mmdeploy.apis.coreml as coreml_apis
    logger.info(f'coreml_is_available: {coreml_apis.is_available()}')


def check_codebase():
    codebase_versions = get_codebase_version()
    for k, v in codebase_versions.items():
        logger.info(f'{k}:\t{v}')


if __name__ == '__main__':
    logger = get_root_logger()
    logger.info('\n')
    logger.info('**********Environmental information**********')
    for name, val in collect_env().items():
        logger.info('{}: {}'.format(name, val))
    logger.info('\n')
    logger.info('**********Backend information**********')
    check_backend()
    logger.info('\n')
    logger.info('**********Codebase information**********')
    check_codebase()
