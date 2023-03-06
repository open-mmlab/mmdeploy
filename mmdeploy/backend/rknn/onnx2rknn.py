# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from mmdeploy.utils import get_root_logger


@dataclass
class RKNNConfig:
    """RKNN Config."""
    mean_values: List[List[int]] = None
    std_values: List[List[int]] = None
    optimization_level: int = 1
    target_platform: str = None


def rknn_package_info():
    """Get the rknn package information."""
    import pkg_resources
    toolkit = pkg_resources.working_set.by_key.get('rknn-toolkit', None)
    toolkit = pkg_resources.working_set.by_key.get('rknn-toolkit2', toolkit)
    if toolkit is None:
        return dict(name=None, version=None)
    else:
        return dict(name=toolkit.project_name, version=toolkit.version)


def onnx2rknn(onnx_model: str,
              output_path: str,
              input_names: List[str],
              output_names: List[str],
              input_shapes: Dict[str, List],
              rknn_config: RKNNConfig,
              do_quantization: bool = False,
              dataset: Optional[str] = None,
              pre_compile: bool = False):
    """Convert ONNX to RKNN.

    RKNN-Toolkit2 is a software development kit for users to perform model
    conversion, inference and performance evaluation on PC and Rockchip
    NPU platforms.

    Args:
        onnx_model (str): Input onnx model.
        output_path (str): File path to save RKNN model.
        input_names (List[str]): Names of the inputs.
        output_names (List[str]): Names of the outputs.
        input_shapes (ShapeType): The Default shape of the inputs.
        rknn_config (RKNNConfig): Config of the rknn toolset. Defined in
            `mmdeploy.backend.rknn.onnx2rknn`.
        do_quantization (bool): Enable model quantization.
        dataset (str): Dataset file. Each line is an image path.
        pre_compile (bool): Pre compile the model (smaller size and load
            quicker, but can't run on simulator)
    """
    from rknn.api import RKNN
    logger = get_root_logger()

    # get input/output names
    input_size_list = [list(input_shapes[name][1:]) for name in input_names]

    # init rknn
    rknn = RKNN(verbose=True)
    rknn.config(**asdict(rknn_config))

    # load onnx
    ret = rknn.load_onnx(
        model=onnx_model,
        inputs=input_names,
        input_size_list=input_size_list,
        outputs=output_names)
    if ret != 0:
        logger.error('Load model failed!')
        exit(ret)

    # quantization
    quantization_cfg = dict()
    if do_quantization:
        # disable quantization if dataset not exist
        if not osp.exists(dataset):
            do_quantization = False
            logger.warning('no dataset passed in, quantization is skipped')
        else:
            quantization_cfg['dataset'] = dataset

    # set batch size
    if do_quantization:
        batch_size = input_size_list[0][0]
        assert all(batch_size == shape[0] for shape in input_size_list)
        quantization_cfg['rknn_batch_size'] = batch_size

        # set pre compile
        if rknn_package_info()['name'] != 'rknn-toolkit2':
            quantization_cfg['pre_compile'] = pre_compile

    # do quantization
    quantization_cfg['do_quantization'] = do_quantization
    ret = rknn.build(**quantization_cfg)
    if ret != 0:
        logger.error('Build model failed!')
        exit(ret)

    # export
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        logger.error('Export rknn model failed!')
        exit(ret)
