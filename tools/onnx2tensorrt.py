# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging

from mmdeploy.backend.tensorrt import create_trt_engine, save_trt_engine
from mmdeploy.backend.tensorrt.utils import get_trt_log_level
from mmdeploy.utils import get_common_config, get_model_inputs, load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Convert ONNX to TensorRT.')
    parser.add_argument('deploy_cfg', help='deploy config path')
    parser.add_argument('onnx_path', help='ONNX model path')
    parser.add_argument('output', help='output TensorRT engine path')
    parser.add_argument('--device-id', help='`the CUDA device id', default=0)
    parser.add_argument(
        '--calib-file',
        help='`the calibration data used to calibrate engine to int8',
        default=None)
    parser.add_argument(
        '--log-level',
        help='set log level',
        default='INFO',
        choices=list(logging._nameToLevel.keys()))
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    logging.basicConfig(
        format='%(asctime)s,%(name)s %(levelname)-8s'
        ' [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(args.log_level)

    deploy_cfg_path = args.deploy_cfg
    deploy_cfg = load_config(deploy_cfg_path)[0]
    onnx_path = args.onnx_path
    output_path = args.output
    device_id = args.device_id
    calib_file = args.calib_file

    model_id = 0
    common_params = get_common_config(deploy_cfg)
    model_params = get_model_inputs(deploy_cfg)[model_id]

    final_params = common_params
    final_params.update(model_params)

    int8_param = final_params.get('int8_param', dict())

    if calib_file is not None:
        int8_param['calib_file'] = calib_file
        # do not support partition model calibration for now
        int8_param['model_type'] = 'end2end'

    logging.info(f'onnx2tensorrt: \n\tonnx_path: {onnx_path} '
                 f'\n\tdeploy_cfg: {deploy_cfg_path}')
    try:
        engine = create_trt_engine(
            onnx_path,
            input_shapes=final_params['input_shapes'],
            log_level=get_trt_log_level(),
            fp16_mode=final_params.get('fp16_mode', False),
            int8_mode=final_params.get('int8_mode', False),
            int8_param=int8_param,
            max_workspace_size=final_params.get('max_workspace_size', 0),
            device_id=device_id)

        save_trt_engine(engine, output_path)
        logging.info('onnx2tensorrt success.')
    except Exception as e:
        logging.error(e)
        logging.error('onnx2tensorrt failed.')


if __name__ == '__main__':
    main()
