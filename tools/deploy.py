# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os.path as osp
import sys
import traceback
from functools import partial

import mmcv
import torch.multiprocessing as mp
from torch.multiprocessing import Process, set_start_method

from mmdeploy.apis import (create_calib_table, extract_model,
                           get_predefined_partition_cfg, torch2onnx,
                           visualize_model)
from mmdeploy.utils import (Backend, get_backend, get_calib_filename,
                            get_model_inputs, get_onnx_config,
                            get_partition_config, load_config)
from mmdeploy.utils.export_info import dump_info


def parse_args():
    parser = argparse.ArgumentParser(description='Export model to backends.')
    parser.add_argument('deploy_cfg', help='deploy config path')
    parser.add_argument('model_cfg', help='model config path')
    parser.add_argument('checkpoint', help='model checkpoint path')
    parser.add_argument('img', help='image used to convert model model')
    parser.add_argument(
        '--test-img', default=None, help='image used to test model')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--calib-dataset-cfg',
        help='dataset config path used to calibrate.',
        default=None)
    parser.add_argument(
        '--device', help='device used for conversion', default='cpu')
    parser.add_argument(
        '--log-level',
        help='set log level',
        default='INFO',
        choices=list(logging._nameToLevel.keys()))
    parser.add_argument(
        '--show', action='store_true', help='Show detection outputs')
    parser.add_argument(
        '--dump-info', action='store_true', help='Output information for SDK')
    args = parser.parse_args()

    return args


def target_wrapper(target, log_level, ret_value, *args, **kwargs):
    logger = logging.getLogger()
    logging.basicConfig(
        format='%(asctime)s,%(name)s %(levelname)-8s'
        ' [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S')
    logger.level
    logger.setLevel(log_level)
    if ret_value is not None:
        ret_value.value = -1
    try:
        result = target(*args, **kwargs)
        if ret_value is not None:
            ret_value.value = 0
        return result
    except Exception as e:
        logging.error(e)
        traceback.print_exc(file=sys.stdout)


def create_process(name, target, args, kwargs, ret_value=None):
    logging.info(f'{name} start.')
    log_level = logging.getLogger().level

    wrap_func = partial(target_wrapper, target, log_level, ret_value)

    process = Process(target=wrap_func, args=args, kwargs=kwargs)
    process.start()
    process.join()

    if ret_value is not None:
        if ret_value.value != 0:
            logging.error(f'{name} failed.')
            exit()
        else:
            logging.info(f'{name} success.')


def main():
    args = parse_args()
    set_start_method('spawn')
    logging.basicConfig(
        format='%(asctime)s,%(name)s %(levelname)-8s'
        ' [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(args.log_level)

    deploy_cfg_path = args.deploy_cfg
    model_cfg_path = args.model_cfg
    checkpoint_path = args.checkpoint

    # load deploy_cfg
    deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)

    # create work_dir if not
    mmcv.mkdir_or_exist(osp.abspath(args.work_dir))

    if args.dump_info:
        dump_info(deploy_cfg, model_cfg, args.work_dir, pth=checkpoint_path)

    ret_value = mp.Value('d', 0, lock=False)

    # convert onnx
    onnx_save_file = get_onnx_config(deploy_cfg)['save_file']
    create_process(
        'torch2onnx',
        target=torch2onnx,
        args=(args.img, args.work_dir, onnx_save_file, deploy_cfg_path,
              model_cfg_path, checkpoint_path),
        kwargs=dict(device=args.device),
        ret_value=ret_value)

    # convert backend
    onnx_files = [osp.join(args.work_dir, onnx_save_file)]

    # partition model
    partition_cfgs = get_partition_config(deploy_cfg)

    if partition_cfgs is not None:

        if 'partition_cfg' in partition_cfgs:
            partition_cfgs = partition_cfgs.get('partition_cfg', None)
        else:
            assert 'type' in partition_cfgs
            partition_cfgs = get_predefined_partition_cfg(
                deploy_cfg, partition_cfgs['type'])

        origin_onnx_file = onnx_files[0]
        onnx_files = []
        for partition_cfg in partition_cfgs:
            save_file = partition_cfg['save_file']
            save_path = osp.join(args.work_dir, save_file)
            start = partition_cfg['start']
            end = partition_cfg['end']
            dynamic_axes = partition_cfg.get('dynamic_axes', None)

            create_process(
                f'partition model {save_file} with start: {start}, end: {end}',
                extract_model,
                args=(origin_onnx_file, start, end),
                kwargs=dict(dynamic_axes=dynamic_axes, save_file=save_path),
                ret_value=ret_value)

            onnx_files.append(save_path)

    # calib data
    calib_filename = get_calib_filename(deploy_cfg)
    if calib_filename is not None:
        calib_path = osp.join(args.work_dir, calib_filename)

        create_process(
            'calibration',
            create_calib_table,
            args=(calib_path, deploy_cfg_path, model_cfg_path,
                  checkpoint_path),
            kwargs=dict(
                dataset_cfg=args.calib_dataset_cfg,
                dataset_type='val',
                device=args.device),
            ret_value=ret_value)

    backend_files = onnx_files
    # convert backend
    backend = get_backend(deploy_cfg)
    if backend == Backend.TENSORRT:
        model_params = get_model_inputs(deploy_cfg)
        assert len(model_params) == len(onnx_files)

        from mmdeploy.apis.tensorrt import is_available as trt_is_available
        from mmdeploy.apis.tensorrt import onnx2tensorrt
        assert trt_is_available(
        ), 'TensorRT is not available,' \
            + ' please install TensorRT and build TensorRT custom ops first.'
        backend_files = []
        for model_id, model_param, onnx_path in zip(
                range(len(onnx_files)), model_params, onnx_files):
            onnx_name = osp.splitext(osp.split(onnx_path)[1])[0]
            save_file = model_param.get('save_file', onnx_name + '.engine')

            partition_type = 'end2end' if partition_cfgs is None \
                else onnx_name
            create_process(
                f'onnx2tensorrt of {onnx_path}',
                target=onnx2tensorrt,
                args=(args.work_dir, save_file, model_id, deploy_cfg_path,
                      onnx_path),
                kwargs=dict(device=args.device, partition_type=partition_type),
                ret_value=ret_value)

            backend_files.append(osp.join(args.work_dir, save_file))

    elif backend == Backend.NCNN:
        from mmdeploy.apis.ncnn import is_available as is_available_ncnn

        if not is_available_ncnn():
            logging.error('ncnn support is not available.')
            exit(-1)

        from mmdeploy.apis.ncnn import onnx2ncnn, get_output_model_file

        backend_files = []
        for onnx_path in onnx_files:
            model_param_path, model_bin_path = get_output_model_file(
                onnx_path, args.work_dir)
            create_process(
                f'onnx2ncnn with {onnx_path}',
                target=onnx2ncnn,
                args=(onnx_path, model_param_path, model_bin_path),
                kwargs=dict(),
                ret_value=ret_value)
            backend_files += [model_param_path, model_bin_path]

    elif backend == Backend.OPENVINO:
        from mmdeploy.apis.openvino import \
            is_available as is_available_openvino
        assert is_available_openvino(), \
            'OpenVINO is not available, please install OpenVINO first.'

        from mmdeploy.apis.openvino import (onnx2openvino,
                                            get_output_model_file,
                                            get_input_shape_from_cfg)
        openvino_files = []
        for onnx_path in onnx_files:
            model_xml_path = get_output_model_file(onnx_path, args.work_dir)
            input_name = deploy_cfg.onnx_config.input_names
            input_shape = [get_input_shape_from_cfg(deploy_cfg, model_cfg)]
            input_info = dict(zip(input_name, input_shape))
            output_names = deploy_cfg.onnx_config.output_names
            create_process(
                f'onnx2openvino with {onnx_path}',
                target=onnx2openvino,
                args=(input_info, output_names, onnx_path, args.work_dir),
                kwargs=dict(),
                ret_value=ret_value)
            openvino_files.append(model_xml_path)
        backend_files = openvino_files

    elif backend == Backend.PPLNN:
        from mmdeploy.apis.pplnn import \
            is_available as is_available_pplnn
        assert is_available_pplnn(), \
            'PPLNN is not available, please install PPLNN first.'

        from mmdeploy.apis.pplnn import onnx2pplnn
        pplnn_files = []
        for onnx_path in onnx_files:
            algo_file = onnx_path.replace('.onnx', '.json')
            model_inputs = get_model_inputs(deploy_cfg)
            assert 'opt_shape' in model_inputs, 'expect opt_shape '
            'in deploy config for pplnn'
            # PPLNN accepts only 1 input shape for optimization,
            # may get changed in the future
            input_shapes = [model_inputs.opt_shape]
            create_process(
                f'onnx2pplnn with {onnx_path}',
                target=onnx2pplnn,
                args=(algo_file, onnx_path),
                kwargs=dict(device=args.device, input_shapes=input_shapes),
                ret_value=ret_value)
            pplnn_files += [onnx_path, algo_file]
        backend_files = pplnn_files

    if args.test_img is None:
        args.test_img = args.img
    # visualize model of the backend
    create_process(
        f'visualize {backend.value} model',
        target=visualize_model,
        args=(model_cfg_path, deploy_cfg_path, backend_files, args.test_img,
              args.device),
        kwargs=dict(
            backend=backend,
            output_file=osp.join(args.work_dir, f'output_{backend.value}.jpg'),
            show_result=args.show),
        ret_value=ret_value)

    # visualize pytorch model
    create_process(
        'visualize pytorch model',
        target=visualize_model,
        args=(model_cfg_path, deploy_cfg_path, [checkpoint_path],
              args.test_img, args.device),
        kwargs=dict(
            backend=Backend.PYTORCH,
            output_file=osp.join(args.work_dir, 'output_pytorch.jpg'),
            show_result=args.show),
        ret_value=ret_value)

    logging.info('All process success.')


if __name__ == '__main__':
    main()
