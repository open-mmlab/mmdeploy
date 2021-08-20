import argparse
import logging
import os.path as osp
import subprocess
from functools import partial

import mmcv
import torch.multiprocessing as mp
from torch.multiprocessing import Process, set_start_method

from mmdeploy.apis import (assert_cfg_valid, create_calib_table, extract_model,
                           inference_model, torch2onnx)
from mmdeploy.apis.utils import get_split_cfg


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
    args = parser.parse_args()

    return args


def target_wrapper(target, log_level, *args, **kwargs):
    logger = logging.getLogger()
    logger.level
    logger.setLevel(log_level)
    return target(*args, **kwargs)


def create_process(name, target, args, kwargs, ret_value=None):
    logging.info(f'{name} start.')
    log_level = logging.getLogger().level

    wrap_func = partial(target_wrapper, target, log_level)

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

    logger = logging.getLogger()
    logger.setLevel(args.log_level)

    deploy_cfg_path = args.deploy_cfg
    model_cfg_path = args.model_cfg
    checkpoint_path = args.checkpoint

    # load deploy_cfg
    deploy_cfg = mmcv.Config.fromfile(deploy_cfg_path)
    assert_cfg_valid(deploy_cfg, model_cfg_path)

    # create work_dir if not
    mmcv.mkdir_or_exist(osp.abspath(args.work_dir))

    ret_value = mp.Value('d', 0, lock=False)

    # convert onnx
    onnx_save_file = deploy_cfg['pytorch2onnx']['save_file']
    create_process(
        'torch2onnx',
        target=torch2onnx,
        args=(args.img, args.work_dir, onnx_save_file, deploy_cfg_path,
              model_cfg_path, checkpoint_path),
        kwargs=dict(device=args.device, ret_value=ret_value),
        ret_value=ret_value)

    # convert backend
    onnx_files = [osp.join(args.work_dir, onnx_save_file)]

    # split model
    apply_marks = deploy_cfg.get('apply_marks', False)
    if apply_marks:
        assert hasattr(deploy_cfg, 'split_params')
        split_params = deploy_cfg['split_params']

        if 'split_cfg' in split_params:
            split_cfgs = split_params.get('split_cfg', None)
        else:
            assert 'split_type' in split_params
            split_cfgs = get_split_cfg(deploy_cfg['codebase'],
                                       split_params['split_type'])

        origin_onnx_file = onnx_files[0]
        onnx_files = []
        for split_cfg in split_cfgs:
            save_file = split_cfg['save_file']
            save_path = osp.join(args.work_dir, save_file)
            start = split_cfg['start']
            end = split_cfg['end']
            dynamic_axes = split_cfg.get('dynamic_axes', None)

            create_process(
                f'split model {save_file} with start: {start}, end: {end}',
                extract_model,
                args=(origin_onnx_file, start, end),
                kwargs=dict(
                    dynamic_axes=dynamic_axes,
                    save_file=save_path,
                    ret_value=ret_value),
                ret_value=ret_value)

            onnx_files.append(save_path)

    # calib data
    create_calib = deploy_cfg.get('create_calib', False)
    if create_calib:
        calib_params = deploy_cfg.get('calib_params', dict())
        calib_file = calib_params.get('calib_file', 'calib_file.h5')
        calib_file = osp.join(args.work_dir, calib_file)

        create_process(
            'calibration',
            create_calib_table,
            args=(calib_file, deploy_cfg_path, model_cfg_path,
                  checkpoint_path),
            kwargs=dict(
                dataset_cfg=args.calib_dataset_cfg,
                dataset_type='val',
                device=args.device,
                ret_value=ret_value),
            ret_value=ret_value)

    backend_files = onnx_files
    # convert backend
    backend = deploy_cfg.get('backend', 'default')
    if backend == 'tensorrt':
        assert hasattr(deploy_cfg, 'tensorrt_params')
        tensorrt_params = deploy_cfg['tensorrt_params']
        model_params = tensorrt_params.get('model_params', [])
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

            split_type = 'end2end' if not apply_marks else onnx_name
            create_process(
                f'onnx2tensorrt of {onnx_path}',
                target=onnx2tensorrt,
                args=(args.work_dir, save_file, model_id, deploy_cfg_path,
                      onnx_path),
                kwargs=dict(
                    device=args.device,
                    split_type=split_type,
                    ret_value=ret_value),
                ret_value=ret_value)

            backend_files.append(osp.join(args.work_dir, save_file))

    elif backend == 'ncnn':
        from mmdeploy.apis.ncnn import get_onnx2ncnn_path
        from mmdeploy.apis.ncnn import is_available as is_available_ncnn

        if not is_available_ncnn():
            logging.error('ncnn support is not available.')
            exit(-1)

        onnx2ncnn_path = get_onnx2ncnn_path()

        backend_files = []
        for onnx_path in onnx_files:
            onnx_name = osp.splitext(osp.split(onnx_path)[1])[0]
            save_param = onnx_name + '.param'
            save_bin = onnx_name + '.bin'

            save_param = osp.join(args.work_dir, save_param)
            save_bin = osp.join(args.work_dir, save_bin)

            subprocess.call([onnx2ncnn_path, onnx_path, save_param, save_bin])

            backend_files += [save_param, save_bin]

    if args.test_img is None:
        args.test_img = args.img
    # visualize model of the backend
    create_process(
        f'visualize {backend} model',
        target=inference_model,
        args=(model_cfg_path, deploy_cfg_path, backend_files, args.test_img),
        kwargs=dict(
            device=args.device,
            output_file=f'output_{backend}.jpg',
            show_result=args.show,
            ret_value=ret_value),
        ret_value=ret_value)

    # visualize pytorch model
    create_process(
        'visualize pytorch model',
        target=inference_model,
        args=(model_cfg_path, deploy_cfg_path, [checkpoint_path],
              args.test_img),
        kwargs=dict(
            device=args.device,
            backend='pytorch',
            output_file='output_pytorch.jpg',
            show_result=args.show,
            ret_value=ret_value),
        ret_value=ret_value)

    logging.info('All process success.')


if __name__ == '__main__':
    main()
