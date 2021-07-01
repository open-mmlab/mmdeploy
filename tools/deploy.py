import argparse
import logging
import os.path as osp

import mmcv
import torch.multiprocessing as mp
from torch.multiprocessing import Process, set_start_method

from mmdeploy.apis import torch2onnx


def parse_args():
    parser = argparse.ArgumentParser(description='Export model to backend.')
    parser.add_argument('deploy_cfg', help='deploy config path')
    parser.add_argument('model_cfg', help='model config path')
    parser.add_argument('checkpoint', help='model checkpoint path')
    parser.add_argument(
        'img', help='image used to convert model and test model')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--device', help='device used for conversion', default='cpu')
    parser.add_argument(
        '--log-level',
        help='set log level',
        default='INFO',
        choices=list(logging._nameToLevel.keys()))
    args = parser.parse_args()

    return args


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
    if not isinstance(deploy_cfg, mmcv.Config):
        raise TypeError('deploy_cfg must be a filename or Config object, '
                        f'but got {type(deploy_cfg)}')

    # create work_dir if not
    mmcv.mkdir_or_exist(osp.abspath(args.work_dir))

    ret_value = mp.Value('d', 0, lock=False)

    # convert onnx
    logging.info('start torch2onnx conversion.')
    onnx_save_file = deploy_cfg['pytorch2onnx']['save_file']
    process = Process(
        target=torch2onnx,
        args=(args.img, args.work_dir, onnx_save_file, deploy_cfg_path,
              model_cfg_path, checkpoint_path),
        kwargs=dict(device=args.device, ret_value=ret_value))
    process.start()
    process.join()

    if ret_value.value != 0:
        logging.error('torch2onnx failed.')
        exit()
    else:
        logging.info('torch2onnx success.')

    # convert backend
    onnx_pathes = [osp.join(args.work_dir, onnx_save_file)]

    backend = deploy_cfg.get('backend', 'default')
    if backend == 'tensorrt':
        assert hasattr(deploy_cfg, 'tensorrt_param')
        tensorrt_param = deploy_cfg['tensorrt_param']
        model_params = tensorrt_param.get('model_params', [])
        assert len(model_params) == len(onnx_pathes)

        logging.info('start onnx2tensorrt conversion.')
        from mmdeploy.apis.tensorrt import onnx2tensorrt
        for model_id, model_param, onnx_path in zip(
                range(len(onnx_pathes)), model_params, onnx_pathes):
            onnx_name = osp.splitext(osp.split(onnx_path)[1])[0]
            save_file = model_param.get('save_file', onnx_name + '.engine')
            process = Process(
                target=onnx2tensorrt,
                args=(args.work_dir, save_file, model_id, deploy_cfg_path,
                      onnx_path),
                kwargs=dict(device=args.device, ret_value=ret_value))
            process.start()
            process.join()

            if ret_value.value != 0:
                logging.error('onnx2tensorrt failed.')
                exit()
            else:
                logging.info('onnx2tensorrt success.')

    logging.info('All process success.')


if __name__ == '__main__':
    main()
