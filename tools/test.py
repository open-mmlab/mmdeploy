# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

from mmcv import DictAction
from mmcv.parallel import MMDataParallel

from mmdeploy.apis import build_task_processor
from mmdeploy.utils.config_utils import load_config
from mmdeploy.utils.device import parse_device_id
from mmdeploy.utils.timer import TimeCounter


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDeploy test (and eval) a backend.')
    parser.add_argument('deploy_cfg', help='Deploy config path')
    parser.add_argument('model_cfg', help='Model config path')
    parser.add_argument(
        '--model', type=str, nargs='+', help='Input model files.')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the codebase and the '
        'dataset, e.g., "bbox", "segm", "proposal" for COCO, and "mAP", '
        '"recall" for PASCAL VOC in mmdet; "accuracy", "precision", "recall", '
        '"f1_score", "support" for single label dataset, and "mAP", "CP", "CR"'
        ', "CF1", "OP", "OR", "OF1" for multi-label dataset in mmcls')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--device', help='device used for conversion', default='cpu')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--metric-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--log2file',
        type=str,
        help='log evaluation results and speed to file',
        default=None)
    parser.add_argument(
        '--json-file',
        type=str,
        help='log evaluation results to json file',
        default='./results.json')
    parser.add_argument(
        '--speed-test', action='store_true', help='activate speed test')
    parser.add_argument(
        '--warmup',
        type=int,
        help='warmup before counting inference elapse, require setting '
        'speed-test first',
        default=10)
    parser.add_argument(
        '--log-interval',
        type=int,
        help='the interval between each log, require setting '
        'speed-test first',
        default=100)
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='the batch size for test, would override `samples_per_gpu`'
        'in  data config.')
    parser.add_argument(
        '--uri',
        help='Remote ipv4:port or ipv6:port for inference on edge device.')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')
    deploy_cfg_path = args.deploy_cfg
    model_cfg_path = args.model_cfg

    # load deploy_cfg
    deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)

    # merge options for model cfg
    if args.cfg_options is not None:
        model_cfg.merge_from_dict(args.cfg_options)

    task_processor = build_task_processor(model_cfg, deploy_cfg, args.device)

    # prepare the dataset loader
    dataset_type = 'test'
    dataset = task_processor.build_dataset(model_cfg, dataset_type)
    # override samples_per_gpu that used for training
    model_cfg.data['samples_per_gpu'] = args.batch_size
    data_loader = task_processor.build_dataloader(
        dataset,
        samples_per_gpu=model_cfg.data.samples_per_gpu,
        workers_per_gpu=model_cfg.data.workers_per_gpu)

    # load the model of the backend
    model = task_processor.init_backend_model(args.model, uri=args.uri)

    is_device_cpu = (args.device == 'cpu')
    device_id = None if is_device_cpu else parse_device_id(args.device)

    destroy_model = model.destroy
    model = MMDataParallel(model, device_ids=[device_id])
    # The whole dataset test wrapped a MMDataParallel class outside the module.
    # As mmcls.apis.test.py single_gpu_test defined, the MMDataParallel needs
    # a 'CLASSES' attribute. So we ensure the MMDataParallel class has the same
    # CLASSES attribute as the inside module.
    if hasattr(model.module, 'CLASSES'):
        model.CLASSES = model.module.CLASSES
    if args.speed_test:
        with_sync = not is_device_cpu

        with TimeCounter.activate(
                warmup=args.warmup,
                log_interval=args.log_interval,
                with_sync=with_sync,
                file=args.log2file,
                batch_size=model_cfg.data.samples_per_gpu):
            outputs = task_processor.single_gpu_test(model, data_loader,
                                                     args.show, args.show_dir)
    else:
        outputs = task_processor.single_gpu_test(model, data_loader, args.show,
                                                 args.show_dir)
    json_dir, _ = os.path.split(args.json_file)
    if json_dir:
        os.makedirs(json_dir, exist_ok=True)
    task_processor.evaluate_outputs(
        model_cfg,
        outputs,
        dataset,
        args.metrics,
        args.out,
        args.metric_options,
        args.format_only,
        args.log2file,
        json_file=args.json_file)
    # only effective when the backend requires explicit clean-up (e.g. Ascend)
    destroy_model()


if __name__ == '__main__':
    main()
