# Copyright (c) OpenMMLab. All rights reserved.

import argparse
import os
import subprocess

# list of tuple: config, pretrained model, onnx filename
CONFIGS = [
    (
        'mmpretrain/configs/vision_transformer/vit-base-p32_ft-64xb64_in1k-384.py',  # noqa: E501
        'https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p32_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-9cea8599.pth',  # noqa: E501
        'vit.onnx'),
    (
        'mmpretrain/configs/resnet/resnet50_8xb32_in1k.py',
        'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',  # noqa: E501
        'resnet50.onnx',
    ),
    (
        'mmpretrain/configs/resnet/resnet18_8xb32_in1k.py',
        'https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth',  # noqa: E501
        'resnet18.onnx',
        'https://github.com/open-mmlab/mmdeploy/releases/download/v0.1.0/resnet18.onnx',  # noqa: E501
    ),
    (
        'mmpretrain/configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py',
        'https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth',  # noqa: E501
        'mobilenet-v2.onnx',
        'https://github.com/open-mmlab/mmdeploy/releases/download/v0.1.0/mobilenet-v2.onnx',  # noqa: E501
    )
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDeploy onnx2ncnn test tool.')
    parser.add_argument(
        '--run', type=bool, help='Execute mmdeploy_onnx2ncnn bin.')
    parser.add_argument(
        '--repo-dir', type=str, default='~/', help='mmpretrain directory.')
    parser.add_argument(
        '--out',
        type=str,
        default='onnx_output',
        help='onnx model output directory.')
    parser.add_argument(
        '--generate-onnx', type=bool, help='Generate onnx model.')
    args = parser.parse_args()
    return args


def generate_onnx(args):
    import mmengine
    mmengine.mkdir_or_exist(args.out)
    for conf in CONFIGS:
        config = os.path.join(args.repo_dir, conf[0])
        model = conf[1]
        convert_cmd = [
            'python', 'tools/deploy.py',
            'configs/mmpretrain/classification_ncnn_static.py', config, model,
            'cat-dog.png', '--work-dir', 'work_dir', '--device', 'cpu'
        ]
        print(subprocess.call(convert_cmd))

        move_cmd = [
            'mv', 'work_dir/end2end.onnx',
            os.path.join(args.out, conf[2])
        ]
        print(subprocess.call(move_cmd))


def run(args):
    for conf in CONFIGS:
        if len(conf) < 4:
            continue
        download_url = conf[3]
        filename = conf[2]
        download_cmd = ['wget', download_url]
        # show processbar
        os.system(' '.join(download_cmd))

        convert_cmd = [
            './mmdeploy_onnx2ncnn', filename, 'onnx.param', 'onnx.bin'
        ]
        subprocess.run(convert_cmd, capture_output=True, check=True)


def main():
    """test `onnx2ncnn.cpp`

    First generate onnx model then convert it with `mmdeploy_onnx2ncnn`.
    """
    args = parse_args()
    if args.generate_onnx:
        generate_onnx(args)
    if args.run:
        run(args)


if __name__ == '__main__':
    main()
