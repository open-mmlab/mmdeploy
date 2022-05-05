# Copyright (c) OpenMMLab. All rights reserved.

import argparse
import os
import subprocess

import mmcv

# list of tuple: config, pretrained model, onnx filename
CONFIGS = [
    (
        'mmclassification/configs/vision_transformer/vit-base-p32_ft-64xb64_in1k-384.py',  # noqa: E501
        'https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p32_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-9cea8599.pth',  # noqa: E501
        'vit.onnx',
        'https://pjlab-my.sharepoint.cn/personal/konghuanjun_pjlab_org_cn/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fkonghuanjun%5Fpjlab%5Forg%5Fcn%2FDocuments%2Fmmdeploy%2Donnx2ncnn%2Dci%2Fvit%2Eonnx'  # noqa: E501
    ),  # noqa: E501
    (
        'mmclassification/configs/resnet/resnet50_8xb32_in1k.py',
        'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',  # noqa: E501
        'resnet50.onnx',
        'https://pjlab-my.sharepoint.cn/personal/konghuanjun_pjlab_org_cn/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fkonghuanjun%5Fpjlab%5Forg%5Fcn%2FDocuments%2Fmmdeploy%2Donnx2ncnn%2Dci%2Fresnet50%2Eonnx'  # noqa: E501
    ),  # noqa: E501
    (
        'mmclassification/configs/resnet/resnet18_8xb32_in1k.py',
        'https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth',  # noqa: E501
        'resnet18.onnx',
        'https://pjlab-my.sharepoint.cn/personal/konghuanjun_pjlab_org_cn/_layouts/15/download.aspx?UniqueId=07f4e5ae%2D84d0%2D43d1%2D9412%2D74300a0b9bb8'  # noqa: E501
    ),  # noqa: E501
    (
        'mmclassification/configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py',
        'https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth',  # noqa: E501
        'mobilenet-v2.onnx',
        'https://pjlab-my.sharepoint.cn/personal/konghuanjun_pjlab_org_cn/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fkonghuanjun%5Fpjlab%5Forg%5Fcn%2FDocuments%2Fmmdeploy%2Donnx2ncnn%2Dci%2Fmobilenet%2Dv2%2Eonnx'  # noqa: E501
    )  # noqa: E501
]

TEST_LIST = [2, 3]


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDeploy onnx2ncnn test tool.')
    parser.add_argument('--run', type=bool, help='Execute onnx2ncnn bin.')
    parser.add_argument(
        '--repo-dir', type=str, default='~/', help='mmcls directory.')
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
    mmcv.mkdir_or_exist(args.out)
    for conf in CONFIGS:
        config = os.path.join(args.repo_dir, conf[0])
        model = conf[1]
        convert_cmd = [
            'python3', 'tools/deploy.py',
            'configs/mmcls/classification_ncnn_static.py', config, model,
            'cat-dog.png', '--work-dir', 'work_dir', '--device', 'cpu'
        ]
        print(subprocess.call(convert_cmd))

        move_cmd = [
            'mv', 'work_dir/end2end.onnx',
            os.path.join(args.out, conf[2])
        ]
        print(subprocess.call(move_cmd))


def run(args):
    for idx in TEST_LIST:
        download_url = CONFIGS[idx][3]
        filename = CONFIGS[idx][2]
        download_cmd = ['wget', download_url, '-o', filename]
        subprocess.call(download_cmd)

        convert_cmd = ['./onnx2ncnn', filename, 'onnx.param', 'onnx.bin']
        subprocess.call(convert_cmd)


def main():
    """test `onnx2ncnn.cpp`

    First generate onnx model then convert it with `onnx2ncnn`.
    """
    args = parse_args()
    if args.generate_onnx:
        generate_onnx(args)
    if args.run:
        run(args)


if __name__ == '__main__':
    main()
