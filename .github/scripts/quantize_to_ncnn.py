# Copyright (c) OpenMMLab. All rights reserved.

import os
import subprocess

# list of tuple: onnx filename, download_url and model_config.

CONFIGS = [
    [
        'hrnet.onnx',
        'https://media.githubusercontent.com/media/tpoisonooo/mmdeploy-onnx2ncnn-testdata/main/hrnet.onnx',  # noqa: E501
        '~/mmclassification/configs/hrnet/hrnet-w18_4xb32_in1k.py',
    ],
    [
        'resnet18.onnx',
        'https://media.githubusercontent.com/media/tpoisonooo/mmdeploy-onnx2ncnn-testdata/main/resnet18.onnx',  # noqa: E501
        '~/mmclassification/configs/resnet/resnet18_8xb16_cifar10.py',
    ],
    [
        'mobilenet-v2.onnx',
        'https://media.githubusercontent.com/media/tpoisonooo/mmdeploy-onnx2ncnn-testdata/main/mobilenet-v2.onnx',  # noqa: E501
        '~/mmclassification/configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py',
    ]
]


def prepare_dataset():
    DATASET = (
        'dataset',
        'https://media.githubusercontent.com/media/tpoisonooo/mmdeploy-onnx2ncnn-testdata/main/dataset.tar'  # noqa: E501
    )  # noqa: E501
    os.system('wget {}'.format(DATASET[1]))
    os.system('tar xvf dataset.tar')
    return DATASET[0]


def main():
    """test `tools/onnx2ncnn_quant_table.py`

    First quantize onnx model to ncnn with ppq.
    """
    data_dir = prepare_dataset()

    for conf in CONFIGS:
        model = conf[0]
        os.system('wget {}'.format(conf[1]))
        model_cfg = conf[2]
        deploy_cfg = 'configs/mmcls/classification_ncnn-int8_static.py'
        quant_cmd = [
            'python3', 'tools/onnx2ncnn_quant_table.py', '--onnx', model,
            '--deploy-cfg', deploy_cfg, '--model-cfg', model_cfg, '--out-onnx',
            'quant.onnx', '--out-table', 'ncnn.table', '--image-dir', data_dir
        ]
        print(' '.join(quant_cmd))
        print(subprocess.call(quant_cmd))


if __name__ == '__main__':
    main()
