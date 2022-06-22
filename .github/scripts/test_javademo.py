# Copyright (c) OpenMMLab. All rights reserved.

import os
import subprocess

# list of dict: task name and deploy configs.

PARAMS = [
    {
        'task':
        'ImageClassification',
        'configs': [[
            'resnet',
            'configs/mmcls/classification_ncnn_static.py',
            '~/mmclassification/configs/resnet/resnet18_8xb32_in1k.py',
            'https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth',  # noqa: E501
            'demo/resources/human-pose.jpg'
        ]]
    },
    {
        'task':
        'ObjectDetection',
        'configs': [[
            'mobilessd',
            'configs/mmdet/detection/single-stage_ncnn_static-300x300.py',
            '~/mmdetection/configs/ssd/ssdlite_mobilenetv2_scratch_600e_coco.py',  # noqa: E501
            'https://download.openmmlab.com/mmdetection/v2.0/ssd/ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth',  # noqa: E501
            '~/mmdetection/demo/demo.jpg',
        ]]
    },
    {
        'task':
        'ImageSegmentation',
        'configs': [[
            'fcn',
            'configs/mmseg/segmentation/segmentation_ncnn_static-1024x2048.py',  # noqa: E501
            '~/mmsegmentation/configs/fcn/fcn_r18b-d8_512x1024_80k_cityscapes.py',  # noqa: E501
            'https://download.openmmlab.com/mmsegmentation/v0.5/fcn/fcn_r18b-d8_512x1024_80k_cityscapes/fcn_r18b-d8_512x1024_80k_cityscapes_20201225_230143-92c0f445.pth',  # noqa: E501
            '~/mmsegmentation/demo/demo.png',
        ]]
    },
    {
        'task':
        'ImageRestorer',
        'configs': [[
            'srcnn',
            'configs/mmedit/super-resolution/super-resolution_ncnn_dynamic.py',  # noqa: E501
            '~/mmediting/configs/restorers/srcnn/srcnn_x4k915_g1_1000k_div2k.py',  # noqa: E501
            'https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth',  # noqa: E501
            '~/mmclassification/demo/dog.jpg',
        ]]
    },
    {
        'task':
        'OCR',
        'configs': [
            [
                'dbnet',
                'configs/mmocr/text-detection/text-detection_ncnn_static.py',
                '~/mmocr/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py',  # noqa: E501
                'https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth',  # noqa: E501
                '~/mmocr/demo/demo_text_det.jpg',
            ],
            [
                'crnn',
                'configs/mmocr/text-recognition/text-recognition_ncnn_static.py',  # noqa: E501
                '~/mmocr/configs/textrecog/crnn/crnn_academic_dataset.py',
                'https://download.openmmlab.com/mmocr/textrecog/crnn/crnn_academic-a723a1c5.pth',  # noqa: E501
                '~/mmocr/demo/demo_text_recog.jpg',
            ]
        ]
    },
    {
        'task':
        'PoseDetection',
        'configs': [[
            'litehrnet',
            'configs/mmpose/pose-detection_ncnn_static-256x192.py',
            '~/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/litehrnet_18_coco_256x192.py',  # noqa: E501
            'https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet18_coco_256x192-6bace359_20211230.pth',  # noqa: E501
            'demo/resources/human-pose.png',
        ]]
    }
]


def main():
    """test java apis and demos.

    Run all java demos for test.
    """

    for params in PARAMS:
        task = params['task']
        configs = params['configs']
        java_demo_cmd = [
            'java', '-cp', 'csrc/mmdeploy/apis/java/mmdeploy',
            'demo/java/' + task + '.java', 'cpu'
        ]
        for config in configs:
            model_name = config[0]
            deploy_cfg = config[1]
            model_cfg = config[2]
            model = config[3]
            test_img = config[4]
            os.system('wget {}'.format(model))
            deploy_cmd = [
                'python3', 'tools/deploy.py', deploy_cfg, model_cfg,
                model.split('/')[-1], test_img, '--work-dir',
                'dump_info/' + model_name, '--dump-info'
            ]
            print(' '.join(deploy_cmd))
            print(subprocess.call(deploy_cmd))
            java_demo_cmd.append('dump_info/' + model_name)
        java_demo_cmd.append(configs[0][4])
        print(' '.join(java_demo_cmd))
        export_library_cmd = 'export LD_LIBRARY_PATH=build/install/lib\
            :${LD_LIBRARY_PATH}'

        print(subprocess.call(export_library_cmd + '&&' + java_demo_cmd))


if __name__ == '__main__':
    main()
