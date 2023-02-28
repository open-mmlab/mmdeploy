# Copyright (c) OpenMMLab. All rights reserved.

import os

# list of dict: task name and deploy configs.

PARAMS = [
    {
        'task':
        'ImageClassification',
        'configs': [
            'https://media.githubusercontent.com/media/hanrui1sensetime/mmdeploy-javaapi-testdata/master/resnet.tar'  # noqa: E501
        ]
    },
    {
        'task':
        'ObjectDetection',
        'configs': [
            'https://media.githubusercontent.com/media/hanrui1sensetime/mmdeploy-javaapi-testdata/master/mobilessd.tar'  # noqa: E501
        ]
    },
    {
        'task':
        'ImageSegmentation',
        'configs': [
            'https://media.githubusercontent.com/media/hanrui1sensetime/mmdeploy-javaapi-testdata/master/fcn.tar'  # noqa: E501
        ]
    },
    {
        'task':
        'ImageRestorer',
        'configs': [
            'https://media.githubusercontent.com/media/hanrui1sensetime/mmdeploy-javaapi-testdata/master/srcnn.tar'  # noqa: E501
        ]
    },
    {
        'task':
        'Ocr',
        'configs': [
            'https://media.githubusercontent.com/media/hanrui1sensetime/mmdeploy-javaapi-testdata/master/dbnet.tar',  # noqa: E501
            'https://media.githubusercontent.com/media/hanrui1sensetime/mmdeploy-javaapi-testdata/master/crnn.tar'  # noqa: E501
        ]
    },
    {
        'task':
        'PoseDetection',
        'configs': [
            'https://media.githubusercontent.com/media/hanrui1sensetime/mmdeploy-javaapi-testdata/master/litehrnet.tar'  # noqa: E501
        ]
    },
    {
        'task':
        'PoseTracker',
        'configs': [
            'https://media.githubusercontent.com/media/hanrui1sensetime/mmdeploy-javaapi-testdata/master/rtmdet-nano.tar',  # noqa: E501
            'https://media.githubusercontent.com/media/hanrui1sensetime/mmdeploy-javaapi-testdata/master/rtmpose-tiny.tar'  # noqa: E501
        ]
    },
    {
        'task':
        'RotatedDetection',
        'configs': [
            'https://media.githubusercontent.com/media/hanrui1sensetime/mmdeploy-javaapi-testdata/master/gliding-vertex.tar'  # noqa: E501
        ]
    }
]


def main():
    """test java apis and demos.

    Run all java demos for test.
    """

    for params in PARAMS:
        task = params['task']
        configs = params['configs']
        java_command = '\"cpu'
        for config in configs:
            model_url = config
            os.system('wget {} && tar xvf {}'.format(model_url,
                                                     model_url.split('/')[-1]))
            model_dir = model_url.split('/')[-1].split('.')[0]
            java_command += (' ' + model_dir)
        if task in [
                'ImageClassification', 'ObjectDetection', 'ImageSegmentation',
                'ImageRestorer', 'PoseDetection', 'RotatedDetection'
        ]:
            java_command += (' $GITHUB_WORKSPACE/demo' +
                             '/resources/human-pose.jpg\"')
        elif task in ['Ocr']:
            java_command += (' $GITHUB_WORKSPACE/demo' +
                             '/resources/text_det.jpg\"')
        elif task in ['PoseTracker']:
            os.system(
                'wget https://media.githubusercontent.com/media/hanrui1sensetime/mmdeploy-javaapi-testdata/master/dance.mp4'  # noqa: E501
            )
            java_command += ' dance.mp4\"'
        else:
            java_command += '\"'
        print(f'java_command: {java_command}')
        os.system('ant -DtaskName=' + task +
                  ' -DjarDir=${OPENCV_DIR}/build/bin ' +
                  '-DlibDir=${OPENCV_DIR}/build/lib:$GITHUB_WORKSPACE/' +
                  'build/lib -Dcommand=' + java_command)


if __name__ == '__main__':
    main()
