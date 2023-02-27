# Copyright (c) OpenMMLab. All rights reserved.

import os

# list of dict: task name and deploy configs.

PARAMS = [
    {
        'task':
        'ImageClassification',
        'configs': [
            'https://media.githubusercontent.com/media/hanrui1sensetime/mmdeploy-javaapi-testdata/master/resnet.tar'  # noqa: E501
        ],
        'input_type':
        'image'
    },
    {
        'task':
        'ObjectDetection',
        'configs': [
            'https://media.githubusercontent.com/media/hanrui1sensetime/mmdeploy-javaapi-testdata/master/mobilessd.tar'  # noqa: E501
        ],
        'input_type':
        'image'
    },
    {
        'task':
        'ImageSegmentation',
        'configs': [
            'https://media.githubusercontent.com/media/hanrui1sensetime/mmdeploy-javaapi-testdata/master/fcn.tar'  # noqa: E501
        ],
        'input_type':
        'image'
    },
    {
        'task':
        'ImageRestorer',
        'configs': [
            'https://media.githubusercontent.com/media/hanrui1sensetime/mmdeploy-javaapi-testdata/master/srcnn.tar'  # noqa: E501
        ],
        'input_type':
        'image'
    },
    {
        'task':
        'Ocr',
        'configs': [
            'https://media.githubusercontent.com/media/hanrui1sensetime/mmdeploy-javaapi-testdata/master/dbnet.tar',  # noqa: E501
            'https://media.githubusercontent.com/media/hanrui1sensetime/mmdeploy-javaapi-testdata/master/crnn.tar'  # noqa: E501
        ],
        'input_type':
        'text-image'
    },
    {
        'task':
        'PoseDetection',
        'configs': [
            'https://media.githubusercontent.com/media/hanrui1sensetime/mmdeploy-javaapi-testdata/master/litehrnet.tar'  # noqa: E501
        ],
        'input_type':
        'image'
    },
    {
        'task':
        'PoseTracker',
        'configs': [
            'https://media.githubusercontent.com/media/hanrui1sensetime/mmdeploy-javaapi-testdata/master/rtmdet-nano.tar',  # noqa: E501
            'https://media.githubusercontent.com/media/hanrui1sensetime/mmdeploy-javaapi-testdata/master/rtmpose-tiny.tar'  # noqa: E501
        ],
        'input_type':
        'video'
    }
]


def main():
    """test java apis and demos.

    Run all java demos for test.
    """

    for params in PARAMS:
        task = params['task']
        configs = params['configs']
        input_type = params['input_type']
        java_command = '\"cpu'
        for config in configs:
            model_url = config
            os.system('wget {} && tar xvf {}'.format(model_url,
                                                     model_url.split('/')[-1]))
            model_dir = model_url.split('/')[-1].split('.')[0]
            java_command += (' ' + model_dir)
        if input_type == 'image':
            java_command += (' /home/runner/work/mmdeploy/mmdeploy/demo' +
                             '/resources/human-pose.jpg\"')
        elif input_type == 'text-image':
            java_command += (' /home/runner/work/mmdeploy/mmdeploy/demo' +
                             '/resources/text_det.jpg\"')
        elif input_type == 'video':
            os.system(
                'wget https://media.githubusercontent.com/media/hanrui1sensetime/mmdeploy-javaapi-testdata/master/dance.mp4'  # noqa: E501
            )
            java_command += ' dance.mp4\"'
        else:
            java_command += '\"'
        print(f'java_command: {java_command}')
        os.system(
            'find /home/runner/work/opencv/build/lib/libopencv_java470.so')
        os.system(
            'ant -DtaskName=' + task + ' -DjarDir=${OPENCV_DIR}/build/bin ' +
            '-DlibDir=${OPENCV_DIR}/build/lib:/home/runner/work/mmdeploy/' +
            'mmdeploy/build/lib -Dcommand=' + java_command)


if __name__ == '__main__':
    main()
