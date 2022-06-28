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
            'java', '-cp', 'csrc/mmdeploy/apis/java:demo/java',
            'demo/java/' + task + '.java', 'cpu'
        ]
        for config in configs:
            model_url = config
            os.system('wget {} && tar xvf {}'.format(model_url,
                                                     model_url.split('/')[-1]))
            model_dir = model_url.split('/')[-1].split('.')[0]
            java_demo_cmd.append(model_dir)
        java_demo_cmd.append('/home/runner/work/mmdeploy/mmdeploy/demo' +
                             '/resources/human-pose.jpg')
        java_demo_cmd_str = ' '.join(java_demo_cmd)
        os.system('export JAVA_HOME=/home/runner/work/mmdeploy/mmdeploy/' +
                  'jdk-18 && export PATH=${JAVA_HOME}/bin:${PATH} && java' +
                  ' --version && export LD_LIBRARY_PATH=/home/runner/work/' +
                  'mmdeploy/mmdeploy/build/lib:${LD_LIBRARY_PATH} && ' +
                  java_demo_cmd_str)


if __name__ == '__main__':
    main()
