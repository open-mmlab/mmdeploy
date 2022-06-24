# Copyright (c) OpenMMLab. All rights reserved.

import os
import subprocess

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
        'OCR',
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
            'java', '-cp', 'csrc/mmdeploy/apis/java',
            'demo/java/' + task + '.java'
        ]
        for config in configs:
            model_url = config
            os.system('wget {} && tar xvf {}'.format(model_url,
                                                     model_url.split('/')[-1]))
            model_dir = model_url.split('/')[-1].split('.')[0]
            java_demo_cmd.append(model_dir)
        java_demo_cmd.append('cpu')
        java_demo_cmd.append('demo/resources/human-pose.png')
        export_library_cmd = 'export LD_LIBRARY_PATH=build/lib' + \
            ':${LD_LIBRARY_PATH}'
        print(subprocess.call('export JAVA_HOME=/home/runner/work/mmdeploy/mmdeploy/jdk-18 && export PATH=${JAVA_HOME}/bin:${PATH} && java --version'))
        print(subprocess.call(' '.join(java_demo_cmd)))
        '''
        print(
            subprocess.call(export_library_cmd + ' && ' +
                            ' '.join(java_demo_cmd)))
        '''

if __name__ == '__main__':
    main()
