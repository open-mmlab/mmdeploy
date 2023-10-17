# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import sys
from distutils.util import get_platform

import yaml


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='MMDeploy create build config')
    parser.add_argument(
        '--backend',
        required=True,
        type=str,
        help='target backend. Eg: "ort;trt"')
    parser.add_argument(
        '--system',
        required=True,
        type=str,
        help='target system, Eg: windows/linux/jetson')
    parser.add_argument(
        '--build-mmdeploy',
        action='store_true',
        help='whether build mmdeploy runtime package')
    parser.add_argument(
        '--build-sdk', action='store_true', help='whether build sdk c/cpp api')
    parser.add_argument(
        '--sdk-dynamic-net',
        action='store_true',
        help='whether build mmdeploy sdk dynamic net')
    parser.add_argument('--device', type=str, help='target device. Eg: "cpu"')
    parser.add_argument(
        '--shared', action='store_true', help='whether build shared lib')
    parser.add_argument(
        '--build-sdk-monolithic',
        action='store_true',
        help='whether build sdk monolithic')
    parser.add_argument(
        '--build-sdk-python',
        action='store_true',
        help='whether build sdk python api')
    parser.add_argument(
        '--opencv-dir',
        type=str,
        help='opencv path that contains OpenCVConfig.cmake, '
        'default use $ENV{OpenCV_DIR}')
    parser.add_argument(
        '--pplcv-dir',
        type=str,
        help='pplcv path that contains pplcv-config.cmake, '
        'default use $ENV{pplcv_DIR}')
    parser.add_argument(
        '--onnxruntime-dir',
        type=str,
        help='onnxruntime root path, default use $ENV{ONNXRUNTIME_DIR}')
    parser.add_argument(
        '--tensorrt-dir',
        type=str,
        help='tensorrt root path, default use $ENV{TENSORRT_DIR}')
    parser.add_argument(
        '--cudnn-dir',
        type=str,
        help='cudnn root dir, default use $ENV{CUDNN_DIR}')
    parser.add_argument('--cxx11abi', action='store_true', help='new cxxabi')
    parser.add_argument(
        '--output', required=True, type=str, help='output config file path')

    return parser.parse_args()


def generate_config(args):
    config = {}
    cmake_cfg = {}

    # wheel platform tag
    if args.system in ['linux']:
        config['PLATFORM_TAG'] = 'manylinux2014_x86_64'
    elif args.system in ['jetson']:
        config['PLATFORM_TAG'] = 'any'
    else:
        config['PLATFORM_TAG'] = get_platform().replace('-',
                                                        '_').replace('.', '_')

    config['BUILD_MMDEPLOY'] = 'ON' if args.build_mmdeploy else 'OFF'

    # deps for mmdeploy
    cmake_cfg['MMDEPLOY_TARGET_BACKENDS'] = args.backend
    if 'ort' in args.backend:
        if args.onnxruntime_dir:
            cmake_cfg['ONNXRUNTIME_DIR'] = args.onnxruntime_dir
        elif 'ONNXRUNTIME_DIR' in os.environ:
            cmake_cfg['ONNXRUNTIME_DIR'] = os.environ['ONNXRUNTIME_DIR']
        else:
            raise Exception('please provide --onnxruntime-dir')
    if 'trt' in args.backend:
        if args.tensorrt_dir:
            cmake_cfg['TENSORRT_DIR'] = args.tensorrt_dir
        elif 'TENSORRT_DIR' in os.environ:
            cmake_cfg['TENSORRT_DIR'] = os.environ['TENSORRT_DIR']
        else:
            raise Exception('please provide --tensorrt-dir')

        if args.cudnn_dir:
            cmake_cfg['CUDNN_DIR'] = args.cudnn_dir
        elif 'CUDNN_DIR' in os.environ:
            cmake_cfg['CUDNN_DIR'] = os.environ['CUDNN_DIR']
        else:
            raise Exception('please provide --cudnn-dir')

    # deps for mmdeploy-python
    if args.build_sdk:
        cmake_cfg['MMDEPLOY_BUILD_SDK'] = 'ON'
        cmake_cfg[
            'MMDEPLOY_BUILD_SDK_MONOLITHIC'] = 'ON' \
            if args.build_sdk_monolithic else 'OFF'
        cmake_cfg[
            'MMDEPLOY_BUILD_SDK_PYTHON_API'] = 'ON' \
            if args.build_sdk_python else 'OFF'
        cmake_cfg['MMDEPLOY_SHARED_LIBS'] = 'ON' if args.shared else 'OFF'
        cmake_cfg['MMDEPLOY_TARGET_DEVICES'] = args.device
        cmake_cfg[
            'MMDEPLOY_DYNAMIC_BACKEND'] = 'ON' \
            if args.sdk_dynamic_net else 'OFF'
        cmake_cfg['MMDEPLOY_ZIP_MODEL'] = 'ON'

        if args.opencv_dir:
            cmake_cfg['OpenCV_DIR'] = args.opencv_dir
        elif 'OpenCV_DIR' in os.environ:
            cmake_cfg['OpenCV_DIR'] = os.environ['OpenCV_DIR']
        else:
            raise Exception('please provide --opencv-dir')

        if args.device == 'cuda':
            if args.pplcv_dir:
                cmake_cfg['pplcv_DIR'] = args.pplcv_dir
            elif 'pplcv_DIR' in os.environ:
                cmake_cfg['pplcv_DIR'] = os.environ['pplcv_DIR']
            else:
                raise Exception('please provide --pplcv-dir')

        # sdk package template
        if args.system in ['windows', 'linux']:
            name = 'mmdeploy-{mmdeploy_v}-{system}-{machine}'
            if args.cxx11abi:
                name = name + '-cxx11abi'
            if args.device == 'cpu':
                pass
            elif args.device == 'cuda':
                name = '{}-cuda'.format(name) + '{cuda_v}'
            else:
                raise Exception('unsupported device')
            config['BUILD_SDK_NAME'] = name
        elif args.system == 'jetson':
            config['BUILD_SDK_NAME'] = 'mmdeploy-{mmdeploy_v}-jetson-{machine}'
        else:
            raise Exception('unsupported system')
    else:
        cmake_cfg['MMDEPLOY_BUILD_SDK'] = 'OFF'
        cmake_cfg['MMDEPLOY_BUILD_SDK_PYTHON_API'] = 'OFF'

    config['cmake_cfg'] = cmake_cfg
    return config


def main():
    # Parse arguments
    args = parse_arguments()
    print(args)

    config = generate_config(args)
    with open(args.output, 'w') as f:
        yaml.dump(config, f)


if __name__ == '__main__':
    sys.exit(main())
