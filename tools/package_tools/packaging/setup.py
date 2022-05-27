import os
import os.path as osp
import platform

try:
    from setuptools import find_packages, setup
except ImportError:
    from distutils.core import find_packages, setup

CURDIR = os.path.realpath(os.path.dirname(__file__))
version_file = osp.join(CURDIR, 'mmdeploy_python', 'version.py')


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def get_platform_name():
    return platform.machine()


if __name__ == '__main__':
    setup(
        name='mmdeploy_python',
        version=get_version(),
        description='OpenMMLab Model Deployment SDK python api',
        author='OpenMMLab',
        author_email='openmmlab@gmail.com',
        keywords='computer vision, model deployment',
        url='https://github.com/open-mmlab/mmdeploy',
        packages=find_packages(),
        include_package_data=True,
        platforms=get_platform_name(),
        package_data={'mmdeploy_python': ['*.so*', '*.pyd', '*.pdb']},
        license='Apache License 2.0')
