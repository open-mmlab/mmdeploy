import os
import os.path as osp
import platform
import sys

try:
    from setuptools import find_packages, setup
except ImportError:
    from distutils.core import find_packages, setup

CURDIR = os.path.realpath(os.path.dirname(__file__))
version_file = osp.join(CURDIR, 'mmdeploy_runtime', 'version.py')
package_name = 'mmdeploy_runtime'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def get_platform_name():
    return platform.machine()


def parse_arg_remove_boolean(argv, arg_name):
    arg_value = False
    if arg_name in sys.argv:
        arg_value = True
        argv.remove(arg_name)

    return arg_value


if parse_arg_remove_boolean(sys.argv, '--use-gpu'):
    package_name = package_name + '_gpu'
    if sys.platform == 'win32':
        with open('mmdeploy_runtime/_win_dll_path.py', 'a') as f:
            code = \
                'import os\n' \
                'import sys\n\n' \
                'cuda_bin_dir = ""\n' \
                'if "CUDA_PATH" in os.environ:\n' \
                '    cuda_bin_dir = os.path.join(os.environ["CUDA_PATH"], "bin")\n' \
                'else:\n' \
                '    raise ImportError("Can\'t find environment variable CUDA_PATH")\n' \
                'if sys.version_info >= (3, 8):\n' \
                '    os.add_dll_directory(cuda_bin_dir)\n' \
                'else:\n' \
                '    os.environ["PATH"] = cuda_bin_dir + os.pathsep + os.environ["PATH"]'
            f.write(code)

if __name__ == '__main__':
    setup(
        name=package_name,
        version=get_version(),
        description='OpenMMLab Model Deployment SDK python api',
        author='OpenMMLab',
        author_email='openmmlab@gmail.com',
        keywords='computer vision, model deployment',
        url='https://github.com/open-mmlab/mmdeploy',
        packages=find_packages(),
        include_package_data=True,
        platforms=get_platform_name(),
        package_data={'mmdeploy_runtime': ['*.so*', '*.pyd', '*.pdb']},
        license='Apache License 2.0')
