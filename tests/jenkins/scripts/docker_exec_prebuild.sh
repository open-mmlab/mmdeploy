#!/bin/bash

#### to be removed
apt install -y g++-7 gcc-7
export pplcv_DIR=/root/workspace/ppl.cv/cuda-build/install/lib/cmake/ppl
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.3
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64/:$LD_LIBRARY_PATH

ln -s /root/workspace/mmdeploy_benchmark /root/workspace/mmdeploy/data
cd /root/workspace/mmdeploy

/opt/conda/envs/torch1.10.0/bin/mim install mmdet==v2.20.0
cd /root/workspace
git clone --depth 1 -branch v2.20.0 https://github.com/open-mmlab/mmdetection.git

cd /root/workspace/mmdeploy

conda run --name torch1.10.0 "
    python ./tools/package_tools/mmdeploy_builder.py tools/package_tools/configs/linux_x64.yaml .
" > /root/work/log/build.log 2>&1 &
/opt/conda/envs/torch1.10.0/bin/pip install mmdeploy-0.7.0-linux-x86_64-onnxruntime1.8.1/sdk/python/mmdeploy_python-0.7.0-cp38-none-linux_x86_64.whl
/opt/conda/envs/torch1.10.0/bin/pip install mmdeploy-0.7.0-linux-x86_64-onnxruntime1.8.1/dist/mmdeploy-0.7.0-py3-none-linux_x86_64.whl
/opt/conda/envs/torch1.10.0/bin/pip install mmdeploy-0.7.0-linux-x86_64-cuda11.3-tensorrt8.2.1.8/dist/mmdeploy-0.7.0-py3-none-linux_x86_64.whl
/opt/conda/envs/torch1.10.0/bin/pip install mmdeploy-0.7.0-linux-x86_64-cuda11.3-tensorrt8.2.1.8/sdk/python/mmdeploy_python-0.7.0-cp38-none-linux_x86_64.whl
/opt/conda/envs/torch1.10.0/bin/pip install -r requirements/tests.txt

python tools/check_env.py
mv mmdeploy-*-onnxruntime* ./prebuild-mmdeploy
mv mmdeploy-*-tensorrt* ./prebuild-mmdeploy


conda run --name torch1.10.0 "
    python tools/regression_test.py --codebase mmdet --models ssd --backends onnxruntime tensorrt -p --device cuda
" > /root/workspace/test_prebuild.log 2>&1 &
