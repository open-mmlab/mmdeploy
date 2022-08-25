#!/bin/bash

#### to be removed
apt install g++-7 gcc-7
export pplcv_DIR=/root/workspace/ppl.cv/cuda-build/install/lib/cmake/ppl
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.3

cd /root/workspace/mmdeploy

/opt/conda/envs/torch1.10.0/bin/mim install mmcv
/opt/conda/envs/torch1.10.0/bin/pip install pytest
pip install pytest 

conda run --name torch1.10.0 "
    python ./tools/package_tools/mmdeploy_builder.py tools/package_tools/configs/linux_x64.yaml .
"

mv mmdeploy-*-onnxruntime* ./prebuild-mmdeploy

conda run --name torch1.10.0 "
    pytest -sv ./tests/
"
