#!/bin/bash

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
