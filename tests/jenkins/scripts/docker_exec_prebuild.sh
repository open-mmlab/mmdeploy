#!/bin/bash

## keep container alive
nohup sleep infinity > sleep.log 2>&1 &

## init conda
__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        . "/opt/conda/etc/profile.d/conda.sh"
    else
        export PATH="/opt/conda/bin:$PATH"
    fi
fi
unset __conda_setup

#### to be removed
apt install -y g++-7 gcc-7
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.3
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64/:$LD_LIBRARY_PATH
cp -r cuda/include/cudnn* /usr/local/cuda-11.3/include/
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH/\/root\/workspace\/libtorch\/lib:/}

ln -s /root/workspace/mmdeploy_benchmark /root/workspace/mmdeploy/data

# /opt/conda/envs/torch1.10.0/bin/mim install mmdet==v2.20.0
cd /root/workspace
git clone https://github.com/open-mmlab/mmdetection.git

cd /root/workspace/mmdeploy

## use activate
conda activate torch1.10.0
mim install mmdet
pip install -r requirements/tests.txt
pip install -r requirements/runtime.txt 
pip install -r requirements/build.txt
python ./tools/package_tools/mmdeploy_builder.py tools/package_tools/configs/linux_x64.yaml . > /root/workspace/log/build.log
pip install mmdeploy-0.7.0-linux-x86_64-onnxruntime1.8.1/sdk/python/mmdeploy_python-0.7.0-cp38-none-linux_x86_64.whl
pip install mmdeploy-0.7.0-linux-x86_64-onnxruntime1.8.1/dist/mmdeploy-0.7.0-py3-none-linux_x86_64.whl
pip install mmdeploy-0.7.0-linux-x86_64-cuda11.3-tensorrt8.2.1.8/dist/mmdeploy-0.7.0-py3-none-linux_x86_64.whl
pip install mmdeploy-0.7.0-linux-x86_64-cuda11.3-tensorrt8.2.1.8/sdk/python/mmdeploy_python-0.7.0-cp38-none-linux_x86_64.whl

python tools/check_env.py > /root/workspace/log/check_env.log
mv mmdeploy-*-onnxruntime* /root/workspace/prebuild-mmdeploy 
mv mmdeploy-*-tensorrt* /root/workspace/prebuild-mmdeploy

python tools/regression_test.py --codebase mmdet --models ssd --backends onnxruntime tensorrt -p --device cuda 2>&1 | tee /root/workspace/log/test_prebuild.log