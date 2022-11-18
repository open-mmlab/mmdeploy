#!/bin/bash
repo_version=${1:-v1.0}
## keep container alive
nohup sleep infinity >sleep.log 2>&1 &

## init conda
__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
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

echo "start_time-$(date +%Y%m%d%H%M)"

export MMDEPLOY_DIR=/root/workspace/mmdeploy
ln -s /root/workspace/mmdeploy_benchmark ${MMDEPLOY_DIR}/data

cd /root/workspace
mmdet_version=mmdet
mmdet_branch=master
if [ $repo_version == "v2.0" ]; then
    mmdet_version="mmdet>=3.0.0rc1"
    mmdet_branch=3.x
fi

git clone --depth 1 --branch $mmdet_branch https://github.com/open-mmlab/mmdetection.git

cd ${MMDEPLOY_DIR}

for PYTHON_VERSION in 3.6 3.7 3.8 3.9; do
    conda create -n python${PYTHON_VERSION} python=${PYTHON_VERSION}
    conda activate python${PYTHON_VERSION}
    pip install pyyaml
    pip install -r requirements/build.txt
    python ./tools/package_tools/mmdeploy_builder.py tools/package_tools/configs/linux_x64.yaml .
    prebuilt_path=/root/workspace/prebuild-mmdeploy/python${PYTHON_VERSION}
    mkdir -p ${prebuilt_path}
    mv mmdeploy-*-onnxruntime* ${prebuilt_path}
    mv mmdeploy-*-tensorrt* ${prebuilt_path}
done

conda activate torch1.10.0
pip install openmim
mim install $mmdet_version
pip install -r requirements/tests.txt
pip install -r requirements/runtime.txt
pip install -r requirements/build.txt

export PYTHON_VERSION=$(python -V | awk '{print $2}' | awk '{split($0, a, "."); print a[1]a[2]}')
export MMDEPLOY_VERSION=$(cat mmdeploy/version.py | grep "__version__ = " | awk '{split($0,a,"= "); print a[2]}' | sed "s/'//g")
cp /root/workspace/prebuild-mmdeploy/python${PYTHON_VERSION:0:1}.${PYTHON_VERSION:1}/* /root/workspace/mmdeploy
python ./tools/package_tools/mmdeploy_builder.py tools/package_tools/configs/linux_x64.yaml . >/root/workspace/log/build.log

pip install mmdeploy-${MMDEPLOY_VERSION}-linux-x86_64-onnxruntime${ONNXRUNTIME_VERSION}/sdk/python/mmdeploy_python-${MMDEPLOY_VERSION}-cp${PYTHON_VERSION}-*-linux_x86_64.whl
pip install mmdeploy-${MMDEPLOY_VERSION}-linux-x86_64-onnxruntime${ONNXRUNTIME_VERSION}/dist/mmdeploy-${MMDEPLOY_VERSION}-*-linux_x86_64.whl
pip install mmdeploy-${MMDEPLOY_VERSION}-linux-x86_64-cuda${CUDA_VERSION}-tensorrt${TENSORRT_VERSION}/dist/mmdeploy-${MMDEPLOY_VERSION}-*-linux_x86_64.whl
pip install mmdeploy-${MMDEPLOY_VERSION}-linux-x86_64-cuda${CUDA_VERSION}-tensorrt${TENSORRT_VERSION}/sdk/python/mmdeploy_python-${MMDEPLOY_VERSION}-cp${PYTHON_VERSION}-*-linux_x86_64.whl

python tools/check_env.py 2>&1 | tee /root/workspace/log/check_env.log

python tools/regression_test.py --codebase mmdet --models ssd --backends onnxruntime tensorrt --performance \
    --device cuda:0 2>&1 | tee /root/workspace/log/test_prebuild.log

echo "end_time-$(date +%Y%m%d%H%M)"
