#!/bin/bash

set -e

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
ln -sf /root/workspace/mmdeploy_benchmark ${MMDEPLOY_DIR}/data
ln -sf /root/workspace/jenkins ${MMDEPLOY_DIR}/tests/jenkins

# install tensorrt
export TENSORRT_DIR=/root/workspace/TensorRT
export LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$LD_LIBRARY_PATH

cd /root/workspace
mmdet_version=mmdet
mmdet_branch=master
if [ $repo_version == "v2.0" ]; then
    mmdet_version="mmdet>=3.0.0rc1"
    mmdet_branch=3.x
fi

git clone --depth 1 --branch $mmdet_branch https://github.com/open-mmlab/mmdetection.git

cd ${MMDEPLOY_DIR}
conda deactivate
for PYTHON_VERSION in 3.6 3.7 3.8 3.9; do
    conda create -n mmdeploy-${PYTHON_VERSION} python=${PYTHON_VERSION} -y
    conda activate mmdeploy-${PYTHON_VERSION}
    pip install pyyaml
    pip install -r requirements/build.txt

done

conda deactivate
conda activate mmdeploy-3.6
export PREBUILD_DIR=/root/workspace/prebuild-mmdeploy

python tools/package_tools/mmdeploy_builder.py \
  tools/package_tools/configs/linux_x64.yaml . \
  2>&1 | tee $PREBUILD_DIR/prebuild_log.txt

export onnxruntime_dirname=$(ls | grep  mmdeploy-*-onnxruntime*)
export tensorrt_dirname=$(ls | grep  mmdeploy-*-tensorrt*)

tar -czvf ${PREBUILD_DIR}/${onnxruntime_dirname}.tar.gz ${onnxruntime_dirname}
tar -czvf ${PREBUILD_DIR}/${tensorrt_dirname}.tar.gz ${tensorrt_dirname}

mv ${onnxruntime_dirname} ${tensorrt_dirname} ${PREBUILD_DIR}/

# test prebuilt package
conda deactivate
conda activate torch1.10.0
export PYTHON_VERSION=$(python -V | awk '{print $2}' | awk '{split($0, a, "."); print a[1]a[2]}')

pip install ${TENSORRT_DIR}/python/tensorrt-*-cp${PYTHON_VERSION}-none-linux_x86_64.whl
pip install -U openmim
mim install $mmdet_version
pip install -r requirements/tests.txt


# test onnxruntime
pip install ${PREBUILD_DIR}/${onnxruntime_dirname}/dist/mmdeploy-*-linux_x86_64.whl
pip install ${PREBUILD_DIR}/${onnxruntime_dirname}/sdk/python/mmdeploy_python-*-cp${PYTHON_VERSION}-*-linux_x86_64.whl

test_log_dir=${PREBUILD_DIR}/test_prebuild_onnxruntime
mkdir -p $test_log_dir
python tools/check_env.py 2>&1 | tee $test_log_dir/check_env_log.txt
python tools/regression_test.py --codebase mmdet --models ssd yolov3 --backends onnxruntime --performance \
    --device cpu --work-dir $test_log_dir 2>&1 | tee $test_log_dir/mmdet_regresion_test_log.txt

# must forcely uninstall
pip uninstall mmdeploy mmdeploy_python -y

# test tensorrt
pip install ${PREBUILD_DIR}/${tensorrt_dirname}/dist/mmdeploy-*-linux_x86_64.whl
pip install ${PREBUILD_DIR}/${tensorrt_dirname}/sdk/python/mmdeploy_python-*-cp${PYTHON_VERSION}-*-linux_x86_64.whl

test_log_dir=${PREBUILD_DIR}/test_prebuild_tensorrt
mkdir -p $test_log_dir
python tools/check_env.py 2>&1 | tee $test_log_dir/check_env_log.txt
python tools/regression_test.py --codebase mmdet --models ssd yolov3 --backends tensorrt --performance \
    --device cuda:0 --work-dir $test_log_dir 2>&1 | tee $test_log_dir/mmdet_regresion_test_log.txt

echo "end_time-$(date +%Y%m%d%H%M)"
