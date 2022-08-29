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

## func
function getFullName() {
    local codebase_=$1
    codebase_fullname=""
    if [ "$codebase_" = "mmdet" ]; then codebase_fullname="mmdetection"; fi 
    if [ "$codebase_" = "mmcls" ]; then codebase_fullname="mmclassification"; fi 
    if [ "$codebase_" = "mmdet3d" ]; then codebase_fullname="mmdetection3d"; fi 
    if [ "$codebase_" = "mmedit" ]; then codebase_fullname="mmediting"; fi 
    if [ "$codebase_" = "mmocr" ]; then codebase_fullname="mmocr"; fi 
    if [ "$codebase_" = "mmpose" ]; then codebase_fullname="mmpose"; fi 
    if [ "$codebase_" = "mmrotate" ]; then codebase_fullname="mmrotate"; fi 
    if [ "$codebase_" = "mmseg" ]; then codebase_fullname="mmsegmentation"; fi 
}

## parameters
export codebase=$1
getFullName $codebase
# backends=$2

## clone ${codebase}
cd /root/workspace
git clone https://github.com/open-mmlab/${codebase_fullname}.git

#### wait to be removed
cp -r cuda/include/cudnn* /usr/local/cuda-11.3/include/
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATHa
export LD_LIBRARY_PATH=/root/workspace/TensorRT-8.2.1.8/lib:/root/workspace/onnxruntime-linux-x64-1.8.1/lib:/usr/local/cuda-11.3/lib64/:/root/workspace/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

## build mmdeploy
ln -s /root/workspace/mmdeploy_benchmark /root/workspace/mmdeploy/data
cd mmdeploy
mkdir -p build
cd build
# todo: add openvino
cmake .. -DMMDEPLOY_BUILD_SDK=ON \
        -DMMDEPLOY_BUILD_EXAMPLES=ON \
        -DMMDEPLOY_BUILD_SDK_MONOLITHIC=ON -DMMDEPLOY_BUILD_TEST=ON \
        -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON -DMMDEPLOY_BUILD_SDK_JAVA_API=ON \
        -DMMDEPLOY_BUILD_EXAMPLES=ON -DMMDEPLOY_ZIP_MODEL=ON \
        -DMMDEPLOY_TARGET_BACKENDS="trt;ort;ncnn" \
        -DMMDEPLOY_SHARED_LIBS=OFF \
        -DTENSORRT_DIR=${TENSORRT_DIR} \
        -DCUDNN_DIR=${CUDNN_DIR} \
        -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
        -Dncnn_DIR=${ncnn_DIR} \
        -Dpplcv_DIR=/root/workspace/ppl.cv/cuda-build/install/lib/cmake/ppl \
        -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" 

make -j $(nproc) && make install

# ## start convert
# for TORCH_VERSION in 1.10.0 1.11.0
# do
#     /opt/conda/envs/torch${TORCH_VERSION}/bin/pip install -v -e .
#     /opt/conda/envs/torch${TORCH_VERSION}/bin/pip install -r requirements/tests.txt requirements/build.txt requirements/runtime.txt 
#     ## build ${codebase}
#     /opt/conda/envs/torch${TORCH_VERSION}/bin/mim install ${codebase}
#     cd ../${codebase_fullname} && /opt/conda/bin/pip install -v -e . && cd /root/workspace/mmdeploy
#     ## start regression 
#     mkdir -p root/workspace/mmdeploy_regression_working_dir/${codebase}/torch${TORCH_VERSION}
#     conda run --name torch${TORCH_VERSION} "
#         python ./tools/regression_test.py \
#             --codebase ${codebase} \
#             --work-dir "../mmdeploy_regression_working_dir/${codebase}/torch${TORCH_VERSION}" \
#             --performance
#     " > root/workspace/mmdeploy_regression_working_dir/${codebase}/torch${TORCH_VERSION}/convert.log 2>&1 &
# done


## use activate

cd /root/workspace/mmdeploy
for TORCH_VERSION in 1.10.0 1.11.0
do

    conda activate torch${TORCH_VERSION}
    pip install -v -e .
    pip install -r requirements/tests.txt
    pip install -r requirements/runtime.txt 
    pip install -r requirements/build.txt
    ## build ${codebase}
    mim install ${codebase}
    pip install -v -e /root/workspace/${codebase_fullname} 
    ## start regression
    log_dir=/root/workspace/mmdeploy_regression_working_dir/${codebase}/torch${TORCH_VERSION}
    log_path=${log_dir}/convert.log
    mkdir -p ${log_dir}
    python ./tools/regression_test.py \
        --codebase ${codebase} \
        --work-dir ${log_dir} \
        --backends tensorrt onnxruntime ncnn \
        --performance > ${log_path}
done