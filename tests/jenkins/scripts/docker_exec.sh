## docker run cmd for build
docker run -t ${docker_image}

# docker exec cmd for build
conda activate torch1.8.0
git clone --recursive https://github.com/open-mmlab/mmdeploy.git 
cd mmdeploy

## 校验动态库构建
mkdir -p build_share_libs
cd build_share_libs
cmake .. -DMMDEPLOY_BUILD_SDK=ON -DMMDEPLOY_BUILD_EXAMPLES=ON \
         -DMMDEPLOY_BUILD_SDK_MONOLITHIC=ON -DMMDEPLOY_BUILD_TEST=ON \
         -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON -DMMDEPLOY_BUILD_SDK_JAVA_API=ON \
         -DMMDEPLOY_BUILD_EXAMPLES=ON -DMMDEPLOY_ZIP_MODEL=ON \
         -DMMDEPLOY_TARGET_BACKENDS="ort;pplnn;openvino;ncnn" \
         -DMMDEPLOY_SHARED_LIBS=ON \
         -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR}
make -j $(nproc)
make install
cd ../
pip install -v -e .
# 插入校验语句，确认pip install 成功

## 校验静态库构建
mkdir -p build
cd build 
cmake .. -DMMDEPLOY_BUILD_SDK=ON -DMMDEPLOY_BUILD_EXAMPLES=ON \
         -DMMDEPLOY_BUILD_SDK_MONOLITHIC=ON -DMMDEPLOY_BUILD_TEST=ON \
         -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON -DMMDEPLOY_BUILD_SDK_JAVA_API=ON \
         -DMMDEPLOY_BUILD_EXAMPLES=ON -DMMDEPLOY_ZIP_MODEL=ON \
         -DMMDEPLOY_TARGET_BACKENDS="ort;pplnn;openvino;ncnn" \
         -DMMDEPLOY_SHARED_LIBS=OFF \
         -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR}
make -j $(nproc) && make install
cd ../
pip install -v -e .
# todo 插入校验语句，确认pip install 成功

## 校验不同torch版本下安装
for TORCH_VERSION in (1.9.0 1.10.0 1.11.0 1.12.0)
do
    conda activate torch${TORCH_VERSION}
    pip install -v -e .
    # todo 插入校验语句，确认pip install 成功
done


# docker run cmd for convert
for codebase in ${cobase_list}
do
    log_dir=/data2/regression_log/$(date +%Y%m%d%H%M)/${codebase}
    mkdir -p ${log_dir}
    docker run -it \
        -v /data2/checkpoints/${codebase}:/root/workspace/mmdeploy_checkpoints \
        -v ${log_dir}:/root/workspace/mmdeploy_regression_working_dir \
        -v /data2/benchmark:/root/workspace/mmdeploy_benchmark \
        --name ${codebase}-${docker_image} \
        ${docker_image}
done


# docker exec cmd for convert

for TORCH_VERSION in (1.8.0 1.9.0 1.10.0 1.11.0 1.12.0)
do

    conda activate torch${TORCH_VERSION}

    ## build ${codebase}
    git clone https://github.com/open-mmlab/${codebase_fullname}.git
    /opt/conda/envs/torch${TORCH_VERSION}/bin/mim install ${codebase}

    ## build mmdeploy
    git clone --recursive https://github.com/open-mmlab/mmdeploy.git 
    cd mmdeploy
    mkdir -p build
    cd build 
    cmake .. -DMMDEPLOY_BUILD_SDK=ON -DMMDEPLOY_BUILD_EXAMPLES=ON \
            -DMMDEPLOY_BUILD_SDK_MONOLITHIC=ON -DMMDEPLOY_BUILD_TEST=ON \
            -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON -DMMDEPLOY_BUILD_SDK_JAVA_API=ON \
            -DMMDEPLOY_BUILD_EXAMPLES=ON -DMMDEPLOY_ZIP_MODEL=ON \
            -DMMDEPLOY_TARGET_BACKENDS="ort;pplnn;openvino;ncnn" 
            -DMMDEPLOY_SHARED_LIBS=OFF
            -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR}
    make -j $(nproc) && make install
    cd ../
    pip install -v -e .

    ## start regression   
    pip install -r requirements/tests.txt 
    python ./tools/regression_test.py \
        --codebase ${codebase} \
        --backend ${backend} \
        --work-dir "../mmdeploy_regression_working_dir/torch${TORCH_VERSION}"
    # todo 校验转换是否成功
done