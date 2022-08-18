# docker exec cmd for build
cd /root/workspace

conda activate torch1.8.0
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
for TORCH_VERSION in 1.9.0 1.10.0 1.11.0 1.12.0
do
    conda activate torch${TORCH_VERSION}
    pip install -v -e .
    # todo 插入校验语句，确认pip install 成功
done