# docker exec cmd for build
cd /root/workspace

conda activate torch1.10.0
# export libtorch cmake dir, ran example: /opt/conda/envs/torch1.11.0/lib/python3.8/site-packages/torch/share/cmake/Torch
export Torch_DIR=$(python -c "import torch;print(torch.utils.cmake_prefix_path + '/Torch')")
cd mmdeploy
pip install openmim
mim install mmcv-full
pip install -v .
pip install -r requirements/tests.txt
pip install -r requirements/runtime.txt
pip install -r requirements/build.txt

## 校验动态库构建
mkdir -p build_share_libs
cd build_share_libs
cmake .. -DMMDEPLOY_BUILD_SDK=ON -DMMDEPLOY_BUILD_EXAMPLES=ON \
         -DMMDEPLOY_BUILD_SDK_MONOLITHIC=ON -DMMDEPLOY_BUILD_TEST=ON \
         -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON -DMMDEPLOY_BUILD_SDK_JAVA_API=ON \
         -DMMDEPLOY_BUILD_EXAMPLES=ON -DMMDEPLOY_ZIP_MODEL=ON \
         -DMMDEPLOY_TARGET_BACKENDS="ort;pplnn;openvino;ncnn" \
         -DMMDEPLOY_SHARED_LIBS=ON \
         -DTorch_DIR=${Torch_DIR} \
         -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR}
make -j $(nproc) && make install
cd ../

pip install -v .
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
         -DTorch_DIR=${Torch_DIR} \
         -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR}

make -j $(nproc) && make install
cd ../
pip install -v .
# todo 插入校验语句，确认pip install 成功

## 校验不同torch版本下安装
for TORCH_VERSION in 1.9.0 1.10.0 1.11.0 1.12.0
do
    conda activate torch${TORCH_VERSION}
    pip install -v .
    python tools/check_env.py
    # todo 插入校验语句，确认pip install 成功
done
