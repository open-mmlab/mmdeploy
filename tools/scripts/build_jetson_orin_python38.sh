#! /bin/bash

# check python version is 3.8 or not
check_python_38(){
    MAJOR=`python3 --version | awk '{print $2}' | awk -F . '{print $1}'`
    MINOR=`python3 --version | awk '{print $2}' | awk -F . '{print $2}'`

    if [ ${MAJOR} -ne 3 ];then
        echo 'This script needs python==3.8 +_+'
        exit 0
    fi
    if [ ${MINOR} -ne 8 ];then
        echo 'This script needs python==3.8 +_+'
        exit 0
    fi
}

install_torch() {
    version=`python3 -c "import torch; print(torch.__version__)"`
    if [ -n "$version" ];then
        return 0
    fi
    TORCH_WHL="torch-1.11.0-cp38-cp38-linux_aarch64.whl"
    if [ ! -e "${TORCH_WHL}" ];then
        wget -q --show-progress https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i0q926hy4imzs2ph.whl -O ${TORCH_WHL}
    fi
    python3 -m pip install ${TORCH_WHL}
    python3 -m pip install numpy
    sudo apt install libopenblas-dev -y
    python3 -c "import torch; print(torch.__version__)"
}

install_torchvision() {
    version=`python3 -c "import torchvision; print(torchvision.__version__)"`
    if [ -n "$version" ];then
        return 0
    fi
    sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev -y
    if [ ! -e "torchvision" ];then
        git clone https://github.com/pytorch/vision torchvision --branch v0.11.1  --depth=1
    fi
    cd torchvision
    export BUILD_VERSION=0.11.1

    python3 -m pip install -e .
    python3 -c "import torchvision; print(torchvision.__version__)"
    cd -
}

install_cmake() {
    command -v cmake > /dev/null 2>&1 || {
        python3 -m pip install cmake
    }
    echo "cmake installed $(which cmake)"
}

install_tensorrt() {
    echo 'export PYTHONPATH=/usr/lib/python3.8/dist-packages:${PYTHONPATH}' >> ~/mmdeploy.env
    export PYTHONPATH=/usr/lib/python3.8/dist-packages:${PYTHONPATH}

    echo 'export TENSORRT_DIR=/usr/include/aarch64-linux-gnu' >> ~/mmdeploy.env
    export TENSORRT_DIR=/usr/include/aarch64-linux-gnu

    echo 'export PATH=$PATH:/usr/local/cuda/bin' >> ~/mmdeploy.env
    export PATH=$PATH:/usr/local/cuda/bin

    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64' >> ~/mmdeploy.env
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

    echo 'export CUDA_HOME=/usr/local/cuda-11' >> ~/mmdeploy.env
    export CUDA_HOME=/usr/local/cuda-11

    echo 'export CUDA_ROOT=/usr/local/cuda-11' >> ~/mmdeploy.env
    export CUDA_ROOT=/usr/local/cuda-11

    python -c "import tensorrt; print(tensorrt.__version__)"
}

install_mmcv_pycuda() {
    version=`python3 -c "import mmcv; print(mmcv.__version__)"`
    if [ -n "$version" ];then
        return 0
    fi

    # try prebuilt .whl
    board=`cat /etc/nv_tegra_release  | awk '{print $9}'`
    release=`cat /etc/nv_tegra_release  | awk '{print $2}'`
    revision=`cat /etc/nv_tegra_release  | awk '{print $5}'`
    if [ ${board} = "t186ref," ];then
        if [ ${release} = "R34," ];then
            if [ ${revision} = "1.1," ];then
                # use prebuilt whl
                wget -q --show-progress --https://github.com/tpoisonooo/mmcv-jetson-orin-prebuilt-whl/raw/main/mmcv_full-1.5.1-cp38-cp38-linux_aarch64.whl
                python3 -m pip install mmcv_full-1.5.1-cp38-cp38-linux_aarch64.whl

                wget https://github.com/tpoisonooo/mmcv-jetson-orin-prebuilt-whl/raw/main/pycuda-2022.1-cp38-cp38-linux_aarch64.whl
                python3 -m pip install pycuda-2022.1-cp38-cp38-linux_aarch64.whl
            fi
        fi
    elif [ ! -e "mmcv" ];then
        # source build mmcv and pycuda
        sudo apt-get install -y libssl-dev
        git clone https://github.com/open-mmlab/mmcv.git --branch v1.5.1 --depth=1
        cd mmcv
        echo 'Building mmcv-full with MMCV_WITH_OPS=1 and pycuda, it may take an hour, please wait..'
        MMCV_WITH_OPS=1 python3 -m pip install -e .

        python3 -m pip install pycuda
        cd -
    fi

    python3 -c "import mmcv; print(mmcv.__version__)"
}

install_pplcv() {
    if [ ! -e "ppl.cv" ];then
        git clone https://github.com/openppl-public/ppl.cv.git --depth=1 --recursive
    fi
    cd ppl.cv
    ./build.sh cuda
    echo "PPLCV_DIR=$(pwd)" >> ~/mmdeploy.env
    export PPLCV_DIR=$(pwd)
    cd -
}

install_mmdeploy() {
    sudo apt-get install -y pkg-config libhdf5-103 libhdf5-dev libspdlog-dev
    python3 -m pip install onnx
    python3 -m pip install versioned-hdf5

    # build and install mmdeploy
    cd ../mmdeploy
    git submodule init
    git submodule update

    if [ ! -e "build" ];then
        mkdir -p build
    fi
    cd build
    cmake .. \
        -DMMDEPLOY_BUILD_SDK=ON \
        -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
        -DMMDEPLOY_BUILD_EXAMPLES=ON \
        -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
        -DMMDEPLOY_TARGET_BACKENDS="trt" \
        -DMMDEPLOY_CODEBASES=all \
        -Dpplcv_DIR=${PPLCV_DIR}/cuda-build/install/lib/cmake/ppl
    make -j 7 && make install
    cd -
    python3 -m pip install -v -e .
    python3 tools/check_env.py
}

show_env() {
    echo ""
    echo "----------------------------------------------------------------------------------------------------------"
    echo '>> Install finished, `source ~/mmdeploy.env` to setup your environment !'
    cat ~/mmdeploy.env
    echo "----------------------------------------------------------------------------------------------------------"
}

# setup env
echo "" > ~/mmdeploy.env
echo 'export OPENBLAS_CORETYPE=ARMV8' >> ~/mmdeploy.env
export OPENBLAS_CORETYPE=ARMV8

echo 'export ARCH=aarch64' >> ~/mmdeploy.env
export ARCH=aarch64

check_python_38

if [ ! -e "../mmdeploy-dep" ];then
    mkdir ../mmdeploy-dep
fi
cd ../mmdeploy-dep
echo $(pwd)

install_torch
install_torchvision
install_cmake
install_tensorrt
install_mmcv_pycuda
install_pplcv
install_mmdeploy
show_env
