#!/bin/bash
# build_mmdeploy.sh
#   Date: 08-03-2022, 24-04-2022
#
#   Run this script to build MMDeploy SDK and prerequisites.
#   This script will also setup python venv
#
BUILD_TYPE="Release"
ARCH=$(uname -i)
WITH_PYTHON=1
WITH_CLEAN=1
# Get path to script to ensure script runs from 'logends' root
WORKING_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

PPLCV_DIR=${WORKING_DIR}/ppl.cv
MMDEPLOY_DIR=${WORKING_DIR}/MMDeploy
PPLCV_VER="0.6.2"
MMDEPLOY_VER="0.4.0"
INSTALL_PREFIX="/usr/local"
PYTHON_VENV_DIR="mmdeploy"

appargument1=$1
#appargument2=$2

echo_green() {
  if [ -n "$1" ]; then
    echo "$(tput setaf 10)$1$(tput sgr 0)"
  fi
}
echo_red() {
  if [ -n "$1" ]; then
    echo "$(tput setaf 1)$1$(tput sgr 0)"
  fi
}
echo_blue() {
  if [ -n "$1" ]; then
    echo "$(tput setaf 4)$1$(tput sgr 0)"
  fi
}
contains_element () {
  local e match="$1"
  shift
  for e; do [[ "$e" == "$match" ]] && return 0; done
  return 1
}

prereqs() {
  # spdlog
  read -p "Install latest spdlog from source? (y/n)" -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]
  then
    # remove libspdlog, as it might be an old version
    sudo apt-get remove libspdlog-dev -y

    echo "Building and installing latest spdlog from source"
    git clone https://github.com/gabime/spdlog.git spdlog
    cd spdlog
    git pull
    git checkout tags/v1.8.1
    mkdir build -p && cd build
    # we must build spdlog with -fPIC enabled
    cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON && make -j
    sudo make install
    sudo ldconfig
  fi

  # cmake check
  echo_green "Check your cmake version:"
  cmake --version
  read -p "Install latest CMake? (>=3.18 is required) (y/n)" -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]
  then
    # purge existing
    sudo apt-get purge cmake
    sudo snap remove cmake
    # build cmake from source
    sudo apt-get install -y libssl-dev
    wget https://github.com/Kitware/CMake/releases/download/v3.23.0/cmake-3.23.0.tar.gz
    tar -zxvf cmake-3.23.0.tar.gz
    cd cmake-3.23.0
    ./bootstrap
    make -j
    sudo make install
    #
    source ~/.bashrc
    cmake --version
  fi

  # gcc-7 check
  echo_green "Check your gcc version:"
  gcc --version
  read -p "Upgrade to GCC-7? (>=GCC-7 is required) (y/n)" -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]
  then
    # Add repository if ubuntu < 18.04
    sudo add-apt-repository ppa:ubuntu-toolchain-r/test
    sudo apt-get update
    sudo apt-get install gcc-7
    sudo apt-get install g++-7
    GCC
  fi

  # tensorrt check
  echo_green "Check your TensorRT version:"
  ## Check if ${TENSORRT_DIR} env variable has been set
  if [ -d "${TENSORRT_DIR}" ]; then
    echo_green "TENSORRT_DIR env. variable has been set ${TENSORRT_DIR}"
  else
    echo_red "TENSORRT_DIR env. variable has NOT been set."
    if [[ "$ARCH" == aarch64 ]]; then
      echo_green "Added TENSORRT_DIR, CUDNN_DIR to env."
      echo 'export TENSORRT_DIR="/usr/include/'${ARCH}'-linux-gnu/"' >> ${HOME}/.bashrc
      echo 'export CUDNN_DIR="/usr/include/'${ARCH}'-linux-gnu/"' >> ${HOME}/.bashrc
      echo 'export LD_LIBRARY_PATH="/usr/lib/'${ARCH}'-linux-gnu/:$LD_LIBRARY_PATH"' >> ${HOME}/.bashrc
      source ${HOME}/.bashrc
      exec bash
    else
      echo_red "Please Install TensorRT, CUDNN and add TENSORRT_DIR, CUDNN_DIR to environment variables before running this script!"
      exit
    fi
  fi

  # Determine TensorRT version and set paths accordingly
  echo "Checking TensorRT version...Please verify the detected versions below:"
  if [[ "$ARCH" == aarch64 ]]; then
    cat /usr/include/${ARCH}-linux-gnu/NvInferVersion.h | grep NV_TENSORRT
  else
    cat ${TENSORRT_DIR}/include/NvInferVersion.h | grep NV_TENSORRT
  fi
  read -p "Is TensorRT >=8.0.1.6 installed? (Always installed on Jetson) (y/n)" -n 1 -r
  echo
  if [[ $REPLY =~ ^[Nn]$ ]]
  then
      echo "Error: You must install TensorRT before installing MMDeploy!"
      exit
  fi

  # opencv
  sudo apt-get install libopencv-dev
}

py_venv() {
  # deactivate venv, if it has already been activated
  source deactivate

  #check for python installed version
  pyv="$(python3 -V 2>&1)"
  pyv_old="Python 3.6"

  if echo "$pyv" | grep -q "$pyv_old"; then
      # use python 3.6
      curl https://bootstrap.pypa.io/pip/3.6/get-pip.py -o get-pip.py
  else
      # use python >=3.7
      curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  fi

  # dependencies
  sudo apt-get install protobuf-compiler libprotoc-dev libssl-dev curl ninja-build -y
  sudo apt-get install libopenblas-dev python3-venv python3-dev python3-setuptools -y
  sudo python3 get-pip.py
  pip3 install testresources
  pip3 install --upgrade setuptools wheel

  read -p "Remove existing Python venv? (y/n)" -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]
  then
    rm -r ${PYTHON_VENV_DIR}
    python3 -m venv ${PYTHON_VENV_DIR} --system-site-packages #system site packages to keep trt from system installation
  fi

  source ${PYTHON_VENV_DIR}/bin/activate
  python3 get-pip.py
  pip3 install testresources
  pip3 install --upgrade setuptools wheel
  # protofbuf on jetson is quite old - must be upgraded
  pip3 install --upgrade protobuf
  # Latest pillow is not compatible with mmcv
  pip install Pillow==7.0.0

  if [[ "$ARCH" == aarch64 ]]
  then
    # TODO Numpy might be installed per default so we should not remove it
    pip3 install numpy==1.19.4
  else
    pip3 install numpy
  fi
  pip3 install opencv-python
  pip3 install matplotlib

  # pytorch, torchvision, torchaudio
  if [[ "$ARCH" == aarch64 ]]
  then
    # pytorch
    wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
    pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
    # torchvision
    sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev -y
    sudo rm -r torchvision
    git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
    cd torchvision
    export BUILD_VERSION=0.11.1  # where 0.x.0 is the torchvision version
    python3 setup.py install
    cd ../
    # torchaudio
    #sudo apt-get install -y sox libsox-dev libsox-fmt-all
    #sudo rm -r torchaudio
    #git clone -b v0.10.0 https://github.com/pytorch/audio torchaudio
    #cd torchaudio
    #git submodule update --init --recursive
    #python3 setup.py install
    #cd ../
    # mmcv
    pip3 uninstall mmcv-full
    pip3 install mmcv-full==1.4.1 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
  else
    pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
    # mmcv
    pip3 uninstall mmcv-full
    pip3 install mmcv-full==1.4.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
  fi

}

pplcv() {
  ## ppl.cv - install in /usr/local
  cd ${WORKING_DIR}
  echo_blue "checking out '${PPLCV_DIR}' pkg..."
  if [ -d "${PPLCV_DIR}" ]; then
      echo_green "Already exists! Checking out the requested version..."
  else
      git clone https://github.com/openppl-public/ppl.cv.git ${PPLCV_DIR}
  fi
  cd ${PPLCV_DIR}
  git pull
  git checkout tags/v${PPLCV_VER}
  mkdir build -p && cd build
  cmake -DHPCC_USE_CUDA=ON -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} .. && make -j${processor_num} && sudo make install
  sudo ldconfig
  # pack as tar.gz file
  cd ..
  tar -zcvf ${WORKING_DIR}/pplcv_${PPLCV_VER}_cuda-${ARCH}-build.tar.gz build/
}

mmdeploy(){
  ## h5py (Required by mmdeploy)
  ## h5py not directly supported by jetson and must be built/installed manually
  sudo apt-get install pkg-config libhdf5-10* libhdf5-dev -y
  sudo pip3 install Cython
  sudo env H5PY_SETUP_REQUIRES=0 pip3 install -U h5py==2.9.0

  ## mmdeploy SDK - install in /usr/local
  cd ${WORKING_DIR}
  echo_blue "checking out '${MMDEPLOY_DIR}' pkg..."
  if [ -d "${MMDEPLOY_DIR}" ]; then
    echo_green "Already exists! Checking out the requested version..."
  else
    git clone https://github.com/open-mmlab/mmdeploy.git ${MMDEPLOY_DIR}
  fi
  cd ${MMDEPLOY_DIR}
  git pull
  git checkout tags/v${MMDEPLOY_VER}
  # reinit submodules
  git submodule update --init --recursive

  if [[ $WITH_PYTHON -eq 1 ]]
  then
    pip install -e .
  fi
  rm -r build
  mkdir build -p && cd build
  cmake .. \
    -DMMDEPLOY_BUILD_SDK=ON \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -DCMAKE_CXX_COMPILER=g++ \
    -Dpplcv_DIR=/usr/local/lib/cmake/ppl \
    -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
    -DMMDEPLOY_TARGET_BACKENDS=trt \
    -DMMDEPLOY_CODEBASES=all \
    -DTENSORRT_DIR=${TENSORRT_DIR} \
    -DCUDNN_DIR=${CUDNN_DIR}
  cmake --build . -- -j$(nproc) && sudo cmake --install .
  sudo ldconfig
  # pack as tar.gz file
  cd ..
  tar -zcvf ${WORKING_DIR}/mmdeploysdk_${MMDEPLOY_VER}_${ARCH}-build.tar.gz build/
  # Unpack as tar -zxf mmdeploysdk_*.tar.gz --directory MMDeploy-aarch64

  ## build mmdeploy examples
  cp -r ${WORKING_DIR}/MMDeploy/demo/csrc ${WORKING_DIR}/MMDeploy/build/example
  cd ${WORKING_DIR}/MMDeploy/build/example
  rm -r build
  mkdir build -p && cd build
  cmake .. -DMMDeploy_DIR=${INSTALL_PREFIX}
  make all
}

all() {
  # build all
  prereqs
  py_venv
  pplcv
  mmdeploy
}

#####
# supported package
package_list=(
  "all"
  "prereqs"
  "py_venv"
  "pplcv"
  "mmdeploy"
)

#####
# check input argument
if contains_element "$appargument1" "${package_list[@]}"; then
  echo_green "Build and install '$appargument1'..."
else
  echo_red "Unsupported argument '$appargument1'. Use one of the following:"
  for i in ${package_list[@]}
  do
    echo $i
  done
  exit
fi

# prepare build
if [[ $WITH_PYTHON -eq 1 ]]
then
  source venv-mmdet/bin/activate
fi

# remove all build files
if [[ $WITH_CLEAN -eq 1 ]]
then
  sudo rm -r ${PPLCV_DIR}/build
  sudo rm -r ${MMDEPLOY_DIR}/build
fi

$appargument1
