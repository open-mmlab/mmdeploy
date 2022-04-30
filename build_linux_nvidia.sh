#!/bin/bash
# build_linux_nvidia.sh
#   Date: 08-03-2022, 24-04-2022
#
#   Run this script to build MMDeploy SDK and necessary prerequisites.
#   This script will also setup python venv
#

#####
# Build vars
BUILD_TYPE="Release"
ARCH=$(uname -i)
PROC_NUM=$(nproc)
# Default GCC
GCC_COMPILER="g++"

#####
# Directories
# WORKING_DIR must correspond to script dir, i.e. MMDeploy root
WORKING_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PPLCV_DIR=${WORKING_DIR}/ppl.cv
MMDEPLOY_DIR=${WORKING_DIR}/MMDeploy

#####
# Versions
PPLCV_VER="0.6.2"
MMDEPLOY_VER="0.4.0"
CMAKE_VER="3.23.0"

#####
# Flags
# WITH_PYTHON: Install misc. dependencies in the active venv
WITH_PYTHON=1
# WITH_CLEAN: Remove build output dirs
WITH_CLEAN=1

#####
# Prefix: Set install prefix for ppl.cv, mmdeploy SDK depending on arch
if [[ "$ARCH" == aarch64 ]]; then
  INSTALL_PREFIX="/usr/local/aarch64-linux-gnu"
else
  INSTALL_PREFIX="/usr/local"
fi
PYTHON_VENV_DIR=${WORKING_DIR}/venv-mmdeploy

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
function version {
  echo "$@" | awk -F. '{ printf("%d%03d%03d%03d\n", $1,$2,$3,$4); }';
}

prereqs() {
  # spdlog
  echo_green "Checking spdlog version..."
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
    cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON && make -j${PROC_NUM}
    sudo make install
    sudo ldconfig
  fi

  # cmake check & install
  echo_green "Checking your cmake version..."
  CMAKE_DETECT_VER=$(cmake --version | grep -oP '(?<=version).*')
  if [ $(version $CMAKE_DETECT_VER) -ge $(version "3.14.0") ]; then
    echo_green "Cmake version $CMAKE_DETECT_VER is up to date"
  else
    echo_green "CMake too old, purging existing cmake and installing ${CMAKE_VER}..."
    # purge existing
    sudo apt-get purge cmake
    sudo snap remove cmake
    # install prebuild
    wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VER}/cmake-${CMAKE_VER}-linux-${ARCH}.sh
    chmod +x cmake-${CMAKE_VER}-linux-${ARCH}.sh
    sudo ./cmake-${CMAKE_VER}-linux-${ARCH}.sh --prefix=/usr --skip-license
  fi

  # gcc-7 check
  echo_green "Checking your gcc version..."
  GCC_DETECT_VER=$(gcc --version | grep -oP '(?<=\)).*' -m1)
  if [ $(version $GCC_DETECT_VER) -ge $(version "7.0.0") ]; then
    echo_green "GCC version $GCC_DETECT_VER is up to date"
  else
    echo_green "gcc version too old, installing ${CMAKE_VER}..."
    echo_green "Purge existing cmake and install ${GCC_DETECT_VER}..."
    # Add repository if ubuntu < 18.04
    sudo add-apt-repository ppa:ubuntu-toolchain-r/test
    sudo apt-get update
    sudo apt-get install gcc-7
    sudo apt-get install g++-7
    GCC_COMPILER="g++-7"
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
      echo_green "Please re-run this script for changes to apply!"
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
  deactivate

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

  if [ -d "${PYTHON_VENV_DIR}" ]; then
    read -p "Reinstall existing Python venv ${PYTHON_VENV_DIR}? (y/n)" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
      rm -r ${PYTHON_VENV_DIR}
      python3 -m venv ${PYTHON_VENV_DIR} --system-site-packages #system site packages to keep trt from system installation
    fi
  else
    python3 -m venv ${PYTHON_VENV_DIR} --system-site-packages #system site packages to keep trt from system installation
  fi

  source ${PYTHON_VENV_DIR}/bin/activate
  python3 get-pip.py
  pip3 install testresources
  pip3 install --upgrade setuptools wheel

  # Latest PIL is not compatible with mmcv=1.4.1
  pip install Pillow==7.0.0

  if [[ "$ARCH" == aarch64 ]]
  then
    # protofbuf on jetson is quite old - must be upgraded
    pip3 install --upgrade protobuf
    # Install numpy 1.19.4 as newer versions might give "Illegal instruction (core dumped)" on Jetson
    # TODO Numpy might be installed per default so we should not remove it.
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

  # deactivate python venv again
  deactivate
}

pplcv() {
  ## ppl.cv
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
  cmake -DHPCC_USE_CUDA=ON -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} .. && make -j${PROC_NUM} && sudo make install
  sudo ldconfig

  # TODO pack prebuild as tar.gz file
  #cd ..
  #tar -zcvf ${WORKING_DIR}/pplcv_${PPLCV_VER}_cuda-${ARCH}-build.tar.gz build/
}

mmdeploy(){
  ## mmdeploy SDK
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

  # python dependencies
  if [[ $WITH_PYTHON -eq 1 ]]
  then
    source ${PYTHON_VENV_DIR}/bin/activate

    ## h5py (Required by mmdeploy)
    ## h5py not directly supported by jetson and must be built/installed manually
    sudo apt-get install pkg-config libhdf5-10* libhdf5-dev -y
    sudo pip3 install Cython
    sudo env H5PY_SETUP_REQUIRES=0 pip3 install -U h5py==2.9.0

    pip install -e .
  fi

  # build
  mkdir build -p && cd build
  cmake .. \
    -DMMDEPLOY_BUILD_SDK=ON \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -DCMAKE_CXX_COMPILER=${GCC_COMPILER} \
    -Dpplcv_DIR=${INSTALL_PREFIX}/lib/cmake/ppl \
    -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
    -DMMDEPLOY_TARGET_BACKENDS=trt \
    -DMMDEPLOY_CODEBASES=all \
    -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
    -DTENSORRT_DIR=${TENSORRT_DIR} \
    -DCUDNN_DIR=${CUDNN_DIR}
  cmake --build . -- -j${PROC_NUM} && sudo cmake --install .
  sudo ldconfig

  # TODO Pack build output as tar.gz file
  #cd ..
  # tar -zcvf ${WORKING_DIR}/mmdeploysdk_${MMDEPLOY_VER}_${ARCH}-build.tar.gz build/
  # Unpack as tar -zxf mmdeploysdk_*.tar.gz --directory MMDeploy-aarch64

  ## build mmdeploy examples
  cp -r ${WORKING_DIR}/MMDeploy/demo/csrc ${WORKING_DIR}/MMDeploy/build/example
  cd ${WORKING_DIR}/MMDeploy/build/example
  rm -r build
  mkdir build -p && cd build
  cmake .. -DMMDeploy_DIR=${INSTALL_PREFIX}
  make all

  # deactivate python venv again
  deactivate
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

# remove all build files
if [[ $WITH_CLEAN -eq 1 ]]
then
  sudo rm -r ${PPLCV_DIR}/build
  sudo rm -r ${MMDEPLOY_DIR}/build
fi

$appargument1
