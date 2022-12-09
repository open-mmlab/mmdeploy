# build for IPU

## 1、enable poplar from docker
pull graphcore/poplar image from dockerhub: 
$ docker pull graphcore/poplar
poplar docker detail：[poplar-docker](https://hub.docker.com/r/graphcore/poplar)

## 2、build mmdeploy SDK and demo
1. install c++ dependency
$  apt-get install libopencv-dev libboost-all-dev gcc g++-7
2. install conda
$  downlaod [anaconda](https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh)
   bash Anaconda3-2022.10-Linux-x86_64.sh
   source ~/.bashrc
4. create python3.6 env
$ conda create --name ipu python=3.6 cmake
5. build mmdeploy
$ cmake .. -DCMAKE_CXX_COMPILER=g++-7 -DMMDEPLOY_BUILD_SDK=ON -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON -DMMDEPLOY_BUILD_EXAMPLES=ON -DMMDEPLOY_TARGET_DEVICES=cpu -DMMDEPLOY_TARGET_BACKENDS=ipu -DMMDEPLOY_BUILD_TEST=ON
