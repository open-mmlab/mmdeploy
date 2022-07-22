#!/bin/bash

workdir=`pwd`
build_dir="${workdir}/build"

if [[ `uname` == "Linux" ]]; then
    processor_num=`cat /proc/cpuinfo | grep processor | grep -v grep | wc -l`
else
    processor_num=1
fi

build_type='Release'
options="-DCMAKE_BUILD_TYPE=${build_type}"

mkdir ${build_dir}
cd ${build_dir}
cmd="cmake $options .. && cmake --build . -j ${processor_num} --config ${build_type}"
echo "cmd -> $cmd"
eval "$cmd"
