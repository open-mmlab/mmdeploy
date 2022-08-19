# Cross compile snpe inference server on Ubuntu 18

mmdeploy has provided a prebuilt package, if you want to compile it by self, or need to modify the `.proto` file, you can refer to this document.

Note that the official gRPC documentation does not have complete support for the NDK.

## 1. Environment

| Item               | Version        | Remarks                                                   |
| ------------------ | -------------- | --------------------------------------------------------- |
| snpe               | 1.59           | 1.60 uses clang-8.0, which may cause compatibility issues |
| host OS            | ubuntu18.04    | snpe1.59 specified version                                |
| NDK                | r17c           | snpe1.59 specified version                                |
| gRPC               | commit 6f698b5 | -                                                         |
| Hardware equipment | qcom888        | qcom chip required                                        |

## 2. Cross compile gRPC with NDK

1. Pull gRPC repo, compile `protoc` and `grpc_cpp_plugin` on host

```bash
# Install dependencies
$ apt-get update && apt-get install -y libssl-dev
# Compile
$ git clone https://github.com/grpc/grpc --recursive=1 --depth=1
$ mkdir -p cmake/build
$ pushd cmake/build

$ cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DgRPC_INSTALL=ON \
  -DgRPC_BUILD_TESTS=OFF \
  -DgRPC_SSL_PROVIDER=package \
  ../..
# Install to host
$ make -j
$ sudo make install
```

2. Download the NDK and cross-compile the static libraries with android aarch64 format

```bash
$ wget https://dl.google.com/android/repository/android-ndk-r17c-linux-x86_64.zip
$ unzip android-ndk-r17c-linux-x86_64.zip

$ export ANDROID_NDK=/path/to/android-ndk-r17c

$ cd /path/to/grpc
$ mkdir -p cmake/build_aarch64  && pushd cmake/build_aarch64

$ cmake ../.. \
 -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake \
 -DANDROID_ABI=arm64-v8a \
 -DANDROID_PLATFORM=android-26 \
 -DANDROID_TOOLCHAIN=clang \
 -DANDROID_STL=c++_shared \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_INSTALL_PREFIX=/tmp/android_grpc_install_shared

$ make -j
$ make install
```

3. At this point `/tmp/android_grpc_install` should have the complete installation file

```bash
$ cd /tmp/android_grpc_install
$ tree -L 1
.
├── bin
├── include
├── lib
└── share
```

## 3. (Skipable) Self-test whether NDK gRPC is available

1. Compile the helloworld that comes with gRPC

```bash
$ cd /path/to/grpc/examples/cpp/helloworld/
$ mkdir cmake/build_aarch64 -p && pushd cmake/build_aarch64

$ cmake ../.. \
 -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake \
 -DANDROID_ABI=arm64-v8a \
 -DANDROID_PLATFORM=android-26 \
 -DANDROID_STL=c++_shared \
 -DANDROID_TOOLCHAIN=clang \
 -DCMAKE_BUILD_TYPE=Release \
 -Dabsl_DIR=/tmp/android_grpc_install_shared/lib/cmake/absl \
 -DProtobuf_DIR=/tmp/android_grpc_install_shared/lib/cmake/protobuf \
 -DgRPC_DIR=/tmp/android_grpc_install_shared/lib/cmake/grpc

$ make -j
$ ls greeter*
greeter_async_client   greeter_async_server     greeter_callback_server  greeter_server
greeter_async_client2  greeter_callback_client  greeter_client
```

2. Turn on debug mode on your phone, push the binary to `/data/local/tmp`

```bash
$ adb push greeter* /data/local/tmp
```

3. `adb shell` into the phone, execute client/server

```bash
/data/local/tmp $ ./greeter_client
Greeter received: Hello world
```

## 4. Cross compile snpe inference server

1. Open the [snpe tools website](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk/tools) and download version 1.59. Unzip and set environment variables

> Note that snpe >= 1.60 starts using `clang-8.0`, which may cause incompatibility with `libc++_shared.so` on older devices.

```bash
$ export SNPE_ROOT=/path/to/snpe-1.59.0.3230
```

2. Open the snpe server directory within mmdeploy, use the options when cross-compiling gRPC

```bash
$ cd /path/to/mmdeploy
$ cd service/snpe/server

$ mkdir -p build && cd build
$ export ANDROID_NDK=/path/to/android-ndk-r17c
$ cmake .. \
 -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake \
 -DANDROID_ABI=arm64-v8a \
 -DANDROID_PLATFORM=android-26 \
 -DANDROID_STL=c++_shared \
 -DANDROID_TOOLCHAIN=clang \
 -DCMAKE_BUILD_TYPE=Release \
 -Dabsl_DIR=/tmp/android_grpc_install_shared/lib/cmake/absl \
 -DProtobuf_DIR=/tmp/android_grpc_install_shared/lib/cmake/protobuf \
 -DgRPC_DIR=/tmp/android_grpc_install_shared/lib/cmake/grpc

 $ make -j
 $ file inference_server
inference_server: ELF 64-bit LSB shared object, ARM aarch64, version 1 (SYSV), dynamically linked, interpreter /system/bin/linker64, BuildID[sha1]=252aa04e2b982681603dacb74b571be2851176d2, with debug_info, not stripped
```

Finally,  you can see `infernece_server`, `adb push` it to the device and execute.

## 5. Regenerate the proto interface

If you have changed `inference.proto`, you need to regenerate the .cpp and .py interfaces

```Shell
$ python3 -m pip install grpc_tools --user
$ python3 -m  grpc_tools.protoc -I./ --python_out=./client/ --grpc_python_out=./client/ inference.proto

$ ln -s `which protoc-gen-grpc`
$ protoc --cpp_out=./ --grpc_out=./  --plugin=protoc-gen-grpc=grpc_cpp_plugin  inference.proto
```

## Reference

- snpe tutorial https://developer.qualcomm.com/sites/default/files/docs/snpe/cplus_plus_tutorial.html
- gRPC cross build script https://raw.githubusercontent.com/grpc/grpc/master/test/distrib/cpp/run_distrib_test_cmake_aarch64_cross.sh
- stackoverflow https://stackoverflow.com/questions/54052229/build-grpc-c-for-android-using-ndk-arm-linux-androideabi-clang-compiler
