# Build for Android

- [Build for Android](#build-for-android)
  - [Build From Source](#build-from-source)
    - [Install Toolchains](#install-toolchains)
    - [Install Dependencies](#install-dependencies)
      - [Install Dependencies for SDK](#install-dependencies-for-sdk)
    - [Build MMDeploy](#build-mmdeploy)
      - [Build Options Spec](#build-options-spec)
      - [Build SDK](#build-sdk)
      - [Build Demo](#build-demo)

______________________________________________________________________

MMDeploy provides cross compile for android platform.

Model converter is executed on linux platform, and SDK is executed on android platform.

Here are two steps for android build.

1. Build model converter on linux, please refer to [How to build linux](./linux-x86_64.md)

2. Build SDK using android toolchain on linux.

This doc is only for how to build SDK using android toolchain on linux.

## Build From Source

### Install Toolchains

- cmake

  **Make sure cmake version >= 3.14.0**. If not, you can follow instructions below to install cmake 3.20.0. For more versions of cmake, please refer to [cmake website](https://cmake.org/install).

  ```bash
  wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0-linux-x86_64.tar.gz
  tar -xzvf cmake-3.20.0-linux-x86_64.tar.gz
  sudo ln -sf $(pwd)/cmake-3.20.0-linux-x86_64/bin/* /usr/bin/
  ```

- ANDROID NDK 19+

  **Make sure android ndk version >= 19.0**. If not, you can follow instructions below to install android ndk r23b. For more versions of android ndk, please refer to [android ndk website](https://developer.android.com/ndk/downloads).

  ```bash
  wget https://dl.google.com/android/repository/android-ndk-r23b-linux.zip
  unzip android-ndk-r23b-linux.zip
  cd android-ndk-r23b
  export NDK_PATH=${PWD}
  ```

### Install Dependencies

#### Install Dependencies for SDK

You can skip this chapter if only interested in model converter.

<table>
<thead>
  <tr>
    <th>NAME </th>
    <th>INSTALLATION </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>OpenCV<br>(>=3.0) </td>
    <td>
<pre><code>
export OPENCV_VERSION=4.5.4
wget https://github.com/opencv/opencv/releases/download/${OPENCV_VERSION}/opencv-${OPENCV_VERSION}-android-sdk.zip
unzip opencv-${OPENCV_VERSION}-android-sdk.zip
export OPENCV_ANDROID_SDK_DIR=${PWD}/OpenCV-android-sdk
</code></pre>
    </td>

</tr>
  <tr>
    <td>ncnn </td>
    <td>A high-performance neural network inference computing framework supporting for android.</br>
  <b> Now, MMDeploy supports v20220216 and has to use <code>git clone</code> to download it.</b><br>
<pre><code>
git clone -b 20220216 https://github.com/Tencent/ncnn.git
cd ncnn
git submodule update --init
export NCNN_DIR=${PWD}
mkdir -p build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=${NDK_PATH}/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-30 -DNCNN_VULKAN=ON -DNCNN_DISABLE_EXCEPTION=OFF -DNCNN_DISABLE_RTTI=OFF ..
make install
</code></pre>
   </td>
  </tr>
</tbody>
</table>

### Build MMDeploy

#### Build Options Spec

<table>
<thead>
  <tr>
    <th>NAME</th>
    <th>VALUE</th>
    <th>DEFAULT</th>
    <th>REMARK</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>MMDEPLOY_BUILD_SDK</td>
    <td>{ON, OFF}</td>
    <td>OFF</td>
    <td>Switch to build MMDeploy SDK</td>
  </tr>
  <tr>
    <td>MMDEPLOY_BUILD_SDK_PYTHON_API</td>
    <td>{ON, OFF}</td>
    <td>OFF</td>
    <td>switch to build MMDeploy SDK python package</td>
  </tr>
  <tr>
    <td>MMDEPLOY_BUILD_TEST</td>
    <td>{ON, OFF}</td>
    <td>OFF</td>
    <td>Switch to build MMDeploy SDK unittest cases</td>
  </tr>
  <tr>
    <td>MMDEPLOY_TARGET_DEVICES</td>
    <td>{"cpu"}</td>
    <td>cpu</td>
    <td>Enable target device. <br>If you want use ncnn vulkan accelerate, you still fill <code>{"cpu"}</code> here. Because, vulkan accelerate is only for ncnn net. The other part of inference is still using cpu.</td>
  </tr>
  <tr>
    <td>MMDEPLOY_TARGET_BACKENDS</td>
    <td>{"ncnn"}</td>
    <td>N/A</td>
    <td>Enabling inference engine. <br><b>By default, no target inference engine is set, since it highly depends on the use case.</b><br> Only ncnn backend is supported for android platform now.<br>
    After specifying the inference engine, it's package path has to be passed to cmake as follows, <br>
    1. <b>ncnn</b>: ncnn. <code>ncnn_DIR</code> is needed.
<pre><code>-Dncnn_DIR=${NCNN_DIR}/build/install/lib/cmake/ncnn</code></pre>
   </td>
  </tr>
  <tr>
    <td>MMDEPLOY_CODEBASES</td>
    <td>{"mmcls", "mmdet", "mmseg", "mmedit", "mmocr", "all"}</td>
    <td>N/A</td>
    <td>Enable codebase's postprocess modules. It MUST be set by a semicolon separated list of codebase names. The currently supported codebases are 'mmcls', 'mmdet', 'mmedit', 'mmseg', 'mmocr'. Instead of listing them one by one, you can also pass <code>all</code> to enable them all, i.e., <code>-DMMDEPLOY_CODEBASES=all</code></td>
  </tr>
  <tr>
    <td>MMDEPLOY_SHARED_LIBS</td>
    <td>{ON, OFF}</td>
    <td>ON</td>
    <td>Switch to build shared library or static library of MMDeploy SDK. Now you should build static library for android. Bug will be fixed soon.</td>
  </tr>
</tbody>
</table>

#### Build SDK

MMDeploy provides a recipe as shown below for building SDK with ncnn as inference engine for android.

- cpu + ncnn
  ```Bash
  cd ${MMDEPLOY_DIR}
  mkdir -p build && cd build
  cmake .. \
      -DMMDEPLOY_BUILD_SDK=ON \
      -DOpenCV_DIR=${OPENCV_ANDROID_SDK_DIR}/sdk/native/jni/abi-arm64-v8a \
      -Dncnn_DIR=${NCNN_DIR}/build/install/lib/cmake/ncnn \
      -DMMDEPLOY_TARGET_BACKENDS=ncnn \
      -DMMDEPLOY_CODEBASES=all \
      -DMMDEPLOY_SHARED_LIBS=OFF \
      -DCMAKE_TOOLCHAIN_FILE=${NDK_PATH}/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-30 \
      -DANDROID_CPP_FEATURES="rtti exceptions"

  make -j$(nproc) && make install
  ```

#### Build Demo

```Bash
cd ${MMDEPLOY_DIR}/build/install/example
mkdir -p build && cd build
cmake .. \
      -DOpenCV_DIR=${OPENCV_ANDROID_SDK_DIR}/sdk/native/jni/abi-arm64-v8a \
      -Dncnn_DIR=${NCNN_DIR}/build/install/lib/cmake/ncnn \
      -DMMDeploy_DIR=${MMDEPLOY_DIR}/build/install/lib/cmake/MMDeploy \
      -DCMAKE_TOOLCHAIN_FILE=${NDK_PATH}/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-30
make -j$(nproc)
```
