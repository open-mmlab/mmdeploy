# Build for Android

- [Build for Android](#build-for-android)
  - [Build From Source](#build-from-source)
    - [Install Toolchains](#install-toolchains)
    - [Install Dependencies](#install-dependencies)
      - [Install Dependencies for SDK](#install-dependencies-for-sdk)
    - [Build MMDeploy](#build-mmdeploy)
      - [Build Options Spec](#build-options-spec)
      - [Build SDK and Demos](#build-sdk-and-demos)

______________________________________________________________________

MMDeploy provides cross compile for android platform.

Model converter is executed on linux platform, and SDK is executed on android platform.

Here are two steps for android build.

1. Build model converter on linux, please refer to [How to build linux](linux-x86_64.md)

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

  **Make sure android ndk version >= 19.0**. If not, you can follow instructions below to install android ndk r23c. For more versions of android ndk, please refer to [android ndk website](https://developer.android.com/ndk/downloads).

  ```bash
  wget https://dl.google.com/android/repository/android-ndk-r23c-linux.zip
  unzip android-ndk-r23c-linux.zip
  cd android-ndk-r23c
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
export OPENCV_VERSION=4.6.0
wget https://github.com/opencv/opencv/releases/download/${OPENCV_VERSION}/opencv-${OPENCV_VERSION}-android-sdk.zip
unzip opencv-${OPENCV_VERSION}-android-sdk.zip
export OPENCV_ANDROID_SDK_DIR=${PWD}/OpenCV-android-sdk
</code></pre>
    </td>

</tr>
  <tr>
    <td>ncnn </td>
    <td>A high-performance neural network inference computing framework supporting for android.</br>
  <b> Now, MMDeploy supports 20221128 and has to use <code>git clone</code> to download it. For supported android ABI, see <a href='https://github.com/Tencent/ncnn/releases'> here </a>. </b><br>
<pre><code>
git clone -b 20221128 https://github.com/Tencent/ncnn.git
cd ncnn
git submodule update --init
export NCNN_DIR=${PWD}

export ANDROID_ABI=arm64-v8a

mkdir -p build\_${ANDROID_ABI}
cd build\_${ANDROID_ABI}

cmake -DCMAKE_TOOLCHAIN_FILE=${NDK_PATH}/build/cmake/android.toolchain.cmake -DANDROID_ABI="${ANDROID_ABI}" -DANDROID_PLATFORM=android-30 -DNCNN_VULKAN=ON -DNCNN_DISABLE_EXCEPTION=OFF -DNCNN_DISABLE_RTTI=OFF ..
make -j$(nproc) install
</code></pre>

</td>
  </tr>
  <tr>
  <td>OpenJDK </td>
  <td>It is necessary for building Java API.</br>
  See <a href='https://github.com/open-mmlab/mmdeploy/blob/master/csrc/mmdeploy/apis/java/README.md'> Java API build </a> for building tutorials.
  </td>
  </tr>
</tbody>
</table>

### Build MMDeploy

#### Build SDK and Demos

MMDeploy provides a recipe as shown below for building SDK with ncnn as inference engine for android.

- cpu + ncnn
  ```Bash
  export ANDROID_ABI=arm64-v8a
  cd ${MMDEPLOY_DIR}
  mkdir -p build_${ANDROID_ABI} && cd build_${ANDROID_ABI}
  cmake .. \
      -DMMDEPLOY_BUILD_SDK=ON \
      -DMMDEPLOY_BUILD_SDK_JAVA_API=ON \
      -DMMDEPLOY_BUILD_EXAMPLES=ON \
      -DOpenCV_DIR=${OPENCV_ANDROID_SDK_DIR}/sdk/native/jni/abi-${ANDROID_ABI} \
      -Dncnn_DIR=${NCNN_DIR}/build_${ANDROID_ABI}/install/lib/cmake/ncnn \
      -DMMDEPLOY_TARGET_BACKENDS=ncnn \
      -DMMDEPLOY_SHARED_LIBS=ON \
      -DCMAKE_TOOLCHAIN_FILE=${NDK_PATH}/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=${ANDROID_ABI} \
      -DANDROID_PLATFORM=android-30 \
      -DANDROID_CPP_FEATURES="rtti exceptions"

  make -j$(nproc) && make install
  ```

Please check [cmake build option spec](cmake_option.md)
