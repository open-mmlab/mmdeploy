# Android 下构建方式

- [Android 下构建方式](#android-下构建方式)
  - [源码安装](#源码安装)
    - [安装构建和编译工具链](#安装构建和编译工具链)
    - [安装依赖包](#安装依赖包)
      - [安装 MMDeploy SDK 依赖](#安装-mmdeploy-sdk-依赖)
    - [编译 MMDeploy](#编译-mmdeploy)
      - [编译 SDK 和 Demos](#编译-sdk-和-demos)

______________________________________________________________________

MMDeploy 为 android 平台提供交叉编译的构建方式.

MMDeploy converter 部分在 linux 平台上执行,SDK 部分在 android 平台上执行.

MMDeploy 的交叉编译分为两步:

1. 在 linux 平台上构建 MMDeploy converter. 请根据 [How to build linux](linux-x86_64.md) 进行构建.

2. 使用 android 工具链构建 MMDeploy SDK.

本文档仅提供在 linux 平台上使用 android 工具链进行交叉编译构建 MMDeploy SDK 的方法.

## 源码安装

### 安装构建和编译工具链

- cmake

  **保证 cmake的版本 >= 3.14.0**. 如果不是,可以参考以下命令安装 3.20.0 版本. 如要获取其他版本,请参考 [这里](https://cmake.org/install)

  ```bash
  wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0-linux-x86_64.tar.gz
  tar -xzvf cmake-3.20.0-linux-x86_64.tar.gz
  sudo ln -sf $(pwd)/cmake-3.20.0-linux-x86_64/bin/* /usr/bin/
  ```

- ANDROID NDK 19+

  **保证 android ndk 的版本 >= 19.0**. 如果不是,可以参考以下命令安装 r23c 版本. 如要获取其他版本,请参考 [这里](https://developer.android.com/ndk/downloads)

  ```bash
  wget https://dl.google.com/android/repository/android-ndk-r23c-linux.zip
  unzip android-ndk-r23c-linux.zip
  cd android-ndk-r23c
  export NDK_PATH=${PWD}
  ```

### 安装依赖包

#### 安装 MMDeploy SDK 依赖

如果您只对模型转换感兴趣,那么可以跳过本章节.

<table>
<thead>
  <tr>
    <th>名称 </th>
    <th>安装方式 </th>
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
    <td>ncnn 是支持 android 平台的高效神经网络推理计算框架</br>
  <b> 目前, MMDeploy 支持 ncnn 的 20220721 版本, 且必须使用<code>git clone</code> 下载源码的方式安装。请到 <a href='https://github.com/Tencent/ncnn/releases'> 这里 </a> 查询 ncnn 支持的 android ABI。</b><br>

<pre><code>
git clone -b 20220721 https://github.com/Tencent/ncnn.git
cd ncnn
git submodule update --init
export NCNN_DIR=${PWD}

export ANDROID_ABI=arm64-v8a

mkdir -p build_${ANDROID_ABI}
cd build_${ANDROID_ABI}

cmake -DCMAKE_TOOLCHAIN_FILE=${NDK_PATH}/build/cmake/android.toolchain.cmake -DANDROID_ABI="${ANDROID_ABI}" -DANDROID_PLATFORM=android-30 -DNCNN_VULKAN=ON -DNCNN_DISABLE_EXCEPTION=OFF -DNCNN_DISABLE_RTTI=OFF -DANDROID_USE_LEGACY_TOOLCHAIN_FILE=False ..
make -j$(nproc) install
</code></pre>

</td>
  </tr>
    <tr>
  <td>OpenJDK </td>
  <td>编译Java API之前需要先准备OpenJDK开发环境</br>
  请参考 <a href='https://github.com/open-mmlab/mmdeploy/tree/main/csrc/mmdeploy/apis/java/README.md'> Java API 编译 </a> 进行构建.
  </td>
  </tr>
</tbody>
</table>

### 编译 MMDeploy

#### 编译 SDK 和 Demos

下文展示构建 SDK 的样例，用 ncnn 作为推理引擎。

- cpu + ncnn
  ```Bash
  export ANDROID_ABI=arm64-v8a
  cd ${MMDEPLOY_DIR}
  mkdir -p build_${ANDROID_ABI} && cd build_${ANDROID_ABI}
  cmake .. \
      -DMMDEPLOY_BUILD_SDK=ON \
      -DMMDEPLOY_BUILD_EXAMPLES=ON \
      -DMMDEPLOY_BUILD_SDK_JAVA_API=ON \
      -DOpenCV_DIR=${OPENCV_ANDROID_SDK_DIR}/sdk/native/jni/abi-${ANDROID_ABI} \
      -Dncnn_DIR=${NCNN_DIR}/build_${ANDROID_ABI}/install/lib/cmake/ncnn \
      -DMMDEPLOY_TARGET_BACKENDS=ncnn \
      -DMMDEPLOY_SHARED_LIBS=OFF \
      -DCMAKE_TOOLCHAIN_FILE=${NDK_PATH}/build/cmake/android.toolchain.cmake \
      -DANDROID_USE_LEGACY_TOOLCHAIN_FILE=False \
      -DANDROID_ABI=${ANDROID_ABI} \
      -DANDROID_PLATFORM=android-30 \
      -DANDROID_CPP_FEATURES="rtti exceptions"

  make -j$(nproc) && make install
  ```

参考 [cmake 选项说明](cmake_option.md)
