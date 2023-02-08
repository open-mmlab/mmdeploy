# Win10 下构建方式

- [Win10 下构建方式](#win10-下构建方式)
  - [源码安装](#源码安装)
    - [安装构建和编译工具链](#安装构建和编译工具链)
    - [安装依赖包](#安装依赖包)
      - [安装 MMDeploy Converter 依赖](#安装-mmdeploy-converter-依赖)
      - [安装推理引擎以及 MMDeploy SDK 依赖](#安装推理引擎以及-mmdeploy-sdk-依赖)
    - [编译 MMDeploy](#编译-mmdeploy)
    - [注意事项](#注意事项)

______________________________________________________________________

## 源码安装

下述安装方式，均是在 **Windows 10** 下进行，使用 **PowerShell Preview** 版本。

### 安装构建和编译工具链

1. 下载并安装 [Visual Studio 2019](https://visualstudio.microsoft.com) 。安装时请勾选 "使用C++的桌面开发, "Windows 10 SDK <br>
2. 把 cmake 路径加入到环境变量 PATH 中, "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\Common7\\IDE\\CommonExtensions\\Microsoft\\CMake\\CMake\\bin" <br>
3. 如果系统中配置了 NVIDIA 显卡，根据[官网教程](https://developer.nvidia.com\/cuda-downloads)，下载并安装 cuda toolkit。<br>

### 安装依赖包

#### 安装 MMDeploy Converter 依赖

<table class="docutils">
<thead>
  <tr>
    <th>名称 </th>
    <th>安装方法 </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>conda </td>
    <td>请参考 <a href="https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html">这里</a> 安装 conda。安装完毕后，打开系统开始菜单，<b>以管理员的身份打开 anaconda powershell prompt</b>。 因为，<br>
<b>1. 下文中的安装命令均是在 anaconda powershell 中测试验证的。</b><br>
<b>2. 使用管理员权限，可以把第三方库安装到系统目录。能够简化 MMDeploy 编译命令。</b><br>
<b>说明：如果你对 cmake 工作原理很熟悉，也可以使用普通用户权限打开 anaconda powershell prompt</b>。
</td>
  </tr>
  <tr>
    <td>PyTorch <br>(>=1.8.0) </td>
    <td> 安装 PyTorch，要求版本是 torch>=1.8.0。可查看<a href="https://pytorch.org/">官网</a>获取更多详细的安装教程。请确保 PyTorch 要求的 CUDA 版本和您主机的 CUDA 版本是一致<br>
<pre><code>
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
</code></pre>
    </td>
  </tr>
  <tr>
    <td>mmcv-full </td>
    <td>参考如下命令安装 mmcv-full。更多安装方式，可查看 <a href="https://github.com/open-mmlab/mmcv">mmcv 官网</a><br>
<pre><code>
$env:cu_version="cu111"
$env:torch_version="torch1.8"
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/$env:cu_version/$env:torch_version/index.html
</code></pre>
    </td>
  </tr>
</tbody>
</table>

#### 安装推理引擎以及 MMDeploy SDK 依赖

可以通过安装脚本简化相关依赖的安装，首先根据需求修改`.\tools\scripts\mmdeploy_init.ps1` 中的`Model converter && SDK config` 部分。下面以编译TensorRT自定义算子以及SDK为例展示脚本的用法，详细配置说明可参考 [cmake 选项说明](./cmake_option.md).

- a) 修改配置参数如下：
  ```
  $CMAKE_BUILD_TYPE = "Release"
  $MMDEPLOY_TARGET_BACKENDS = "trt"
  $tensorrtVersion = "8.2.3"
  $cudnnVersion = "8.2.1"
  $cudaVersion = "11.x"
  $MMDEPLOY_BUILD_SDK = "ON"
  $MMDEPLOY_TARGET_DEVICES = "cpu;cuda"
  $opencvVersion = "4.5.5"
  $MMDEPLOY_CODEBASES = "all"
  ```
- b) 在 MMDeploy 根目录下执行
  ```powershell
  .\mmdeploy_init.ps1 -Action Download
  ```
  该操作会根据配置信息自动下载相关依赖到`3rdparty`文件夹，并将相关变量写入`env.txt`文件。

### 编译 MMDeploy

- a) 编译自定义算子以及SDK

  ```powershell
  .\tools\scripts\mmdeploy_init.ps1 -Action Build
  ```

  该操作会先试图读取`env.txt`文件中的变量，如果不存在该文件，则会自动执行`Download`操作，请确保按需求修改了`Model converter && SDK config`。
  之后会编译自定义算子和SDK(如果打开了SDK编译选项)。模型转换需要的自定义算子库会安装在`\mmdeploy\lib`目录下，SDK相关文件会安装在`.\build\install`目录下

- b) 安装MMDeploy的python转换工具

  ```
  pip install -e .
  ```

### 注意事项

1. Release / Debug 库不能混用。MMDeploy 要是编译 Release 版本，所有第三方依赖都要是 Release 版本。反之亦然。
