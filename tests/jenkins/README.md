# 目录结构

```shell
tree

.
├── Jenkinsfile
├── win_Jenkinsfile
├── README.md
├── conf
│   ├── Linux-2080_cuda-102_dev-1.x_full-test.config
│   ├── Linux-2080_cuda-102_master_full-test.config
│   ├── Linux-2080_cuda-111_dev-1.x_full-test.config
│   ├── Linux-2080_cuda-111_master_full-test.config
│   ├── Linux-3090_cuda-111_master_full-test.config
│   ├── Linux-3090_cuda-113_dev-1.x_full-test.config
│   ├── Linux-3090_cuda-113_dev-1.x_quick-test.config
│   ├── Linux-3090_cuda-113_master_full-test.config
│   ├── Linux-2080_cuda-113_master_quick-test.config
│   ├── README.md
│   ├── Windows-3080_cuda-113_master_full-test.config
│   ├── requirementV1.0.json
│   ├── requirementV2.0.json
│   ├── tmp.config
│   └── win_default.config
├── docker
│   ├── mmdeploy-ci-ubuntu-18.04-cu111
│   │   └── Dockerfile
│   ├── mmdeploy-ci-ubuntu-18.04-cu102
│   │   └── Dockerfile
│   ├── mmdeploy-ci-ubuntu-20.04-cu111
│   │   └── Dockerfile
│   └── mmdeploy-ci-ubuntu-20.04-cu113
│       └── Dockerfile
├── scripts
│   ├── docker_exec_build.sh
│   ├── check_results.py
│   ├── docker_exec_convert_gpu.sh
│   ├── docker_exec_prebuild.sh
│   ├── docker_exec_ut.sh
│   ├── get_log.py
│   ├── get_requirements.py
│   ├── test_build.sh
│   ├── test_convert.ps1
│   ├── test_convert.sh
│   ├── test_prebuild.sh
│   ├── test_ut.sh
│   ├── utils.psm1
│   ├── win_convert_exec.ps1
│   └── win_default.config
└── todolist.md

```

## conf

存放脚本运行时的配置文件分成linux和windows系统的默认配置文件
windows配置如下：

- CUDA版本（cu111, cu113可选）
- 支持的codebase
- 是否进行精度测试
- 最大线程数
- mmdeploy的分支
- mmdeploy仓库的地址

linux配置如下：

- CUDA版本（cu102,cu111, cu113可选）
- 显卡型号（2080,3090可选）
- docker镜像
- 支持的codebase
- torch版本选择
- 不同模型精度测试
- 精度测试后端选择
- 是否进行精度测试
- 最大线程数
- mmdeploy的分支
- mmdeploy仓库的地址和版本
- tensorrt版本
- 指定算法库的分支和依赖

```
tmp.config #测试版配置
```

## docker

存放Dockerfile文件位置，命名方式为：

```shell
mmdeploy-ci-${OS}-${OS_Version}-${CUDA_Version}
```

Dockerfile中会安装：

- 已编译安装的各个后端及所需环境变量
- 所需apt包
- 所需python包
- 多个conda环境，并分别安装不同torch版本

## scripts

存放执行任务所需shell脚本，命名方式为：

### linux

```shell
test_${Job_Type}.sh ## 执行任务的入口脚本
docker_exec_${Job_Type}.sh ## 在容器中实际运行的脚本


```

- test\_${Job_Type}.sh:
  - parameters
    - docker_image: str: 指定执行任务的docker image
    - codebase_list: str: 部分脚本需要，执行执行的codebase(简写)
    - exec_performance: str: 部分脚本需要，执行性能测试
    - max_job_nums：int: 部分脚本需要，多线程执行脚本
    - mmdeploy_branch str: 选择mmdeploy分支
    - repo_url: str: mmdeploy地址
    - tensorrt_version: str: 部分脚本需要，选择tensorrt版本
  - 读取配置文件
  - log存放路径
  - 创建运行时container(一个或多个)
  - 在容器中git clone mmdeploy
  - 在容器中执行docker_exec\_${Job_Type}.sh脚本
- docker_exec\_${Job_Type}.sh:
  - parameters
    - codebase_list: str: 部分脚本需要，执行执行的codebase(简写)，多个codebase用空格连接，并用" "包起，作为字符串参数传入
  - 实际执行的任务步骤

### windows

```shell
test_convert.ps1 ## 执行任务的入口
win_convert_exec.ps1 ## 回归测试的实际运行的脚本
utils.psm1 ## 工具类
```

- test_convert.ps1:
  - parameters
    - cuda_version: str: 选择cuda版本（cu111, cu113可选）
    - codebase_list: str: 部分脚本需要，执行执行的codebase(简写)
    - exec_performance: str: 部分脚本需要，执行性能测试
    - max_job_nums：int: 部分脚本需要，多线程执行脚本
    - mmdeploy_branch str: 选择mmdeploy分支
    - repo_url: str: mmdeploy地址
  - 读取配置文件
  - 切换cuda版本
  - git clone codebase
  - git clone mmdeploy
  - 设置环境变量
  - 编译mmdeploy sdk
  - log存放路径
  - 多线程执行win_convert_exec.ps1
- win_convert_exec.ps1
  - parameters
    - codebase: str: codebase简写
    - exec_performance: str: 执行性能测试
    - codebase_fullname_opt:  hashtable: codebase简写和全名的映射集合
  - 根据codebase安装配套mmcv
  - 执行回归测试

## Jenkinsfile

Jenkins执行任务时所需的pipeline配置文件

# 如何构建镜像

## step1

由于部分文件需要登录才能下载(主要是nvidia相关)，因此需要先手动下载这些包，并存放至本机/data2/shared(可通过修改dockerfile来自定义路径)，需要下载的包为dockerfile中`http://${HOST}/`后所跟文件

## step2

在/data2/shared中启动一个文件下载服务

```python
cd /data2/shared
nohup python3 -m http.server 9000 > file_server.log 2>&1 &
```

## step3

执行构建命令

```shell
cd /the/path/mmdeploy
docker build ./tests/jenkins/docker/mmdeploy-ci-${OS}-${OS_Version}-${CUDA_Version} \
--build-arg HOST=${host_ip}:9000 \
-t mmdeploy-ci-${OS}-${OS_Version}-${CUDA_Version}
```

# 如何运行

## step１

在执行机上build镜像

## step2

连接至执行机，进行mmdeploy目录，执行test\_${Job_Type}.sh脚本并传入配置文件参数，未指定配置文件时默认使用default.config

```shell
cd the/path/mmdeploy
./tests/jenkins/scripts/test_convert tmp.config
```

## step3

shell返回container='xxx'，任务开始运行

## step4

等待任务运行完成，查看日志

# 如何查看日志

## linux日志存放路径

所有运行日志均存放在执行机的/data2/regression_log中

```shell
cd /data2/regression_log
tree -L 1 -d
## 分别存放不同类型任务的log
.
├── build_log
├── convert_log
├── prebuild_log
└── ut_log

cd convert_log
tree -L 1 -d

.
├── 20220903 # 一级目录，年/月/日
    ├── 202209051049 # 二级目录，年/月/日/时/分
    │   ├── mmrotate # 三级目录，区分codebase
    │   │   ├── torch1.10.0 # 四级目录，区分不同torch_version
    │   │   │   ├── convert.log
    │   │   │   └── mmrotate_report.txt
    │   │   └── torch1.11.0
    │   │       ├── convert.log
    │   │       └── mmrotate_report.txt
    │   └── mmrotate.log
    └── 202209051054
        ├── mmrotate
        │   ├── torch1.10.0
        │   │   ├── convert.log
        │   │   └── mmrotate_report.txt
        │   └── torch1.11.0
        │       ├── convert.log
        │       └── mmrotate_report.txt
        └── mmrotate.log

├── 20220903 # 一级目录，年/月/日
│   ├── 202209031533 # 二级目录，年/月/日/时/分
│   │   ├── mmcls # 三级目录，区分codebase
│   │   │   ├── torch1.10.0 # 四级目录，区分不同torch_version
│   │   │   │   ├── convert.log # 转换任务log
│   │   │   │   ├── mmcls # 不同backend/model执行日志
│   │   │   │   ├── mmcls_report.txt # regression执行结果
│   │   │   │   ├── mmcls_report.xlsx
│   │   │   │   └── mmdeploy_regression_test_0.7.0.xlsx
│   │   │   └── torch1.11.0
│   │   │       ├── convert.log
│   │   │       ├── mmcls
│   │   │       ├── mmcls_report.txt
│   │   │       ├── mmcls_report.xlsx
│   │   │       └── mmdeploy_regression_test_0.7.0.xlsx
│   │   └── mmcls.log # 任务全过程日志，排查问题使用
```

## 日志查看方式

1. 直接连接到执行机，通过cat/tail/less等命令查看
2. 浏览器访问宿主机IP:8989，远程查看日志目录，下载相应日志查看

```shell
${exec_host_ip}:8989
```

## windows查看日志

所有运行日志均存放在执行机的与mmdeploy同级目录regression_log中
目录结构参考linux日志目录结构

# 各个docker镜像包含内容

## mmdeploy-ci-ubuntu-18.04-cu102

- miniconda
- cmake3.24
- jdk1.8
- opencv4.5
- TensorRT
  - TensorRT-8.4.3.1-cudnn8.4
  - TensorRT-8.2.5.1-cudnn8.2
- cudnn
  - cudnn-8.2.4.15
  - cudnn-8.4.1.50
- onnxruntime1.8.1
- pplnn
- ncnn
- pplcv
- conda env
  - torch1.8.1
  - torch1.9.0
  - torch1.10.0
  - torch1.11.0
  - torch1.12.0

## mmdeploy-ci-ubuntu-20.04-cu111

- TensorRT
  - TensorRT-8.4.1.5-cudnn8.4
  - TensorRT-8.2.5.1-cudnn8.2
- cudnn
  - cudnn-8.2.1.32
  - cudnn-8.4.1.50

## mmdeploy-ci-ubuntu-20.04-cu113

- miniconda
- cmake3.24.1
- jdk1.8
- opencv4.5
- TensorRT
  - TensorRT-8.4.1.5-cudnn8.4
  - TensorRT-8.2.5.1-cudnn8.2
- cudnn
  - cudnn-8.2.1.32
  - cudnn-8.4.1.50
    onnxruntime1.8.1
- pplnn0.8.1
- ncnn20220721
- pplcv0.7.0
- conda env
  - torch1.10.0
  - torch1.11.0
  - torch1.12.0

## windows环境

- OpenCV 4.6.0
- pplcv
- ONNXRuntime>=1.8.1
- TensorRT-8.2.3.0.Windows10.x86_64.cuda-11.4.cudnn8.2
- cudnn-11.3-windows-x64-v8.2.1.32
