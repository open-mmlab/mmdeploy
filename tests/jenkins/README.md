# 目录结构

```shell
tree

.
├── docker
│   ├── mmdeploy-ci-ubuntu-18.04
│   │   └── Dockerfile
│   ├── mmdeploy-ci-ubuntu-18.04-cu102
│   │   └── Dockerfile
│   ├── mmdeploy-ci-ubuntu-20.04-cu111
│   │   └── Dockerfile
│   ├── mmdeploy-ci-ubuntu-20.04-cu113
│   │   └── Dockerfile
├── Jenkinsfile
├── README.md
└── scripts
    ├── docker_exec_build.sh
    ├── docker_exec_convert_gpu.sh
    ├── docker_exec_convert.sh
    ├── docker_exec_prebuild.sh
    ├── docker_exec_ut.sh
    ├── test_build.sh
    ├── test_convert.sh
    ├── test_prebuild.sh
    └── test_ut.sh
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

## jenkins
存放执行任务所需shell脚本，命名方式为：
```shell
test_${Job_Type}.sh ## 执行任务的入口脚本
docker_exec_${Job_Type}.sh ## 在容器中实际运行的脚本
```
- test_${Job_Type}.sh:  
  - parameters
    - docker_image: str: 指定执行任务的docker image
    - codebase_list: str: 部分脚本需要，执行执行的codebase(简写)，多个codebase用空格连接，并用" "包起，作为字符串参数传入
  - log存放路径
  - 创建运行时container(一个或多个)
  - 在容器中git clone mmdeploy
  - 在容器中执行docker_exec_${Job_Type}.sh脚本
- docker_exec_${Job_Type}.sh:
  - parameters
    - codebase_list: str: 部分脚本需要，执行执行的codebase(简写)，多个codebase用空格连接，并用" "包起，作为字符串参数传入
  - 实际执行的任务步骤
## Jenkinsfile
Jenkins执行任务时所需的pipeline配置文件

# 如何运行
## step１
在执行机上执行docker build ${image}执行，build镜像
## step2
连接至执行机，进行mmdeploy目录，执行test_${Job_Type}.sh脚本并传入参数，以执行mmdet mmcls在mmdeploy-ci-ubuntu-20.04-cu113的convert任务为例
```shell
cd the/path/mmdeploy
./tests/jenkins/scripts/test_convert mmdeploy-ci-ubuntu-20.04-cu113 'mmdet mmcls'
```
## step3
shell返回container='xxx'，任务开始运行
## step4
等待任务运行完成，查看日志

# 如何查看日志
## 日志存放路径
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

