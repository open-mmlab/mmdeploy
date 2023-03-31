# VACC Backend

- cmake 3.10.0+
- gcc/g++ 7.5.0
- llvm 9.0.1
- ubuntu 18.04

## PCIE

### 1.package

- dkms (>=1.95)
- linux-headers
- dpkg (Ubuntu)
- rpm  (CentOS)
- python2
- python3

查看是否有瀚博推理卡：`lspci -d:0100`

1. 环境准备

   ```bash
   sudo apt-get install dkms dpkg python2 python3
   ```

2. driver安装

   ```bash
   sudo dpkg -i vastai-pci_xx.xx.xx.xx_xx.deb
   ```

3. 查看安装

   ```bash
   # 1.查看deb包是否安装成功
   dpkg --status vastai-pci-xxx

   #output
   Package: vastai-pci-dkms
   Status: install ok installed
   ……
   Version: xx.xx.xx.xx
   Provides: vastai-pci-modules (= xx.xx.xx.xx)
   Depends: dkms (>= 1.95)
   Description: vastai-pci driver in DKMS format.

   # 2.查看驱动是否已加载到内核
   lsmod | grep vastai_pci

   #output
   vastai_pci        xxx  x
   ```

4. 升级驱动

   ```bash
   sudo dpkg -i vastai-pci_dkms_xx.xx.xx.xx_xx.deb
   ```

5. 卸载驱动

   ```bash
   sudo dpkg -r vastai-pci_dkms_xx.xx.xx.xx_xx
   ```

### 2.reboot pcie

```bash
sudo chmod 666 /dev/kchar:0 && sudo echo reboot > /dev/kchar:0
```

## SDK

### step.1

```bash
pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0
pip install onnx==1.10.0 tqdm==4.64.1
pip install h5py==3.8.0
pip install decorator==5.1.1 scipy==1.7.3
```

### step.2

```bash
sudo vi ~/.bashrc

export VASTSTREAM_PIPELINE=true
export VACC_IRTEXT_ENABLE=1
export TVM_HOME="/opt/vastai/vaststream/tvm"
export VASTSTREAM_HOME="/opt/vastai/vaststream/vacl"
export LD_LIBRARY_PATH=$TVM_HOME/lib:$VASTSTREAM_HOME/lib
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/vacc/python:$TVM_HOME/topi/python:${PYTHONPATH}:$VASTSTREAM_HOME/python

source ~/.bashrc
```
