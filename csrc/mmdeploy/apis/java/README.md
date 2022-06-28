# Build Java API

## From Source

### Requirements

- OpenJDK >= 10

**Step 1.** Download OpenJDK. Using OpenJDK-18 as example:

```bash
wget https://download.java.net/java/GA/jdk18/43f95e8614114aeaa8e8a5fcf20a682d/36/GPL/openjdk-18_linux-x64_bin.tar.gz
tar xvf openjdk-18_linux-x64_bin.tar.gz
```

**Step 2.** Setting environment variables:

```bash
export JAVA_HOME=${PWD}/jdk-18
export PATH=${JAVA_HOME}/bin:${PATH}
```

**Step 3.** Switch default Java version:

```bash
sudo update-alternatives --config java
sudo update-alternatives --config javac
```

You should select the version you will use.

### Installation

For using Java apis, you should build Java class and C++ SDK.

**Step 1.** Build Java class.

Build Java `.class` files.

```bash
cd csrc/mmdeploy/apis/java
javac mmdeploy/*.java
cd ../../../..
```

**Step 2.** Build SDK.

Build MMDeploy SDK. Please follow this [tutorial](../../../../docs/en/01-how-to-build/linux-x86_64.md)/[教程](../../../../docs/zh_cn/01-how-to-build/linux-x86_64.md) to build SDK. Remember to set the MMDEPLOY_BUILD_SDK_JAVA_API option to ON.
