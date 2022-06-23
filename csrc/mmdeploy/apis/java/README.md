# Build Java API

## From Source

### Requirements

- OpenJDK >= 10

### Installation

For using Java apis, you should build Java class and C++ SDK.

**Step 1.** Build Java class.

Build Java `.class` files.

```bash
cd csrc/mmdeploy/apis/Java
Javac mmdeploy/*.Java
cd ../../../..
```

**Step 2.** Build SDK.

Build MMDeploy SDK. Please follow this [tutorial](../../../../docs/en/01-how-to-build/linux-x86_64.md)/[教程](../../../../docs/zh_cn/01-how-to-build/linux-x86_64.md) to build SDK. Remember to set the MMDEPLOY_BUILD_SDK_JAVA_API option to ON.
