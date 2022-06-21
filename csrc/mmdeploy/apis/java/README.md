# Build Java API

## From Source

### Requirements

- OpenJDK >= 10

### Installation

For using java apis, you should build java class and c++ sdk.

**Step 1.** Build java class.

Build java `.class` files.

```bash
cd csrc/mmdeploy/apis/java
javac mmdeploy/*.java
cd ../../../..
```

**Step 2.** Build sdk.

Build mmdeploy sdk. Please follow this [tutorial](../../../../docs/en/01-how-to-build/linux-x86_64.md)/[教程](../../../../docs/zh_cn/01-how-to-build/linux-x86_64.md) to build sdk. Remember to set the MMDEPLOY_BUILD_SDK_JAVA_API option to ON.
