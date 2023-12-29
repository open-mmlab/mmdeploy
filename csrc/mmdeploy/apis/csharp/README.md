# Installation

## Binaries

We provide nuget package on our [release page](https://github.com/open-mmlab/mmdeploy/releases). Currently the prebuilt package only support tensorrt and onnxruntiem backend.

To use the nuget package, you also need to download the backend dependencies. For example, if you want to use the tensorrt backend, you should install cudatoolkit, cudnn and tensorrt, remember to add the dll directories to your system path. The version of backend dependencies that our prebuit nuget package used will be offered in release note.

| backend     | dependencies                  |
| ----------- | ----------------------------- |
| tensorrt    | cudatoolkit, cudnn, tensorrt  |
| onnxruntime | onnxruntime / onnxruntime-gpu |

## From Source

### Requirements

- Environment required by building sdk
- .NET Framework 4.8 / .NET core 3.1
- Visual Studio 2019+

### Installation

**Step 0.** Build sdk.

Before building the c# api, you need to build sdk first. Please follow this [tutorial](../../../docs/en/build/windows.md)/[教程](../../../docs/zh_cn/build/windows.md) to build sdk. Remember to set the MMDEPLOY_BUILD_SDK_CSHARP_API option to ON. We recommend setting `MMDEPLOY_SHARED_LIBS` to OFF and use the static third party libraries(pplcv, opencv, etc.). If so, you only need add the backend dependencies to your system path, or you need to add all dependencies.

If you follow the tutorial, the mmdeploy.dll will be built in `build\bin\release`. Make sure the expected dll is in that path or the next step will throw a file-not-exist error.

**Step 1.** Build MMDeploy nuget package.

There are two methods to build the nuget package.

(*option 1*) Use the command.

If your environment is well prepared, you can just go to the `csrc\apis\csharp` folder, open a terminal and type the following command, the nupkg will be built in `csrc\apis\csharp\MMDeploy\bin\Release\MMDeployCSharp.1.3.1.nupkg`.

```shell
dotnet build --configuration Release -p:Version=1.3.1
```

(*option 2*) Open MMDeploy.sln && Build.

You can set the package-version through `Properties -> Package Version`. The default version is 1.3.1 if you don't set it.

If you encounter missing dependencies, follow the instructions for MSVC.
