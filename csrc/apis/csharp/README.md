## Installation

1. Build mmdeploy sdk with DMMDEPLOY_BUILD_SDK=ON.
a) When build mmdeploy sdk, there will generate MMDeploySharpExtern.dll (may located in build\bin\Release, depend on your building).
b) Add third party engine runtime to your path(tensorrt, onnxruntime for example, depend on what engine the sdk built with). 
c) It is better to build stacic library (mmdeploy、spdlog、opencv). If so, you only need add MMDeploySharpExtern.dll to the system path, or you need add spdlog, mmdeploy and opencv runtime to the system path additionally.

2. Open MMDeploySharp.sln && Build MMDeploySharp
Open csrc\apis\csharp\MMDeploySharp.sln, generate solution, and it will generate a nuget package located in csrc\apis\csharp\src\MMDeploySharp\bin\{Debug\Release}\MMDeploySharp.1.0.0.nupkg
