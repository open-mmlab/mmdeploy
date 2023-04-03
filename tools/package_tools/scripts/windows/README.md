# build sdk

1. open windows powerShell with administrator privileges
   set-ExecutionPolicy RemoteSigned

2. installed opencv (you can skip this step if you have installed it)
   in sdk folder:

   `.\install_opencv.ps1`

3. set environment variable and path
   in sdk folder:

   `. .\set_env.ps1`

   if you use sdk cuda version:

   (**you have to additionally install cuda and cudnn if use sdk cuda version**)

   (**may need to set CUDNN environment variable point to cudnn root folder if use sdk cuda version**)

4. build sdk
   in sdk folder:

   `. .\build_sdk.ps1` \
   (if you installed opencv by install_opencv.ps1)

   or

   `. .\build_sdk.ps1 "path/to/folder/of/OpenCVConfig.cmake"` \
   (if you installed opencv yourself)

   the executable will be generated in:
   `example\cpp\build\Release`
