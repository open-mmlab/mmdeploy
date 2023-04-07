# build sdk

1. installed opencv (you can skip this step if you have installed it)
   in sdk folder:

   `./install_opencv.sh`

2. set environment variable and path
   in sdk folder:

   `source ./set_env.sh`

   (**you have to additionally install cuda and cudnn if use sdk cuda version**)

   (**may need to set CUDNN environment variable point to cudnn root folder if use sdk cuda version**)

3. build sdk
   in sdk folder:

   `./build_sdk.sh` \
   (if you installed opencv by ./install_opencv.sh)

   or

   `./build_sdk.sh "path/to/folder/of/OpenCVConfig.cmake"` \
   (if you installed opencv yourself)

   the executable will be generated in: `bin/`
