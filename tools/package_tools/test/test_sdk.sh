set -e

WORKSPACE="."
MODEL_DIR="/__w/mmdeploy/testmodel/mmcls"
SDK_DIR="sdk"

if [[ -n "$1" ]]; then
    WORKSPACE=$1
fi

pushd $WORKSPACE
pushd $SDK_DIR

test_pkg=$(find "." -type d -iname "*-x86_64")
work_dir=/tmp/_test
cp -r $test_pkg $work_dir

pushd $work_dir

# opencv
if [ ! -d "$OpenCV_DIR" ]; then
    ./install_opencv.sh
fi

# env
source ./set_env.sh $(pwd)

# build
./build_sdk.sh $OpenCV_DIR

# run
./bin/classifier $MODEL_DIR $MODEL_DIR/demo.jpg

popd
rm -rf $work_dir
