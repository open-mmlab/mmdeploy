set -e

WORKSPACE="."
MODEL_DIR="/__w/mmdeploy/testmodel/mmcls"
SDK_DIR="sdk"

if [[ -n "$1" ]]; then
    WORKSPACE=$1
fi

pushd $WORKSPACE
pushd $SDK_DIR

test_pkg=$(ls | grep *cpu*)
pushd $test_pkg


# opencv
if [[ ! -d $OpenCV_DIR ]]; then
    ./install_opencv.sh
fi

# env
source ./set_env.sh $(pwd)

# build
./build_sdk.sh $OpenCV_DIR

# run
./bin/classifier $MODEL_DIR $MODEL_DIR/demo.jpg