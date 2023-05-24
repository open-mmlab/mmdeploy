set -e

WORKSPACE="."
MODEL_DIR="/__w/mmdeploy/testmodel/mmcls"
SDK_PYTHON_DIR="mmdeploy_runtime"

if [[ -n "$1" ]]; then
    WORKSPACE=$1
fi

cd $WORKSPACE
cd $SDK_PYTHON_DIR

PY_VERSION=$(python3 -V | awk '{print $2}' | awk '{split($0, a, "."); print a[1]a[2]}')
test_pkg=$(ls | grep mmdeploy_runtime-*cp${PY_VERSION}*x86_64.whl)

python3 -m pip install $test_pkg --force-reinstall
python3 -m pip install opencv-python

code="
import cv2
from mmdeploy_runtime import Classifier
import sys
handle = Classifier('$MODEL_DIR', 'cpu', 0)
img = cv2.imread('$MODEL_DIR/demo.jpg')
try:
    res = handle(img)
    print(res)
except:
    print('error')
    sys.exit(1)
"

python3 -c "$code"
