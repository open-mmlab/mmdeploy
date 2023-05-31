$ErrorActionPreference = 'Stop'

$WORKSPACE = ""
$MODEL_DIR = "D:\DEPS\citest\mmcls"
$SDK_PYTHON_DIR = "mmdeploy_runtime"

if ($args.Count -gt 0) {
    $WORKSPACE = $args[0]
}

Push-Location $WORKSPACE
Push-Location $SDK_PYTHON_DIR

$pkgs = $(ls).Name
$test_pkg = ""
if ($pkgs.Count -gt 1) {
    foreach ($pkg in $pkgs) {
        if ($pkg -like 'mmdeploy_runtime-*cp38*-win_amd64.whl') {
            $test_pkg = $pkg
            break
        }
    }
}

pip install $test_pkg --force-reinstall

$code = "
import cv2
from mmdeploy_runtime import Classifier
import sys
handle = Classifier('$MODEL_DIR', 'cpu', 0)
img = cv2.imread('$MODEL_DIR\demo.jpg')
try:
    res = handle(img)
    print(res)
except:
    print('error')
    sys.exit(1)
"

python -c $code

Pop-Location
Pop-Location
