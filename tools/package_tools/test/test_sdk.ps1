$ErrorActionPreference = 'Stop'

$WORKSPACE = ""
$MODEL_DIR = "D:\DEPS\citest\mmcls"
$SDK_DIR = "sdk"

if ($args.Count -gt 0) {
    $WORKSPACE = $args[0]
}

Push-Location $WORKSPACE
Push-Location $SDK_DIR

$pkgs = $(ls).Name
$test_pkg = $pkgs[0]
if ($pkgs.Count -gt 1) {
    foreach ($pkg in $pkgs) {
        if ($pkg -like '*cpu*') {
            $test_pkg = $pkg
            break
        }
    }
}

Push-Location $test_pkg


# opencv
$OPENCV = ""
if (-Not (Test-Path $env:OpenCV_DIR)) {
    .\install_opencv.ps1
}

# env
. .\set_env.ps1

# build
.\build_sdk.ps1 $env:OpenCV_DIR

# run
write-host "run run"
.\example\cpp\build\Release\classifier.exe "D:\DEPS\citest\mmcls" "$MODEL_DIR\demo.jpg"

Pop-Location
Pop-Location
Pop-Location
