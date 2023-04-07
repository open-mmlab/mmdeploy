$ErrorActionPreference = 'Stop'

$WORKSPACE = $PSScriptRoot
$OPENCV_DIR = ""

if ($args.Count -gt 0) {
    $OPENCV_DIR = $args[0]
    if (-Not (Test-Path -Path $OPENCV_DIR -PathType Container)) {
        Write-Error "OPENCV_DIR $OPENCV_DIR doesn't exist"
        Exit 1
    }
    $OPENCV_CONFIG = [IO.PATH]::Combine("$OPENCV_DIR", "OpenCVConfig.cmake")
    if (-Not (Test-Path -Path $OPENCV_CONFIG -PathType Leaf)) {
        Write-Error "OPENCV_DIR $OPENCV_DIR doesn't contains OpenCVConfig.cmake"
        Exit 1
    }
}

if ($OPENCV_DIR -eq "") {
    # search thirdparty
    $THIRDPARTY_DIR = "${WORKSPACE}/thirdparty"
    $THIRD_OPENCV = [IO.Path]::Combine("$THIRDPARTY_DIR", "opencv", "install")
    if (-Not (Test-Path $THIRD_OPENCV -PathType Container)) {
        Write-Error "Can't find opencv, please provide OPENCV_DIR or install it by install_opencv.ps1"
        Exit 1
    }
    $OPENCV_DIR = $THIRD_OPENCV
}

$MMDEPLOY_DIR = [IO.Path]::Combine("$WORKSPACE", "lib", "cmake", "MMDeploy")

$BUILD_DIR = "${WORKSPACE}/example/cpp/build"
if (Test-Path -Path $BUILD_DIR -PathType Container) {
    Remove-Item $BUILD_DIR -Recurse
}

New-Item -Path $BUILD_DIR -ItemType Directory
Push-Location $BUILD_DIR

Write-Host $MMDEPLOY_DIR

$MSVC_TOOLSET = "-T v142"
if ($env:CUDA_PATH -ne "") {
    $MSVC_TOOLSET = "$MSVC_TOOLSET,cuda=$env:CUDA_PATH"
    Write-Host $MSVC_TOOLSET
}

cmake .. -A x64 $MSVC_TOOLSET `
    -DMMDeploy_DIR="$MMDEPLOY_DIR" `
    -DOpenCV_DIR="$OPENCV_DIR"

cmake --build . --config Release

Pop-Location
