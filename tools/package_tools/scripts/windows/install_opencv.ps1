$opencvVer = "4.5.5"
# ----

$ErrorActionPreference = 'Stop'
$WORKSPACE = $PSScriptRoot
$THIRDPARTY_DIR = "${WORKSPACE}/thirdparty"
$OPENCV_DIR = "${THIRDPARTY_DIR}/opencv/install"

if (-Not (Test-Path -Path $THIRDPARTY_DIR -PathType Container)) {
    New-Item -Path $THIRDPARTY_DIR -ItemType Directory
}

Push-Location "${THIRDPARTY_DIR}"

$url = "https://github.com/opencv/opencv/archive/refs/tags/$opencvVer.zip"
$fileName = [IO.Path]::GetFileName($url)
Start-BitsTransfer $url $fileName
Expand-Archive -Path $fileName -DestinationPath "." -Force
Move-Item "opencv-$opencvVer" "opencv"
Push-Location "opencv"
New-Item -Path "build" -ItemType Directory
Push-Location build

cmake .. -A x64 -T v142 `
    -DBUILD_TESTS=OFF `
    -DBUILD_PERF_TESTS=OFF `
    -DCMAKE_INSTALL_PREFIX="${OPENCV_DIR}"

cmake --build . --config Release -j6
cmake --install . --config Release

Pop-Location
Pop-Location
Pop-Location
