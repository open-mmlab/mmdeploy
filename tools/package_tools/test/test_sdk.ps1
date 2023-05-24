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
        if ($pkg -like '*-windows-amd64') {
            $test_pkg = $pkg
            break
        }
    }
}

$work_dir=[IO.Path]::Combine("$env:TMP", [guid]::NewGuid().ToString())
Copy-item $test_pkg $work_dir -Recurse
Push-Location $work_dir


# opencv
if (-Not (Test-Path $env:OpenCV_DIR)) {
    .\install_opencv.ps1
}

# env
. .\set_env.ps1

# build
.\build_sdk.ps1 $env:OpenCV_DIR

# run
.\example\cpp\build\Release\classifier.exe "D:\DEPS\citest\mmcls" "$MODEL_DIR\demo.jpg"

Pop-Location
Remove-Item $work_dir -Recurse

Pop-Location
Pop-Location
