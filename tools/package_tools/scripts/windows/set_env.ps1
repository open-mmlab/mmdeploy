$WORKSPACE = $PSScriptRoot
$THIRDPARTY_DIR = "${WORKSPACE}/thirdparty"
Push-Location $THIRDPARTY_DIR

if (Test-Path -Path "onnxruntime" -PathType Container) {
    $dir = [IO.Path]::Combine("$pwd", "onnxruntime")
    $env:ONNXRUNTIME_DIR = $dir
    $path = [IO.Path]::Combine("$dir", "lib")
    $env:PATH = "$path;$env:PATH"
}

if (Test-Path -Path "tensorrt" -PathType Container) {
    $dir = [IO.Path]::Combine("$pwd", "tensorrt")
    $env:TENSORRT_DIR = $dir
    $path = [IO.Path]::Combine("$dir", "lib")
    $env:PATH = "$path;$env:PATH"
}

if (Test-Path -Path "openvino" -PathType Container) {
    $root = [IO.Path]::Combine("$pwd", "openvino")
    $dir = [IO.Path]::Combine("root", "runtime", "cmake")
    $env:InferenceEngine_DIR = $dir
    $paths = Get-ChildItem -Path $root -Filter "*.dll" -Recurse | `
        ForEach-Object { $_.Directory.FullName } | Get-Unique
    foreach ($path in $paths) {
        $env:PATH = "$path;$env:PATH"
        Write-Host $path
    }
}

$path = [IO.Path]::Combine("$WORKSPACE", "bin")
$env:PATH = "$path;$env:PATH"


Pop-Location
