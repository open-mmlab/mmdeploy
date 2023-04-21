if ($args.Count -lt 2) {
    Write-Host "wrong command. usage: intall_onnxruntime.ps1 <cpu|cuda> <version>"
    Exit 1
}

$platform = $args[0]
$version = $args[1]

if ($platform -eq "cpu") {
    python -m pip install onnxruntime==$version
    Invoke-WebRequest -Uri https://github.com/microsoft/onnxruntime/releases/download/v$version/onnxruntime-win-x64-$version.zip -OutFile onnxruntime.zip
    Expand-Archive onnxruntime.zip .
    Move-Item onnxruntime-win-x64-$version onnxruntime
} elseif ($platform == "cuda") {
    Write-Host "TODO: install onnxruntime-gpu"
    Exit
} else {
    Write-Host "'$platform' is not supported"
}
