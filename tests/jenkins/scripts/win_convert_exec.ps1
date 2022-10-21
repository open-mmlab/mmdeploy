param(
    $codebase,
    $exec_performance,
    $codebase_fullname
)
$scriptDir = Split-Path -parent $MyInvocation.MyCommand.Path
Import-Module $scriptDir\utils.psm1
Write-Host "exec_path: $pwd"
Write-Host "mim install $codebase"
Write-Host "codebase_fullname = $codebase_fullname"
Write-Host "exec_performance = $exec_performance"
$codebase_path = (Join-Path $env:WORKSPACE $codebase_fullname)
Write-Host "codebase_path = $codebase_path"
$log_dir = (Join-Path $env:WORKSPACE "mmdeploy_regression_working_dir\$codebase\$env:CUDA_VERSION")
Write-Host "log_dir = $log_dir"
InitMim $codebase $codebase_fullname
pip uninstall mmcv-full -y
pip uninstall $codebase -y
if ($codebase -eq "mmdet3d")
{
    mim install mmcv-full==1.5.2
}
else
{
    mim install mmcv-full==1.6.0
}
mim install $codebase
pip install -v $codebase_path
python ./tools/regression_test.py `
    --codebase $codebase `
    --device cuda:0 `
    --backends tensorrt onnxruntime `
    --work-dir $log_dir `
    $exec_performance
if ($LASTEXITCODE -ne 0) {
    throw "regression_test failed"
}
