param(
    $codebase,
    $exec_performance,
    $codebase_path
)

Write-Host "exec_path:  $pwd"
Write-Host "mim install $codebase"
Write-Host "codebase_path = $codebase_path"
$codebase_path = (Join-Path $env:WORKSPACE $codebase_path)
Write-Host "codebase_path = $codebase_path"
mim uninstall mmcv
if ($codebase -eq "mmdet3d")
{
    # mim install $codebase
    mim install mmcv-full==1.5.2

}
else
{
    # mim install $codebase
    mim install mmcv-full==1.6.0
}
pip install -v $codebase_path
python $env:MMDEPLOY_DIR/tools/regression_test.py `
    --codebase $codebase `
    --device cuda:0 `
    --backends tensorrt onnxruntime `
    $exec_performance