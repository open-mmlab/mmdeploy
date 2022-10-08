param(
    $codebase,
    $exec_performance,
    $codebase_fullname_opt
)
cd C:\Users\HZJ\Desktop\mmdeploy_windows\MMDeploy
Write-Host "exec_path:  $pwd"
Write-Host "mim install $codebase"
$codebase_path = (Join-Path $env:WORKSPACE $codebase_fullname_opt.([string]$codebase))
Write-Host "codebase_path = $codebase_path"
mim uninstall mmcv
if ($codebase -eq "mmdet3d")
{
    # mim install $codebase
    mim install mmcv-full==1.5.2
    pip install -v $codebase_path
}
elseif ($codebase -eq "mmedit")
{
    # mim install $codebase
    mim install mmcv-full==1.6.0
    pip install -v $codebase_path
}
elseif ($codebase -eq "mmedit")
{
    # mim install $codebase
    mim install mmcv-full==1.6.0
    pip install -v $codebase_path
}
else
{
    # mim install $codebase
    if ( -not $?)
    {
        mim install mmcv-full
        pip install -v $codebase_path
    }
}
python ./tools/regression_test.py `
    --codebase $codebase `
    --device cuda:0 `
    --backends tensorrt onnxruntime `
    $exec_performance